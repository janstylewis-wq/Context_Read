"""User accounts for the web app: email code / magic link / password sign-in, password reset,
OAuth account linking, profile and free-plan usage. The web BFF calls these through /internal."""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from sqlalchemy import Engine, delete, func, select

from . import mailer
from .config import get_settings
from .db import session_scope
from .models import Account, AuthCode, ChatSession, File, UsageEvent, User

CODE_TTL = timedelta(minutes=10)
RESET_TTL = timedelta(hours=1)
RESEND_AFTER = timedelta(seconds=30)
MAX_ATTEMPTS = 5
MIN_PASSWORD = 8
UNLIMITED = 2**53 - 1  # how the reference app reports a limit that does not apply (Number.MAX_SAFE_INTEGER)


def page_limit() -> int | None:
    """Pages a user may hold (PAGE_LIMIT; 0 or less = no limit)."""
    n = get_settings().page_limit
    return n if n > 0 else None


def chat_limit() -> int | None:
    n = get_settings().chat_limit
    return n if n > 0 else None


class AccountError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def _now() -> datetime:
    return datetime.now(UTC)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if "@" not in email or len(email) > 254 or " " in email:
        raise AccountError(400, "invalid_email", "Enter a valid email address.")
    return email


# ---------- passwords (stdlib scrypt) ----------

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32
        )
    except ValueError:
        return False
    return hmac.compare_digest(got.hex(), digest)


def _check_password_rules(password: str) -> None:
    if len(password or "") < MIN_PASSWORD:
        raise AccountError(400, "weak_password", f"Use at least {MIN_PASSWORD} characters.")


# ---------- user lookup ----------


def user_out(u: User) -> dict:
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "image": u.image,
        "hasPassword": bool(u.password_hash),
        "emailVerified": u.email_verified is not None,
        "createdAt": u.created_at.isoformat(),
        "planName": None,
        "planStatus": None,
        "userType": u.user_type,
        "isAdmin": u.is_admin,
        "allowDataCollection": u.allow_data_collection,
    }


def _get_or_create(s, email: str, verified: bool) -> User:
    user = s.scalars(select(User).where(User.email == email)).first()
    if user is None:
        user = User(email=email, name=email.split("@")[0])
        s.add(user)
    if verified and user.email_verified is None:
        user.email_verified = _now()
    s.flush()
    s.refresh(user)
    return user


def get_user(user_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        u = s.get(User, user_id)
        if u is None:
            raise AccountError(404, "user_not_found", "User not found.")
        return user_out(u)


# ---------- email code + magic link ----------


def request_login_code(email: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        last = s.scalars(
            select(AuthCode)
            .where(AuthCode.email == email, AuthCode.purpose == "login")
            .order_by(AuthCode.created_at.desc())
            .limit(1)
        ).first()
        if last is not None and _now() - last.created_at < RESEND_AFTER:
            raise AccountError(
                429, "too_soon", "A code was just sent. Wait a few seconds before asking again."
            )
        code = f"{secrets.randbelow(1_000_000):06d}"
        token = secrets.token_urlsafe(32)
        s.add(
            AuthCode(
                email=email,
                purpose="login",
                code_hash=_sha(f"{email}:{code}"),
                link_hash=_sha(token),
                expires_at=_now() + CODE_TTL,
            )
        )
    link = f"{get_settings().web_url}/magic-link?{urlencode({'email': email, 'token': token})}"
    mailer.send(
        email,
        f"Your ContextRead sign-in code: {code}",
        f"Your sign-in code is {code}. It expires in 10 minutes.\n\nOr open this link to sign in:\n{link}\n\n"
        "If you did not ask to sign in, ignore this email.",
        meta={"kind": "login", "code": code, "link": link},
    )
    return {"sent": True, "expires_in": int(CODE_TTL.total_seconds())}


def _consume(s, email: str, purpose: str, *, code: str | None = None, token: str | None = None) -> AuthCode:
    row = s.scalars(
        select(AuthCode)
        .where(
            AuthCode.email == email,
            AuthCode.purpose == purpose,
            AuthCode.consumed_at.is_(None),
            AuthCode.expires_at > func.now(),
        )
        .order_by(AuthCode.created_at.desc())
        .limit(1)
        .with_for_update()
    ).first()
    invalid = AccountError(400, "invalid_code", "That code is wrong or has expired. Ask for a new one.")
    if row is None or row.attempts >= MAX_ATTEMPTS:
        raise invalid
    ok = (
        code is not None and row.code_hash and hmac.compare_digest(row.code_hash, _sha(f"{email}:{code}"))
    ) or (token is not None and hmac.compare_digest(row.link_hash, _sha(token)))
    if not ok:
        row.attempts += 1
        s.commit()  # keep the failed attempt even though the caller's transaction rolls back
        raise invalid
    row.consumed_at = _now()
    return row


def verify_login_code(email: str, code: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        _consume(s, email, "login", code=(code or "").strip())
        return user_out(_get_or_create(s, email, verified=True))


def verify_magic_link(email: str, token: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        _consume(s, email, "login", token=token)
        return user_out(_get_or_create(s, email, verified=True))


# ---------- passwords ----------


def password_login(email: str, password: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        u = s.scalars(select(User).where(User.email == email)).first()
        # verify against a dummy hash when the user is missing, so timing does not reveal accounts
        stored = u.password_hash if u is not None else hash_password(secrets.token_hex(8))
        if not verify_password(password or "", stored) or u is None:
            raise AccountError(401, "invalid_credentials", "Invalid email or password.")
        return user_out(u)


def request_password_reset(email: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        exists = s.scalar(select(func.count()).select_from(User).where(User.email == email))
        if exists:
            token = secrets.token_urlsafe(32)
            s.add(
                AuthCode(email=email, purpose="reset", link_hash=_sha(token), expires_at=_now() + RESET_TTL)
            )
    if exists:
        link = f"{get_settings().web_url}/reset-password?{urlencode({'email': email, 'token': token})}"
        mailer.send(
            email,
            "Reset your ContextRead password",
            f"Open this link to choose a new password (valid for 1 hour):\n{link}\n\n"
            "If you did not ask for this, ignore this email.",
            meta={"kind": "reset", "link": link},
        )
    return {"sent": True}  # the same answer either way: no account enumeration


def reset_password(email: str, token: str, new_password: str, engine: Engine | None = None) -> dict:
    email = normalize_email(email)
    _check_password_rules(new_password)
    with session_scope(engine) as s:
        _consume(s, email, "reset", token=token)
        u = _get_or_create(s, email, verified=True)
        u.password_hash = hash_password(new_password)
        return user_out(u)


def set_password(
    user_id: str, new_password: str, current_password: str | None = None, engine: Engine | None = None
) -> dict:
    _check_password_rules(new_password)
    with session_scope(engine) as s:
        u = s.get(User, user_id)
        if u is None:
            raise AccountError(404, "user_not_found", "User not found.")
        if u.password_hash and not verify_password(current_password or "", u.password_hash):
            raise AccountError(400, "wrong_password", "The current password is wrong.")
        u.password_hash = hash_password(new_password)
        return user_out(u)


# ---------- OAuth ----------


def oauth_login(
    provider: str,
    provider_account_id: str,
    email: str,
    name: str | None = None,
    image: str | None = None,
    engine: Engine | None = None,
) -> dict:
    email = normalize_email(email)
    with session_scope(engine) as s:
        account = s.scalars(
            select(Account).where(
                Account.provider == provider, Account.provider_account_id == provider_account_id
            )
        ).first()
        if account is not None:
            return user_out(s.get(User, account.user_id))
        u = _get_or_create(s, email, verified=True)  # providers return verified emails
        u.name, u.image = u.name or name, u.image or image
        s.add(Account(user_id=u.id, type="oauth", provider=provider, provider_account_id=provider_account_id))
        return user_out(u)


# ---------- profile & usage ----------


def update_profile(
    user_id: str,
    name: str | None = None,
    allow_data_collection: bool | None = None,
    engine: Engine | None = None,
) -> dict:
    with session_scope(engine) as s:
        u = s.get(User, user_id)
        if u is None:
            raise AccountError(404, "user_not_found", "User not found.")
        if name is not None:
            name = name.strip()
            if not name or len(name) > 80:
                raise AccountError(400, "invalid_name", "Names are 1-80 characters.")
            u.name = name
        if allow_data_collection is not None:
            u.allow_data_collection = allow_data_collection
        s.flush()
        s.refresh(u)
        return user_out(u)


def pages_used(s, owner: tuple[str, str]) -> int:
    return int(
        s.scalar(
            select(func.coalesce(func.sum(File.page_num), 0)).where(
                File.owner_kind == owner[0], File.owner_id == owner[1], File.status != "failed"
            )
        )
    )


def usage(user_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        pages = pages_used(s, ("user", user_id))
        chats = int(
            s.scalar(
                select(func.coalesce(func.sum(UsageEvent.qty), 0)).where(
                    UsageEvent.owner_kind == "user", UsageEvent.owner_id == user_id, UsageEvent.kind == "chat"
                )
            )
        )
    page_cap, chat_cap = page_limit() or UNLIMITED, chat_limit() or UNLIMITED
    return {
        "page": {"total": pages, "limit": page_cap, "available": max(page_cap - pages, 0)},
        "chat": {
            "total": chats,
            "limit": chat_cap,
            "available": max(chat_cap - chats, 0),
            "percentage": round(chats / chat_cap, 4),
            "shouldShowWarning": chats >= 0.8 * chat_cap,
        },
        "isPro": False,
        "hasSubscription": False,
        "subscriptionStatus": None,
        "canPurchase": False,
    }


def delete_chat_history(user_id: str, engine: Engine | None = None) -> int:
    with session_scope(engine) as s:
        return s.execute(delete(ChatSession).where(ChatSession.user_id == user_id)).rowcount

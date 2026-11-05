"""API keys. The secret is shown once at creation; only its SHA-256 and a display prefix are stored."""

import hashlib
import secrets
from dataclasses import dataclass

from sqlalchemy import Engine, func, select, update

from .db import session_scope
from .models import ApiKey, User

KEY_PREFIXES = {"cloud": "cr_live_", "search": "crs_live_"}


@dataclass(frozen=True)
class Principal:
    user_id: str
    key_id: str
    product: str

    @property
    def owner(self) -> tuple[str, str]:
        return "user", self.user_id


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def ensure_user(email: str, name: str | None = None, engine: Engine | None = None) -> str:
    with session_scope(engine) as s:
        user = s.scalars(select(User).where(User.email == email.lower())).first()
        if user is None:
            user = User(email=email.lower(), name=name)
            s.add(user)
            s.flush()
        return user.id


def create_api_key(
    user_id: str, product: str = "cloud", name: str | None = None, engine: Engine | None = None
) -> tuple[str, str]:
    """Returns (secret, key_id). The secret cannot be recovered later."""
    secret = KEY_PREFIXES[product] + secrets.token_urlsafe(32)
    with session_scope(engine) as s:
        key = ApiKey(
            user_id=user_id, product=product, name=name, prefix=secret[:12], secret_sha256=_digest(secret)
        )
        s.add(key)
        s.flush()
        return secret, key.id


def authenticate(secret: str | None, product: str, engine: Engine | None = None) -> Principal | None:
    if not secret or not secret.startswith(KEY_PREFIXES[product]):
        return None
    with session_scope(engine) as s:
        key = s.scalars(
            select(ApiKey).where(
                ApiKey.secret_sha256 == _digest(secret),
                ApiKey.product == product,
                ApiKey.revoked_at.is_(None),
            )
        ).first()
        if key is None:
            return None
        s.execute(update(ApiKey).where(ApiKey.id == key.id).values(last_used_at=func.now()))
        return Principal(user_id=key.user_id, key_id=key.id, product=product)


def revoke_api_key(key_id: str, engine: Engine | None = None) -> None:
    with session_scope(engine) as s:
        s.execute(update(ApiKey).where(ApiKey.id == key_id).values(revoked_at=func.now()))

"""OAuth 2.1 for MCP clients (Claude Desktop, the @pageindex/mcp connector and other OAuth-only
clients): dynamic client registration, authorization code with PKCE, refresh-token rotation and
revocation. The MCP SDK serves the endpoints (/register, /authorize, /token, /revoke and the
discovery metadata); this module stores their state.

/authorize parks the request and sends the browser to the web app's consent page; the web app
(acting for the signed-in user) approves or denies it through /internal, which issues the code and
returns the client's redirect URL. Codes and tokens are stored only as SHA-256 hashes."""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import anyio
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from sqlalchemy import Engine, func, select, update

from ..config import get_settings
from ..db import session_scope
from ..ids import new_id
from ..models import OAuthClient, OAuthCode, OAuthRequest, User
from ..models import OAuthToken as TokenRow

SCOPE = "mcp:access"
REQUEST_TTL = timedelta(minutes=10)
CODE_TTL = timedelta(minutes=5)
ACCESS_TTL = timedelta(hours=1)
REFRESH_TTL = timedelta(days=30)
ACCESS_PREFIX, REFRESH_PREFIX = "cro_at_", "cro_rt_"


class ConsentError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _scopes(requested: list[str] | None) -> list[str]:
    # one scope covers the whole MCP surface; clients that ask for none get it
    return [SCOPE]


# ---------- consent (called by the web app through /internal) ----------


def _pending(s, request_id: str) -> OAuthRequest:
    req = s.get(OAuthRequest, request_id)
    if req is None or req.expires_at < _now():
        raise ConsentError(
            404, "request_not_found", "This sign-in request has expired. Start again from your app."
        )
    if req.decided_at is not None:
        raise ConsentError(409, "request_decided", "This sign-in request was already answered.")
    return req


def describe_request(request_id: str, engine: Engine | None = None) -> dict:
    """What the consent page shows: which app asks, for what, and where it will be sent back."""
    with session_scope(engine) as s:
        req = _pending(s, request_id)
        info = s.get(OAuthClient, req.client_id).info
        redirect = req.params["redirect_uri"]
        return {
            "requestId": req.id,
            "clientName": info.get("client_name") or "An MCP client",
            "clientUri": info.get("client_uri"),
            "redirectHost": redirect.split("/")[2] if "://" in redirect else redirect,
            "scopes": req.params["scopes"],
            "expiresAt": req.expires_at.isoformat(),
        }


def decide(request_id: str, user_id: str, approve: bool, engine: Engine | None = None) -> dict:
    """Approve (issue a code bound to the user) or deny; returns the client redirect URL."""
    with session_scope(engine) as s:
        req = _pending(s, request_id)
        if s.get(User, user_id) is None:
            raise ConsentError(404, "user_not_found", "User not found.")
        req.decided_at = _now()
        p = req.params
        if not approve:
            url = construct_redirect_uri(
                p["redirect_uri"],
                error="access_denied",
                error_description="The user declined.",
                state=p["state"],
            )
            return {"redirectUrl": url}
        code = secrets.token_urlsafe(32)  # 256 bits
        s.add(
            OAuthCode(
                code_sha256=_sha(code),
                client_id=req.client_id,
                user_id=user_id,
                grant_id=new_id(),
                params=p,
                expires_at=_now() + CODE_TTL,
            )
        )
        return {"redirectUrl": construct_redirect_uri(p["redirect_uri"], code=code, state=p["state"])}


def connections(user_id: str, engine: Engine | None = None) -> dict:
    """Apps holding a live grant for this user (for Settings)."""
    with session_scope(engine) as s:
        rows = s.execute(
            select(TokenRow.client_id, func.min(TokenRow.created_at), func.max(TokenRow.created_at))
            .where(TokenRow.user_id == user_id, TokenRow.revoked_at.is_(None), TokenRow.expires_at > _now())
            .group_by(TokenRow.client_id)
        ).all()
        out = []
        for client_id, first, last in rows:
            info = s.get(OAuthClient, client_id).info
            out.append(
                {
                    "clientId": client_id,
                    "clientName": info.get("client_name") or "An MCP client",
                    "connectedAt": first.isoformat(),
                    "lastRefreshedAt": last.isoformat(),
                }
            )
        return {"connections": sorted(out, key=lambda c: c["connectedAt"], reverse=True)}


def disconnect(user_id: str, client_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        n = s.execute(
            update(TokenRow)
            .where(
                TokenRow.user_id == user_id, TokenRow.client_id == client_id, TokenRow.revoked_at.is_(None)
            )
            .values(revoked_at=_now())
        ).rowcount
    return {"revoked": n}


# ---------- the SDK's provider interface ----------


def _issue(
    s, client_id: str, user_id: str, grant_id: str, scopes: list[str], resource: str | None
) -> OAuthToken:
    access, refresh = ACCESS_PREFIX + secrets.token_urlsafe(32), REFRESH_PREFIX + secrets.token_urlsafe(32)
    for token, kind, ttl in ((access, "access", ACCESS_TTL), (refresh, "refresh", REFRESH_TTL)):
        s.add(
            TokenRow(
                token_sha256=_sha(token),
                kind=kind,
                client_id=client_id,
                user_id=user_id,
                grant_id=grant_id,
                scopes=scopes,
                resource=resource,
                expires_at=_now() + ttl,
            )
        )
    return OAuthToken(
        access_token=access,
        expires_in=int(ACCESS_TTL.total_seconds()),
        scope=" ".join(scopes),
        refresh_token=refresh,
    )


class CodeRow(AuthorizationCode):
    grant_id: str


class RefreshRow(RefreshToken):
    grant_id: str


class Provider:
    """OAuthAuthorizationServerProvider backed by Postgres. The SDK validates PKCE, redirect URIs,
    expiry and client authentication; this class persists and looks things up."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine

    async def _run(self, fn, *args):
        return await anyio.to_thread.run_sync(fn, *args)

    # clients

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        def load():
            with session_scope(self.engine) as s:
                row = s.get(OAuthClient, client_id)
                return OAuthClientInformationFull.model_validate(row.info) if row else None

        return await self._run(load)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        def save():
            with session_scope(self.engine) as s:
                s.add(OAuthClient(id=client_info.client_id, info=client_info.model_dump(mode="json")))

        await self._run(save)

    # authorization

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        def park():
            with session_scope(self.engine) as s:
                req = OAuthRequest(
                    client_id=client.client_id,
                    params={
                        "state": params.state,
                        "scopes": _scopes(params.scopes),
                        "code_challenge": params.code_challenge,
                        "redirect_uri": str(params.redirect_uri),
                        "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                        "resource": params.resource,
                    },
                    expires_at=_now() + REQUEST_TTL,
                )
                s.add(req)
                s.flush()
                return req.id

        request_id = await self._run(park)
        return f"{get_settings().web_url.rstrip('/')}/oauth/authorize?request={request_id}"

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str):
        def load():
            with session_scope(self.engine) as s:
                row = s.get(OAuthCode, _sha(authorization_code))
                if row is None or row.client_id != client.client_id or row.used_at is not None:
                    return None
                p = row.params
                return CodeRow(
                    code=authorization_code,
                    scopes=p["scopes"],
                    expires_at=row.expires_at.timestamp(),
                    client_id=row.client_id,
                    code_challenge=p["code_challenge"],
                    redirect_uri=p["redirect_uri"],
                    redirect_uri_provided_explicitly=p["redirect_uri_provided_explicitly"],
                    resource=p.get("resource"),
                    subject=row.user_id,
                    grant_id=row.grant_id,
                )

        return await self._run(load)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: CodeRow
    ):
        def exchange():
            with session_scope(self.engine) as s:
                # single use, even under concurrent exchanges
                used = s.execute(
                    update(OAuthCode)
                    .where(
                        OAuthCode.code_sha256 == _sha(authorization_code.code), OAuthCode.used_at.is_(None)
                    )
                    .values(used_at=_now())
                ).rowcount
                if not used:
                    raise TokenError("invalid_grant", "authorization code already used")
                return _issue(
                    s,
                    client.client_id,
                    authorization_code.subject,
                    authorization_code.grant_id,
                    authorization_code.scopes,
                    authorization_code.resource,
                )

        return await self._run(exchange)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str):
        def load():
            with session_scope(self.engine) as s:
                row = s.get(TokenRow, _sha(refresh_token))
                if (
                    row is None
                    or row.kind != "refresh"
                    or row.client_id != client.client_id
                    or row.revoked_at is not None
                    or row.expires_at < _now()
                ):
                    return None
                return RefreshRow(
                    token=refresh_token,
                    client_id=row.client_id,
                    scopes=row.scopes,
                    expires_at=int(row.expires_at.timestamp()),
                    resource=row.resource,
                    subject=row.user_id,
                    grant_id=row.grant_id,
                )

        return await self._run(load)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshRow, scopes: list[str]
    ) -> OAuthToken:
        def rotate():
            with session_scope(self.engine) as s:
                # the old pair stops working; a replayed (stolen) refresh token finds nothing
                rotated = s.execute(
                    update(TokenRow)
                    .where(TokenRow.token_sha256 == _sha(refresh_token.token), TokenRow.revoked_at.is_(None))
                    .values(revoked_at=_now())
                ).rowcount
                if not rotated:
                    raise TokenError("invalid_grant", "refresh token already used")
                s.execute(
                    update(TokenRow)
                    .where(
                        TokenRow.grant_id == refresh_token.grant_id,
                        TokenRow.kind == "access",
                        TokenRow.revoked_at.is_(None),
                    )
                    .values(revoked_at=_now())
                )
                kept = [x for x in scopes if x in refresh_token.scopes] or refresh_token.scopes
                return _issue(
                    s,
                    client.client_id,
                    refresh_token.subject,
                    refresh_token.grant_id,
                    kept,
                    refresh_token.resource,
                )

        return await self._run(rotate)

    async def load_access_token(self, token: str) -> AccessToken | None:
        def load():
            with session_scope(self.engine) as s:
                row = s.get(TokenRow, _sha(token))
                if (
                    row is None
                    or row.kind != "access"
                    or row.revoked_at is not None
                    or row.expires_at < _now()
                ):
                    return None
                return AccessToken(
                    token=token,
                    client_id=row.client_id,
                    scopes=row.scopes,
                    expires_at=int(row.expires_at.timestamp()),
                    resource=row.resource,
                    subject=row.user_id,
                )

        return await self._run(load)

    async def revoke_token(self, token) -> None:
        def revoke():
            with session_scope(self.engine) as s:
                row = s.get(TokenRow, _sha(token.token))
                if row is not None:
                    s.execute(
                        update(TokenRow)
                        .where(TokenRow.grant_id == row.grant_id, TokenRow.revoked_at.is_(None))
                        .values(revoked_at=_now())
                    )

        await self._run(revoke)

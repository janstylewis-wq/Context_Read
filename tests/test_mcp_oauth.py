"""OAuth for MCP clients, driven by the official MCP client's OAuth support (discovery, dynamic
registration, PKCE, token exchange) with the consent step answered through /internal, as the web
app's consent page does."""

import base64
import hashlib
import json
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from test_indexing import DISNEY

from contextread import auth, config
from contextread.api.main import app

WEB = "http://web.test"
REDIRECT = "http://localhost:8090/callback"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(engine, bucket):
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PUBLIC_API_URL", base)
        mp.setenv("WEB_URL", WEB)
        config.get_settings.cache_clear()
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(100):
            try:
                if httpx.get(f"{base}/healthz").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        uid = auth.ensure_user("oauth-mcp@contextread.test", engine=engine)
        internal = httpx.Client(
            base_url=f"{base}/internal",
            headers={"x-internal-token": config.get_settings().internal_api_token},
        )
        yield {"base": base, "mcp": f"{base}/mcp", "uid": uid, "internal": internal}
        srv.should_exit = True
    config.get_settings.cache_clear()


def consent(server, consent_url: str, approve: bool = True) -> str:
    """What the consent page does: show the request, then answer it for the signed-in user."""
    assert consent_url.startswith(f"{WEB}/oauth/authorize?request=")
    request_id = parse_qs(urlparse(consent_url).query)["request"][0]
    shown = server["internal"].get(f"/oauth/requests/{request_id}").json()
    assert shown["scopes"] == ["mcp:access"] and shown["redirectHost"] == "localhost:8090"
    decided = server["internal"].post(
        f"/oauth/requests/{request_id}/decision", json={"owner_id": server["uid"], "approve": approve}
    )
    assert decided.status_code == 200, decided.text
    return decided.json()["redirectUrl"]


class MemoryStorage:
    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client: OAuthClientInformationFull | None = None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client

    async def set_client_info(self, client_info):
        self.client = client_info


def test_official_client_connects_through_oauth(server):
    storage, seen = MemoryStorage(), {}

    async def redirect_handler(authorize_url: str) -> None:
        # /authorize parks the request and sends the browser to the web app's consent page
        r = httpx.get(authorize_url, follow_redirects=False)
        assert r.status_code == 302, r.text
        seen["redirect"] = consent(server, r.headers["location"])

    async def callback_handler() -> AuthorizationCodeResult:
        q = parse_qs(urlparse(seen["redirect"]).query)
        return AuthorizationCodeResult(code=q["code"][0], state=q.get("state", [None])[0])

    provider = OAuthClientProvider(
        server_url=server["mcp"],
        client_metadata=OAuthClientMetadata(
            client_name="Test MCP client",
            redirect_uris=[REDIRECT],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            scope="mcp:access",
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    async def main():
        http = httpx2.AsyncClient(auth=provider, timeout=60)
        async with http, Client(streamable_http_client(server["mcp"], http_client=http)) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            listing = await client.call_tool("browse_documents", {})
            return names, listing

    names, listing = anyio.run(main)
    assert "get_page_content" in names and not listing.is_error
    assert storage.tokens.access_token.startswith("cro_at_") and storage.tokens.refresh_token.startswith(
        "cro_rt_"
    )
    assert storage.client.client_name == "Test MCP client"

    # Settings lists the connected app; disconnecting revokes its tokens
    conns = (
        server["internal"].get("/oauth/connections", params={"owner_id": server["uid"]}).json()["connections"]
    )
    assert [c["clientName"] for c in conns] == ["Test MCP client"]
    bearer = {"authorization": f"Bearer {storage.tokens.access_token}"}
    assert list_tools(server, bearer).status_code == 200
    server["internal"].delete(
        f"/oauth/connections/{storage.client.client_id}", params={"owner_id": server["uid"]}
    )
    assert list_tools(server, bearer).status_code == 401


def list_tools(server, headers: dict) -> httpx.Response:
    return httpx.post(
        server["mcp"],
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={
            **headers,
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": "2025-06-18",
        },
    )


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def register(server) -> str:
    r = httpx.post(
        f"{server['base']}/register",
        json={
            "client_name": "Raw client",
            "redirect_uris": [REDIRECT],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["client_id"]


def authorize(server, client_id: str, challenge: str, state: str = "st-1") -> str:
    r = httpx.get(
        f"{server['base']}/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": "mcp:access",
        },
        follow_redirects=False,
    )
    assert r.status_code == 302, r.text
    return r.headers["location"]


def test_codes_are_single_use_tokens_rotate_and_revoke(server):
    client_id = register(server)
    verifier, challenge = pkce()
    redirect = consent(server, authorize(server, client_id, challenge))
    q = parse_qs(urlparse(redirect).query)
    assert q["state"] == ["st-1"]
    code = q["code"][0]
    token_url = f"{server['base']}/token"
    exchange = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    wrong = httpx.post(token_url, data={**exchange, "code_verifier": "x" * 50})
    assert wrong.status_code == 400  # PKCE
    first = httpx.post(token_url, data=exchange)
    assert first.status_code == 200, first.text
    assert httpx.post(token_url, data=exchange).status_code == 400  # a code works once
    tokens = first.json()
    assert tokens["token_type"] == "Bearer" and tokens["scope"] == "mcp:access"

    old_access = {"authorization": f"Bearer {tokens['access_token']}"}
    assert list_tools(server, old_access).status_code == 200
    refresh = {
        "grant_type": "refresh_token",
        "refresh_token": tokens["refresh_token"],
        "client_id": client_id,
    }
    rotated = httpx.post(token_url, data=refresh)
    assert rotated.status_code == 200, rotated.text
    assert list_tools(server, old_access).status_code == 401  # rotation retires the old pair
    assert httpx.post(token_url, data=refresh).status_code == 400  # a replayed refresh token
    new_access = {"authorization": f"Bearer {rotated.json()['access_token']}"}
    assert list_tools(server, new_access).status_code == 200

    revoked = httpx.post(
        # the SDK expects the field even for public clients, which have no secret
        f"{server['base']}/revoke",
        data={"token": rotated.json()["refresh_token"], "client_id": client_id, "client_secret": ""},
    )
    assert revoked.status_code == 200, revoked.text
    assert list_tools(server, new_access).status_code == 401  # revoking either token ends the grant


def test_denied_and_answered_requests(server):
    client_id = register(server)
    _, challenge = pkce()
    consent_url = authorize(server, client_id, challenge, state="st-2")
    redirect = consent(server, consent_url, approve=False)
    q = parse_qs(urlparse(redirect).query)
    assert q["error"] == ["access_denied"] and q["state"] == ["st-2"] and "code" not in q
    request_id = parse_qs(urlparse(consent_url).query)["request"][0]
    again = server["internal"].post(
        f"/oauth/requests/{request_id}/decision", json={"owner_id": server["uid"], "approve": True}
    )
    assert again.status_code == 409
    assert server["internal"].get("/oauth/requests/nope").status_code == 404


def test_local_upload_tools_for_the_connector(server, engine):
    key, _ = auth.create_api_key(server["uid"], "cloud", engine=engine)
    headers = {"authorization": f"Bearer {key}"}

    async def main():
        out = {}
        for url in (server["mcp"], f"{server['mcp']}?local_upload=1"):
            http = httpx2.AsyncClient(headers=headers, timeout=60)
            async with http, Client(streamable_http_client(url, http_client=http)) as client:
                out[url] = [t.name for t in (await client.list_tools()).tools]
                if "local_upload" in url:
                    signed = await client.call_tool(
                        "get_signed_upload_url", {"fileName": "uploaded.pdf", "fileType": "application/pdf"}
                    )
                    info = json.loads(signed.content[0].text)
                    put = httpx.put(
                        info["upload_url"],
                        content=DISNEY.read_bytes(),
                        headers={"content-type": "application/pdf"},  # as the connector sends it
                    )
                    assert put.status_code == 200
                    submitted = await client.call_tool("submit_document", {"file_name": info["file_name"]})
                    out["submitted"] = json.loads(submitted.content[0].text)
                    out["doc"] = await client.call_tool("get_document", {"doc_name": "uploaded.pdf"})
        return out

    out = anyio.run(main)
    plain, local = out[server["mcp"]], out[f"{server['mcp']}?local_upload=1"]
    assert "submit_document" not in plain and {"get_signed_upload_url", "submit_document"} <= set(local)
    assert out["submitted"]["doc_name"] == "uploaded.pdf" and out["submitted"]["page_count"] == 22
    assert not out["doc"].is_error

"""MCP server: streamable HTTP at /mcp. Clients authenticate with `Authorization: Bearer <token>`,
where the token is a cloud API key (the reference product's developer setup) or an OAuth access
token (OAuth-only clients such as Claude Desktop and the @pageindex/mcp connector; see oauth.py).
Tool names, input schemas, descriptions and annotations are served verbatim from the reference
contract (contract.json, MIT, from the PageIndex repository), plus get_document_image, which its
tools point to. With `?local_upload=1` (the connector's URL) the list also carries the two upload
tools its process_document uses."""

import json
import logging
from contextlib import asynccontextmanager
from importlib.resources import files as package_files

import anyio
import mcp_types as types
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Receive, Scope, Send

from .. import auth
from ..config import get_settings
from . import oauth, tools

log = logging.getLogger(__name__)

SCOPE = oauth.SCOPE
CONTRACT = json.loads(package_files("contextread.mcp").joinpath("contract.json").read_text())
EXTRA_TOOLS = {
    "get_document_image": {
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
        "description": "Fetch a rendered page as a PNG image. Use it for scanned, layout-heavy or chart-heavy "
        "pages whose text alone is not enough. Pass an `image_path` exactly as returned in "
        "get_page_content()'s `page_images` array; never construct one.",
        "schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Copy an `image_path` verbatim from get_page_content()'s `page_images`.",
                }
            },
            "required": ["image_path"],
        },
    }
}
INSTRUCTIONS = (
    "ContextRead gives you the user's documents as hierarchical indexes. Find documents with "
    "browse_documents(), check readiness with get_document(), read the outline with "
    "get_document_structure() for long documents, then read only the pages you need with "
    "get_page_content(). Cite the page numbers you read."
)


UPLOAD_TOOLS = {
    "get_signed_upload_url": {
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        "description": "Get a presigned URL to PUT a local PDF to. Used by the local-upload client; pass the "
        "returned `file_name` to submit_document() after uploading.",
        "schema": {
            "type": "object",
            "properties": {
                "fileName": {"type": "string", "minLength": 1, "description": "The PDF's file name."},
                "fileType": {"type": "string", "enum": ["application/pdf"], "default": "application/pdf"},
            },
            "required": ["fileName"],
        },
    },
    "submit_document": {
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        "description": "Submit an uploaded PDF for indexing. Pass `file_name` verbatim from "
        "get_signed_upload_url().",
        "schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string", "minLength": 1},
                "folder_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["file_name"],
        },
    },
}


def tool_list(specs: dict) -> list[types.Tool]:
    return [
        types.Tool.model_validate(
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["schema"],
                "annotations": spec["annotations"],
            }
        )
        for name, spec in specs.items()
    ]


TOOLS = tool_list({**CONTRACT["tools"], **EXTRA_TOOLS})
TOOLS_WITH_UPLOAD = tool_list({**CONTRACT["tools"], **EXTRA_TOOLS, **UPLOAD_TOOLS})


def _error(code: str, message: str) -> types.CallToolResult:
    body = json.dumps({"success": False, "error": code, "message": message}, ensure_ascii=False)
    return types.CallToolResult(content=[types.TextContent(text=body)], is_error=True)


async def _list_tools(ctx, params) -> types.ListToolsResult:
    request = getattr(ctx, "request", None)
    local_upload = request is not None and request.query_params.get("local_upload") == "1"
    return types.ListToolsResult(tools=TOOLS_WITH_UPLOAD if local_upload else TOOLS)


async def _call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    token = get_access_token()
    if token is None or not token.subject:
        return _error("unauthorized", "Send an API key or OAuth token as `Authorization: Bearer <token>`.")
    owner = ("user", token.subject)
    name, args = params.name, params.arguments or {}
    try:
        out = await anyio.to_thread.run_sync(lambda: tools.call(owner, name, args))
    except tools.ToolFailure as e:
        return _error(e.code, str(e))
    except KeyError as e:
        return _error("invalid_arguments", f"Missing required argument {e.args[0]!r}.")
    except Exception:
        log.exception("MCP tool %s failed", name)
        return _error("internal_error", "The tool failed unexpectedly. Try again.")
    if name == "get_document_image":
        data, caption = out
        return types.CallToolResult(
            content=[types.ImageContent(data=data, mime_type="image/png"), types.TextContent(text=caption)]
        )
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(out, ensure_ascii=False))])


class BearerVerifier:
    """Bearer tokens are cloud API keys (cr_live_...) or OAuth access tokens, both checked against
    stored hashes."""

    def __init__(self, provider: oauth.Provider):
        self.provider = provider

    async def verify_token(self, token: str) -> AccessToken | None:
        if token.startswith(oauth.ACCESS_PREFIX):
            return await self.provider.load_access_token(token)
        principal = await anyio.to_thread.run_sync(auth.authenticate, token, "cloud")
        if principal is None:
            return None
        return AccessToken(token=token, client_id=principal.key_id, scopes=[SCOPE], subject=principal.user_id)


server = Server(
    "contextread", version="0.5.0", title="ContextRead", instructions=INSTRUCTIONS,
    on_list_tools=_list_tools, on_call_tool=_call_tool,
)  # fmt: skip


class McpApp:
    """ASGI entry for the MCP routes. The SDK's session manager runs once per instance, so each
    start of the API (the lifespan) builds a fresh one."""

    def __init__(self):
        self._app = None

    @asynccontextmanager
    async def lifespan(self):
        base = get_settings().public_api_url.rstrip("/")
        provider = oauth.Provider()
        app = server.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            # every request must carry a token, so a rebinding page could not use it anyway; and
            # the server is reached under other host names behind a proxy
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
            # this server is its own authorization server; API keys are not issued per resource, so
            # there is no audience to check
            auth=AuthSettings(
                issuer_url=base,
                resource_server_url=f"{base}/mcp",
                required_scopes=[SCOPE],
                validate_token_resource=False,
                client_registration_options=ClientRegistrationOptions(
                    enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
                ),
                revocation_options=RevocationOptions(enabled=True),
            ),
            auth_server_provider=provider,
            token_verifier=BearerVerifier(provider),
        )
        manager = server.session_manager  # the one just built for `app`
        self._app = app
        async with manager.run():
            yield
        # a server shutting down must not unset the app a newer start installed
        if self._app is app:
            self._app = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._app is None:
            raise RuntimeError("the MCP app is used outside the API lifespan")
        await self._app(scope, receive, send)


mcp_app = McpApp()

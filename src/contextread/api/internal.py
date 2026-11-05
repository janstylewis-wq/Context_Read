"""Service-to-service routes for the web BFF. Every request must carry X-Internal-Token.

The BFF authenticates the browser session and passes the acting user's id; these routes still
check that every document and folder belongs to that owner."""

import hmac
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import accounts, citations, doctools, files, folders
from ..chat import sessions
from ..config import get_settings
from ..llm import BudgetedLLM
from ..mcp import oauth as mcp_oauth
from .deps import get_llm


def require_internal_token(x_internal_token: str = Header(default="")) -> None:
    expected = get_settings().internal_api_token
    if not expected or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="invalid internal token")


router = APIRouter(prefix="/internal", dependencies=[Depends(require_internal_token)])

OwnerKind = Literal["user", "teamspace"]
UPLOAD_STATUS = {
    "forbidden": 403,
    "upload_missing": 404,
    "document_not_found": 404,
    "folder_not_found": 404,
    "page_limit": 402,
}


def _fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _call(fn, *args, **kwargs):
    """Run a service call, mapping its domain errors to HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except accounts.AccountError as e:
        raise _fail(e.status, e.code, str(e)) from e
    except folders.FolderError as e:
        raise _fail(e.status, e.code, str(e)) from e
    except files.UploadError as e:
        raise _fail(UPLOAD_STATUS.get(e.code, 422), e.code, str(e)) from e
    except doctools.ToolError as e:
        raise _fail(404, "not_found", str(e)) from e
    except sessions.ChatError as e:
        raise _fail(e.status, e.code, str(e)) from e
    except mcp_oauth.ConsentError as e:
        raise _fail(e.status, e.code, str(e)) from e


# ---------- auth ----------


class EmailIn(BaseModel):
    email: str


class CodeIn(EmailIn):
    code: str


class LinkIn(EmailIn):
    token: str


class PasswordLoginIn(EmailIn):
    password: str


class ResetIn(LinkIn):
    password: str


class OAuthIn(EmailIn):
    provider: str
    provider_account_id: str
    name: str | None = None
    image: str | None = None


@router.post("/auth/request-code")
def request_code(body: EmailIn):
    return _call(accounts.request_login_code, body.email)


@router.post("/auth/verify-code")
def verify_code(body: CodeIn):
    return _call(accounts.verify_login_code, body.email, body.code)


@router.post("/auth/verify-link")
def verify_link(body: LinkIn):
    return _call(accounts.verify_magic_link, body.email, body.token)


@router.post("/auth/password")
def password_login(body: PasswordLoginIn):
    return _call(accounts.password_login, body.email, body.password)


@router.post("/auth/request-reset")
def request_reset(body: EmailIn):
    return _call(accounts.request_password_reset, body.email)


@router.post("/auth/reset")
def reset(body: ResetIn):
    return _call(accounts.reset_password, body.email, body.token, body.password)


@router.post("/auth/oauth")
def oauth(body: OAuthIn):
    return _call(
        accounts.oauth_login, body.provider, body.provider_account_id, body.email, body.name, body.image
    )


# ---------- users ----------


class ProfileIn(BaseModel):
    name: str | None = None
    allow_data_collection: bool | None = None


class SetPasswordIn(BaseModel):
    new_password: str
    current_password: str | None = None


@router.get("/users/{user_id}")
def get_user(user_id: str):
    return _call(accounts.get_user, user_id)


@router.patch("/users/{user_id}")
def update_user(user_id: str, body: ProfileIn):
    return _call(accounts.update_profile, user_id, body.name, body.allow_data_collection)


@router.post("/users/{user_id}/password")
def set_password(user_id: str, body: SetPasswordIn):
    return _call(accounts.set_password, user_id, body.new_password, body.current_password)


@router.get("/users/{user_id}/usage")
def usage(user_id: str):
    return _call(accounts.usage, user_id)


@router.delete("/users/{user_id}/files")
def delete_all_files(user_id: str):
    return {"deleted": _call(files.delete_all, ("user", user_id))}


@router.delete("/users/{user_id}/chats")
def delete_all_chats(user_id: str):
    return {"deleted": _call(accounts.delete_chat_history, user_id)}


# ---------- documents ----------


class UploadUrlIn(BaseModel):
    owner_id: str
    file_name: str
    content_type: str = "application/pdf"


class RegisterIn(BaseModel):
    owner_kind: OwnerKind = "user"
    owner_id: str
    file_name: str
    storage_key: str
    folder_id: str | None = None
    metadata: dict[str, Any] | None = None
    created_by_user_id: str | None = None
    page_limit: int | None = None


class FileUpdateIn(BaseModel):
    owner_id: str
    file_name: str | None = None
    move: bool = False  # with move, folder_id (null = top level) is applied
    folder_id: str | None = None


class ThumbnailsIn(BaseModel):
    owner_id: str
    file_ids: list[str] = Field(max_length=100)


@router.post("/uploads")
def upload_url(body: UploadUrlIn):
    return files.create_upload(body.owner_id, body.file_name, body.content_type)


@router.post("/files", status_code=201)
def register(body: RegisterIn):
    """Uploads from the web app count against the user's plan limit unless a limit is given."""
    args = body.model_dump()
    if args["page_limit"] is None:
        args["page_limit"] = accounts.page_limit()
    return _call(files.register_upload, **args)


@router.get("/files")
def list_files(
    owner_id: str,
    owner_kind: OwnerKind = "user",
    folder_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    return files.list_web((owner_kind, owner_id), folder_id, min(limit, 200), offset)


def _record(file_id: str, owner_id: str | None) -> dict:
    record = files.file_record(file_id)
    if record is None or (owner_id is not None and record["owner_id"] != owner_id):
        raise _fail(404, "document_not_found", "Document not found.")
    return record


@router.get("/files/{file_id}")
def get_file(file_id: str, owner_id: str | None = None):
    return _record(file_id, owner_id)


@router.get("/files/{file_id}/pages/{page}")
def get_page(file_id: str, page: int, owner_id: str | None = None):
    _record(file_id, owner_id)
    content = files.page_content(file_id, page)
    if content is None:
        raise _fail(404, "page_not_found", "Page not found.")
    return content


@router.patch("/files/{file_id}")
def update_file(file_id: str, body: FileUpdateIn):
    folder = body.folder_id if body.move else files._UNSET
    return _call(files.update_file, ("user", body.owner_id), file_id, body.file_name, folder)


@router.delete("/files/{file_id}")
def delete_file(file_id: str, owner_id: str):
    _record(file_id, owner_id)
    files.delete_file(file_id)
    return {"deleted": True}


@router.post("/files/thumbnails")
def thumbnails(body: ThumbnailsIn):
    return files.thumbnails(("user", body.owner_id), body.file_ids)


@router.get("/files/{file_id}/download-url")
def download_url(file_id: str, owner_id: str):
    return {"url": _call(files.download_url, ("user", owner_id), file_id)}


# ---------- folders ----------


class FolderCreateIn(BaseModel):
    owner_id: str
    name: str
    parent_id: str | None = None


class FolderUpdateIn(BaseModel):
    owner_id: str
    name: str | None = None
    move: bool = False
    parent_id: str | None = None


@router.get("/folders")
def folder_tree(owner_id: str):
    return {"folders": folders.folder_tree(("user", owner_id))}


@router.post("/folders", status_code=201)
def create_folder(body: FolderCreateIn):
    return _call(folders.create_folder, ("user", body.owner_id), body.name, None, body.parent_id)["folder"]


@router.patch("/folders/{folder_id}")
def update_folder(folder_id: str, body: FolderUpdateIn):
    return _call(
        folders.update_folder, ("user", body.owner_id), folder_id, body.name, body.parent_id, body.move
    )


@router.delete("/folders/{folder_id}")
def delete_folder(folder_id: str, owner_id: str):
    return _call(folders.delete_folder, ("user", owner_id), folder_id)


# ---------- chat ----------


class ChatSendIn(BaseModel):
    owner_id: str
    session_id: str | None = None
    message: dict[str, Any] | None = None
    thinking: bool = False
    trigger: Literal["submit-message", "regenerate-message"] = "submit-message"
    message_id: str | None = None


class SessionUpdateIn(BaseModel):
    owner_id: str
    title: str


@router.post("/chat/send")
def chat_send(body: ChatSendIn, llm: BudgetedLLM = Depends(get_llm)):
    """Server-sent events, one AI SDK UI-message chunk per event. Checks happen before the stream
    starts, so a refused turn is a plain HTTP error."""
    turn = _call(
        sessions.start_turn,
        body.owner_id,
        body.message,
        llm,
        session_id=body.session_id,
        thinking=body.thinking,
        trigger=body.trigger,
        message_id=body.message_id,
    )

    def events():
        for chunk in turn.stream():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"cache-control": "no-cache"})


@router.get("/chat/sessions")
def chat_sessions(owner_id: str, limit: int = Query(50, ge=1, le=200)):
    return sessions.list_sessions(owner_id, limit)


@router.get("/chat/sessions/{session_id}")
def chat_session(session_id: str, owner_id: str, limit: int = 100):
    return _call(sessions.get_session, owner_id, session_id, limit)


@router.patch("/chat/sessions/{session_id}")
def update_chat_session(session_id: str, body: SessionUpdateIn):
    return _call(sessions.update_session, body.owner_id, session_id, body.title)


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: str, owner_id: str):
    return _call(sessions.delete_session, owner_id, session_id)


# ---------- citations ----------


class CitationItem(BaseModel):
    doc: str
    blockIds: list[str] = Field(default_factory=list, max_length=200)  # noqa: N815 (wire name)


class ResolveCitationsIn(BaseModel):
    owner_id: str
    items: list[CitationItem] = Field(max_length=50)


@router.post("/citations/resolve")
def resolve_citations(body: ResolveCitationsIn):
    return citations.resolve_citations(("user", body.owner_id), [i.model_dump() for i in body.items])


@router.get("/citations/page")
def citation_page(owner_id: str, doc: str, page: int):
    return _call(citations.citation_page, ("user", owner_id), doc, page)


# ---------- OAuth consent for MCP clients ----------


class ConsentIn(BaseModel):
    owner_id: str
    approve: bool


@router.get("/oauth/requests/{request_id}")
def oauth_request(request_id: str):
    return _call(mcp_oauth.describe_request, request_id)


@router.post("/oauth/requests/{request_id}/decision")
def oauth_decision(request_id: str, body: ConsentIn):
    return _call(mcp_oauth.decide, request_id, body.owner_id, body.approve)


@router.get("/oauth/connections")
def oauth_connections(owner_id: str):
    return mcp_oauth.connections(owner_id)


@router.delete("/oauth/connections/{client_id}")
def oauth_disconnect(client_id: str, owner_id: str):
    return mcp_oauth.disconnect(owner_id, client_id)

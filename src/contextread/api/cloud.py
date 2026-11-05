"""Cloud REST API, wire-compatible with the PageIndex SDK's CloudAPI client (header `api_key`).
Routes sit at the server root: /doc/, /docs/, /retrieval/, /chat/completions/, /folder(s)/."""

import io
import json
import queue
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import Engine, func, select

from .. import files, folders, jobs, storage
from ..auth import Principal
from ..chat.agent import run_chat
from ..citations import bbox_thousandths
from ..db import session_scope
from ..ids import new_id
from ..llm import BudgetedLLM, BudgetExceeded
from ..models import Block, Folder, Page, Retrieval
from ..models import File as FileRow
from ..retrieval import retrieval_out
from .deps import cloud_principal, get_db_engine, get_llm

router = APIRouter()


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _own_file(s, principal: Principal, doc_id: str) -> FileRow:
    f = s.get(FileRow, doc_id)
    if f is None or (f.owner_kind, f.owner_id) != principal.owner:
        raise _err(404, "document_not_found", "Document not found.")
    return f


def _json_field(value: str | None, name: str) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise _err(400, "invalid_request", f"{name} must be JSON") from e


def _doc_out(f: FileRow) -> dict:
    return {
        "id": f.id,
        "name": f.file_name,
        "description": f.description,
        "status": f.status,
        "createdAt": f.created_at.isoformat(),
        "pageNum": f.page_num or 0,
        "folderId": f.folder_id,
        "metadata": f.meta,
    }


# ---------- documents ----------


@router.post("/doc/")
async def submit_document(
    file: UploadFile = File(...),
    mode: str | None = Form(None),
    beta_headers: str | None = Form(None),
    folder_id: str | None = Form(None),
    metadata: str | None = Form(None),
    if_retrieval: str | None = Form(None),
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    meta = _json_field(metadata, "metadata")
    headers = _json_field(beta_headers, "beta_headers") or []
    try:
        folders.validate_metadata(principal.owner, folder_id, meta, engine)
    except folders.FolderError as e:
        raise _err(e.status, e.code, str(e)) from e
    data = await file.read()
    try:
        key = files.store_upload(principal.user_id, file.filename or "document.pdf", data)
        reg = files.register_upload(
            *principal.owner,
            file.filename or "document.pdf",
            key,
            folder_id=folder_id,
            metadata=meta,
            created_by_user_id=principal.user_id,
            engine=engine,
        )
    except files.UploadError as e:
        raise _err(404 if e.code == "folder_not_found" else 400, e.code, str(e)) from e
    with session_scope(engine) as s:
        f = s.get(FileRow, reg["doc_id"])
        f.config = {**(f.config or {}), "mode": mode, "beta_headers": headers}
    return {"doc_id": reg["doc_id"], "name": reg["name"]}


def _envelope(f: FileRow, result: Any) -> dict:
    done = f.status == "completed"
    return {
        "doc_id": f.id,
        "status": f.status,
        "retrieval_ready": done,
        "result": result if done else None,
        "metadata": f.meta,
        "features": {"block_reference": bool(f.has_block_ref)},
    }


def _tree_node(node: dict, summary: bool, text: bool) -> dict:
    out = {k: node.get(k) for k in ("title", "node_id", "start_index", "end_index")}
    if summary and node.get("summary") is not None:
        out["summary"] = node["summary"]
    if text:
        out["text"] = node.get("text", "")
    if node.get("nodes"):
        out["nodes"] = [_tree_node(c, summary, text) for c in node["nodes"]]
    return out


@router.get("/doc/{doc_id}/")
def get_document_result(
    doc_id: str,
    type: Literal["tree", "ocr"] = Query(...),
    format: Literal["page", "node", "raw"] = "page",
    summary: str = "false",
    include_text: str = "true",
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    with session_scope(engine) as s:
        f = _own_file(s, principal, doc_id)
        if f.status != "completed":
            return _envelope(f, None)
        if type == "tree":
            want_summary, want_text = summary.lower() == "true", include_text.lower() != "false"
            return _envelope(f, [_tree_node(n, want_summary, want_text) for n in f.structure or []])
        if format == "node":
            result = [
                {"title": m["title"], "level": m["level"], "page_index": m["start_index"], "text": m["text"]}
                for m in f.md_tree or []
            ]
        else:
            pages = s.scalars(select(Page).where(Page.file_id == f.id).order_by(Page.page))
            page_list = [{"page_index": p.page, "markdown": p.text or "", "read": p.read_mode} for p in pages]
            result = page_list if format == "page" else "\n\n".join(p["markdown"] for p in page_list)
        return _envelope(f, result)


@router.get("/doc/{doc_id}/metadata/")
def get_document(
    doc_id: str, principal: Principal = Depends(cloud_principal), engine: Engine = Depends(get_db_engine)
):
    with session_scope(engine) as s:
        return _doc_out(_own_file(s, principal, doc_id))


@router.delete("/doc/{doc_id}/")
def delete_document(
    doc_id: str, principal: Principal = Depends(cloud_principal), engine: Engine = Depends(get_db_engine)
):
    with session_scope(engine) as s:
        _own_file(s, principal, doc_id)
    files.delete_file(doc_id, engine=engine)
    return {"message": "Document deleted successfully."}


@router.get("/docs/")
def list_documents(
    limit: int = Query(50, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    folder_id: str | None = None,
    name: str | None = None,
    recursive: str = "false",
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    with session_scope(engine) as s:
        q = select(FileRow).where(
            FileRow.owner_kind == principal.owner[0], FileRow.owner_id == principal.owner[1]
        )
        if folder_id == "root" and recursive.lower() != "true":
            q = q.where(FileRow.folder_id.is_(None))
        elif folder_id not in (None, "", "root"):
            ids = [folder_id]
            if recursive.lower() == "true":
                frontier = [folder_id]
                while frontier:
                    frontier = list(s.scalars(select(Folder.id).where(Folder.parent_id.in_(frontier))))
                    ids += frontier
            q = q.where(FileRow.folder_id.in_(ids))
        if name is not None:
            q = q.where(FileRow.file_name == name)
        total = s.scalar(select(func.count()).select_from(q.subquery()))
        rows = s.scalars(q.order_by(FileRow.created_at.desc(), FileRow.id).limit(limit).offset(offset))
        documents = [
            {**_doc_out(f), "path": None, "features": {"block_reference": bool(f.has_block_ref)}}
            for f in rows
        ]
    return {"documents": documents, "total": total, "limit": limit, "offset": offset}


@router.get("/doc/{doc_id}/block/{block_id}/")
def get_block(
    doc_id: str,
    block_id: str,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    with session_scope(engine) as s:
        f = _own_file(s, principal, doc_id)
        b = s.get(Block, (f.id, block_id))
        if b is None:
            raise _err(404, "block_not_found", "The document has no such block.")
        page = s.get(Page, (f.id, b.page))
        return {
            "doc_id": f.id,
            "page": b.page,
            "block_id": b.block_id,
            "block_type": b.type,
            "bbox": bbox_thousandths(b.bbox, page.width, page.height),
            "content": b.text,
        }


@router.get("/doc/s3/{doc_id}/images")
def get_page_images(
    doc_id: str,
    start: int = Query(1, ge=1),
    end: int | None = None,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    with session_scope(engine) as s:
        f = _own_file(s, principal, doc_id)
        end = min(end or start, start + 49)
        pages = s.scalars(
            select(Page).where(Page.file_id == f.id, Page.page.between(start, end)).order_by(Page.page)
        )
        return {
            "images": [
                {"page": p.page, "url": storage.presign_get(p.image_key)} for p in pages if p.image_key
            ]
        }


@router.get("/doc/{doc_id}/image/{img_id}/")
def get_document_image(
    doc_id: str,
    img_id: str,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    """Embedded images are image blocks; the crop is cut from the page render on first request."""
    with session_scope(engine) as s:
        f = _own_file(s, principal, doc_id)
        b = s.get(Block, (f.id, img_id.removesuffix(".png")))
        if b is None or b.type != "image":
            raise _err(404, "image_not_found", "The document has no such image.")
        page = s.get(Page, (f.id, b.page))
        key = f"{f.owner_id}/{f.id}/images/{b.block_id}.png"
        page_key, bbox = page.image_key, b.bbox
    try:
        storage.get_client().head_object(Bucket=storage.bucket(), Key=key)
    except Exception:
        crop = Image.open(io.BytesIO(storage.get_bytes(page_key))).crop(tuple(bbox))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        storage.put_bytes(key, buf.getvalue(), "image/png")
    return {"url": storage.presign_get(key)}


# ---------- retrieval (deprecated in the reference API, kept for compatibility) ----------


class RetrievalIn(BaseModel):
    doc_id: str
    query: str
    thinking: bool = False


@router.post("/retrieval/")
def submit_query(
    body: RetrievalIn,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    if not body.query.strip():
        raise _err(400, "invalid_request", "query must not be empty")
    with session_scope(engine) as s:
        _own_file(s, principal, body.doc_id)
        r = Retrieval(
            user_id=principal.user_id, file_id=body.doc_id, query=body.query, thinking=body.thinking
        )
        s.add(r)
        s.flush()
        retrieval_id = r.id
    jobs.enqueue("retrieval", {"retrieval_id": retrieval_id}, engine=engine, max_attempts=2)
    return {"retrieval_id": retrieval_id}


@router.get("/retrieval/{retrieval_id}/")
def get_retrieval(
    retrieval_id: str,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    with session_scope(engine) as s:
        r = s.get(Retrieval, retrieval_id)
        if r is None or r.user_id != principal.user_id:
            raise _err(404, "retrieval_not_found", "Retrieval not found.")
        return retrieval_out(r)


# ---------- chat completions ----------


class ChatIn(BaseModel):
    messages: list[dict]
    stream: bool = False
    doc_id: str | list[str] | None = None
    folder_id: str | None = None
    temperature: float | None = None
    enable_citations: bool = False


def _completion(text: str, finish: str, prompt_tokens: int, completion_tokens: int, cid: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "contextread",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chunk(cid: str, content: str | None = None, finish: str | None = None, meta: dict | None = None) -> str:
    delta = {"content": content} if content is not None else {}
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "contextread",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if meta:
        body["block_metadata"] = meta
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


@router.post("/chat/completions/")
def chat_completions(
    body: ChatIn,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
    llm: BudgetedLLM = Depends(get_llm),
):
    if not body.messages or not any(m.get("role") == "user" for m in body.messages):
        raise _err(400, "invalid_request", "messages must include a user message")
    doc_ids = [body.doc_id] if isinstance(body.doc_id, str) else list(body.doc_id or [])
    cid = "chatcmpl-" + new_id()
    kwargs = dict(
        owner=principal.owner,
        messages=body.messages,
        llm=llm,
        doc_ids=doc_ids,
        folder_id=body.folder_id,
        enable_citations=body.enable_citations,
        temperature=body.temperature,
        engine=engine,
        purpose="chat.cloud_api",
    )
    if not body.stream:
        try:
            result = run_chat(**kwargs)
        except BudgetExceeded as e:
            raise _err(402, "insufficient_balance", str(e)) from e
        return _completion(
            result.text, result.finish_reason, result.prompt_tokens, result.completion_tokens, cid
        )

    events: queue.Queue = queue.Queue()

    def worker():
        try:
            result = run_chat(**kwargs, on_event=events.put)
            events.put({"type": "done", "finish": result.finish_reason})
        except Exception as e:  # reported in-stream; the client raises on an "error" chunk
            events.put({"type": "error", "error": f"{type(e).__name__}: {e}"})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            ev = events.get()
            if ev["type"] == "tool_call":
                yield _chunk(cid, meta={"type": "mcp_tool_use_start", "tool_name": ev["name"]})
                yield _chunk(cid, json.dumps(ev["arguments"]), meta={"type": "tool_use"})
                yield _chunk(cid, meta={"type": "tool_use_stop"})
            elif ev["type"] == "text":
                text = ev["text"]
                for i in range(0, len(text), 48):
                    yield _chunk(cid, text[i : i + 48], meta={"type": "text"})
            elif ev["type"] == "error":
                yield f"data: {json.dumps({'error': ev['error']})}\n\n"
                return
            elif ev["type"] == "done":
                yield _chunk(cid, finish=ev["finish"])
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------- folders & metadata schemas ----------


class FolderIn(BaseModel):
    name: str
    description: str | None = None
    parent_folder_id: str | None = None


class SchemaIn(BaseModel):
    fields: dict[str, Any]


def _folder_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except folders.FolderError as e:
        raise _err(e.status, e.code, str(e)) from e


@router.post("/folder/")
def create_folder(
    body: FolderIn, principal: Principal = Depends(cloud_principal), engine: Engine = Depends(get_db_engine)
):
    return _folder_call(
        folders.create_folder,
        principal.owner,
        body.name,
        body.description,
        body.parent_folder_id,
        engine=engine,
    )


@router.get("/folders/")
def list_folders(
    parent_folder_id: str | None = None,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    return _folder_call(folders.list_folders, principal.owner, parent_folder_id, engine=engine)


@router.put("/folder/{folder_id}/metadata/schema/")
def create_metadata_schema(
    folder_id: str,
    body: SchemaIn,
    principal: Principal = Depends(cloud_principal),
    engine: Engine = Depends(get_db_engine),
):
    return _folder_call(folders.create_schema, principal.owner, folder_id, body.fields, engine=engine)


@router.get("/folder/{folder_id}/metadata/schema/")
def get_metadata_schema(
    folder_id: str, principal: Principal = Depends(cloud_principal), engine: Engine = Depends(get_db_engine)
):
    return _folder_call(folders.get_schema, principal.owner, folder_id, engine=engine)


@router.delete("/folder/{folder_id}/metadata/schema/")
def delete_metadata_schema(
    folder_id: str, principal: Principal = Depends(cloud_principal), engine: Engine = Depends(get_db_engine)
):
    return _folder_call(folders.delete_schema, principal.owner, folder_id, engine=engine)

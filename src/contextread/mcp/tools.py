"""The MCP tools, as plain functions scoped to one owner. Names, arguments and descriptions follow
the reference product's frozen contract (contract.json); results are JSON objects shaped like the
recorded ones (get_page_content's matches the chat app's tool output field for field)."""

import base64
import json
import time
from typing import Any

from sqlalchemy import Engine, case, func, literal, or_, select

from .. import accounts, doctools, files, storage
from ..db import session_scope
from ..models import File, Folder, Page

WAIT_SECONDS = 180
WAIT_POLL = 2.0
STRUCTURE_PART_CHARS = 20_000
VIRTUAL_FOLDERS = ("shared-with-me", "following")  # sharing arrives in P7; empty until then


class ToolFailure(Exception):
    """Reported to the client as an error result (isError), not a protocol error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _folder_filter(folder_id: str | None):
    if folder_id in (None, ""):
        return None
    if folder_id == "root":
        return File.folder_id.is_(None)
    return File.folder_id == folder_id


def _find(s, owner: tuple[str, str], doc_name: str, folder_id: str | None) -> File:
    q = select(File).where(File.owner_kind == owner[0], File.owner_id == owner[1], File.file_name == doc_name)
    scope = _folder_filter(folder_id)
    if scope is not None:
        q = q.where(scope)
    matches = s.scalars(q.limit(2)).all()
    if not matches:
        where = f" in folder {folder_id!r}" if folder_id else ""
        raise ToolFailure(
            "document_not_found",
            f"No document named {doc_name!r}{where}. Call browse_documents() and copy the name verbatim.",
        )
    if len(matches) > 1:
        raise ToolFailure(
            "ambiguous_document",
            f"Several documents are named {doc_name!r}; pass folder_id from browse_documents() to pick one.",
        )
    return matches[0]


def _doc_out(f: File) -> dict:
    return {
        "name": f.file_name,
        "description": f.description,
        "status": f.status,
        "page_count": f.page_num,
        "folder_id": f.folder_id or "root",
        "created_at": f.created_at.isoformat(),
    }


def _next_steps(f: File) -> dict:
    """The hints the reference product attaches to a document card (recorded wording)."""
    name = f.file_name
    if f.status == "completed":
        size = (
            f"This is a large document with {f.page_num} pages."
            if (f.page_num or 0) > 20
            else (f"This document has {f.page_num} pages.")
        )
        return {
            "summary": "Document is ready for analysis and querying.",
            "options": [
                "Document is ready for analysis.",
                size,
                f'Start with first few pages: get_page_content(doc_name: "{name}", pages: "1-3")',
                f'Or view structure first: get_document_structure(doc_name: "{name}")',
                "To read a full rendered page (scanned or layout-heavy pages), pass an image_path from "
                "get_page_content()'s page_images array to get_document_image()",
            ],
        }
    if f.status == "failed":
        return {"summary": "Document processing failed.", "options": [f.error_message or "Upload it again."]}
    return {
        "summary": "Document is still being processed.",
        "options": [
            f'Call get_document(doc_name: "{name}", wait_for_completion: true) to wait until it is ready.'
        ],
    }


def _ready(owner, doc_name: str, folder_id: str | None, wait: bool, engine) -> File:
    """The document once processed; waits up to 3 minutes when asked to."""
    deadline = time.monotonic() + (WAIT_SECONDS if wait else 0)
    while True:
        with session_scope(engine) as s:
            f = _find(s, owner, doc_name, folder_id)
            s.expunge(f)
        if f.status in ("completed", "failed") or time.monotonic() >= deadline:
            return f
        time.sleep(WAIT_POLL)


def _require_completed(f: File) -> None:
    if f.status == "failed":
        raise ToolFailure("document_failed", f"{f.file_name!r} could not be processed: {f.error_message}")
    if f.status != "completed":
        raise ToolFailure(
            "document_processing",
            f"{f.file_name!r} is still being processed ({f.status}). Retry with wait_for_completion: true.",
        )


# ---------- tools ----------


def browse_documents(
    owner: tuple[str, str],
    folder_id: str = "root",
    recursive: bool = False,
    sort: str = "time",
    query: str | None = None,
    offset: int = 0,
    limit: int = 10,
    engine: Engine | None = None,
) -> dict:
    if sort == "relevance" and not (query or "").strip():
        raise ToolFailure("query_required", 'sort="relevance" needs a query.')
    folder_id = folder_id or "root"
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    with session_scope(engine) as s:
        if folder_id in VIRTUAL_FOLDERS:
            return {
                "folder_id": folder_id,
                "folders": [],
                "documents": [],
                "total": 0,
                "offset": offset,
                "next_offset": None,
            }
        all_folders = s.scalars(
            select(Folder).where(Folder.owner_kind == owner[0], Folder.owner_id == owner[1])
        ).all()
        if folder_id != "root" and folder_id not in {f.id for f in all_folders}:
            raise ToolFailure(
                "folder_not_found", f"No folder {folder_id!r}. Copy folder ids from browse_documents()."
            )
        parent = None if folder_id == "root" else folder_id

        q = select(File).where(File.owner_kind == owner[0], File.owner_id == owner[1])
        # a relevance search covers the folder's whole subtree: the agent is looking for a match,
        # not exploring the hierarchy level by level
        if recursive or sort == "relevance":
            if parent is not None:
                scope, frontier = {parent}, [parent]
                while frontier:
                    kids = [f.id for f in all_folders if f.parent_id in frontier]
                    scope.update(kids)
                    frontier = kids
                q = q.where(File.folder_id.in_(scope))
        else:
            q = q.where(File.folder_id.is_(None) if parent is None else File.folder_id == parent)

        if sort == "relevance":
            tsq = func.websearch_to_tsquery("english", query)
            text_rank = (
                select(func.coalesce(func.max(func.ts_rank(Page.text_tsv, tsq)), 0))
                .where(Page.file_id == File.id)
                .scalar_subquery()
            )
            # page text matches, plus a small boost per query word found in the name or description
            words = [w for w in query.lower().split() if len(w) > 2][:8]
            haystack = func.lower(File.file_name + " " + func.coalesce(File.description, ""))
            name_hits = sum(
                (case((func.strpos(haystack, w) > 0, 1), else_=0) for w in words), start=literal(0)
            )
            score = text_rank + 0.1 * name_hits
            q = q.add_columns(score.label("score")).where(or_(text_rank > 0, name_hits > 0))
            rows = s.execute(q.order_by(score.desc(), File.created_at.desc())).all()
            docs = [r[0] for r in rows]
        else:
            docs = s.scalars(q.order_by(File.created_at.desc())).all()

        total = len(docs)
        page = docs[offset : offset + limit]
        counts = dict(
            s.execute(
                select(File.folder_id, func.count())
                .where(File.owner_kind == owner[0], File.owner_id == owner[1], File.folder_id.is_not(None))
                .group_by(File.folder_id)
            ).all()
        )
        folders = (
            []
            if recursive or sort == "relevance"
            else [
                {
                    "id": f.id,
                    "name": f.name,
                    "description": f.description,
                    "document_count": counts.get(f.id, 0),
                }
                for f in sorted(all_folders, key=lambda f: f.name.lower())
                if f.parent_id == parent
            ]
        )
        return {
            "folder_id": folder_id,
            "folders": folders,
            "documents": [_doc_out(f) for f in page],
            "total": total,
            "offset": offset,
            "next_offset": offset + limit if offset + limit < total else None,
        }


def get_document(
    owner, doc_name: str, folder_id: str | None = None, wait_for_completion: bool = False, engine=None
) -> dict:
    f = _ready(owner, doc_name, folder_id, wait_for_completion, engine)
    return {"success": True, **_doc_out(f), "next_steps": _next_steps(f)}


def _parts(nodes: list[dict]) -> list[list[dict]]:
    """Top-level sections grouped into parts of about STRUCTURE_PART_CHARS of JSON each."""
    parts: list[list[dict]] = [[]]
    size = 0
    for n in nodes:
        n_size = len(json.dumps(n, ensure_ascii=False))
        if parts[-1] and size + n_size > STRUCTURE_PART_CHARS:
            parts.append([])
            size = 0
        parts[-1].append(n)
        size += n_size
    return parts


def get_document_structure(
    owner,
    doc_name: str,
    folder_id: str | None = None,
    part: int = 1,
    wait_for_completion: bool = False,
    engine=None,
) -> dict:
    f = _ready(owner, doc_name, folder_id, wait_for_completion, engine)
    _require_completed(f)
    outline = doctools.get_document_structure(owner, f.id, engine=engine)["structure"] or []
    # a single root (the document title) holds everything: page through its children instead
    root = outline[0] if len(outline) == 1 and outline[0].get("nodes") else None
    parts = _parts(root["nodes"] if root else outline)
    part = int(part)
    if not 1 <= part <= len(parts):
        raise ToolFailure("invalid_part", f"part must be between 1 and {len(parts)}.")
    chunk = parts[part - 1]
    structure = [{**{k: v for k, v in root.items() if k != "nodes"}, "nodes": chunk}] if root else chunk
    return {
        "success": True,
        "doc_name": f.file_name,
        "total_pages": f.page_num,
        "structure": structure,
        "pagination": {"part": part, "total_parts": len(parts), "has_more": part < len(parts)},
    }


def _image_path(file_id: str, page: int) -> str:
    return f"{file_id}/page/{page}"


def get_page_content(
    owner,
    doc_name: str,
    pages: str,
    folder_id: str | None = None,
    wait_for_completion: bool = False,
    engine=None,
) -> dict:
    f = _ready(owner, doc_name, folder_id, wait_for_completion, engine)
    _require_completed(f)
    try:
        wanted = doctools.parse_pages(pages, f.page_num or 0)
        out = doctools.get_page_content(owner, f.id, pages, engine=engine)
    except doctools.ToolError as e:
        raise ToolFailure("invalid_pages", str(e)) from e
    returned = sorted({b["page"] for b in out["content"]} | set(wanted[: doctools.MAX_PAGES_PER_CALL]))
    result = {
        "success": True,
        "doc_name": f.file_name,
        "total_pages": f.page_num,
        "requested_pages": pages,
        "returned_pages": out["returned_pages"],
        "content": out["content"],
        "page_images": [{"page": p, "image_path": _image_path(f.id, p)} for p in returned],
    }
    if len(wanted) > doctools.MAX_PAGES_PER_CALL:
        rest = wanted[doctools.MAX_PAGES_PER_CALL :]
        result["note"] = (
            f"At most {doctools.MAX_PAGES_PER_CALL} pages per call; pages {rest[0]}-{rest[-1]} were not returned."
        )
    return result


def get_document_image(owner, image_path: str, engine=None) -> tuple[str, str]:
    """(base64 PNG, caption) for an image_path from get_page_content()."""
    try:
        file_id, kind, page = image_path.strip().split("/")
        page_no = int(page)
    except ValueError as e:
        raise ToolFailure(
            "invalid_image_path", "Pass an image_path exactly as get_page_content() returned it."
        ) from e
    with session_scope(engine) as s:
        f = s.get(File, file_id)
        p = s.get(Page, (file_id, page_no)) if kind == "page" else None
        if f is None or (f.owner_kind, f.owner_id) != owner or p is None or not p.image_key:
            raise ToolFailure("image_not_found", f"No image at {image_path!r}.")
        name, key = f.file_name, p.image_key
    return base64.b64encode(storage.get_bytes(key)).decode(), f"Page {page_no} of {name}"


def remove_document(owner, doc_names: list[str], folder_id: str | None = None, engine=None) -> dict:
    results = []
    for name in list(dict.fromkeys(doc_names))[:10]:
        try:
            with session_scope(engine) as s:
                file_id = _find(s, owner, name, folder_id).id
        except ToolFailure as e:
            status = "not_found" if e.code == "document_not_found" else "failed"
            results.append(
                {"doc_name": name, "status": status, **({"error": str(e)} if status == "failed" else {})}
            )
            continue
        try:
            files.delete_file(file_id, engine=engine)
            results.append({"doc_name": name, "status": "deleted"})
        except Exception as e:  # reported per document, as the contract describes
            results.append({"doc_name": name, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    return {"success": all(r["status"] == "deleted" for r in results), "results": results}


# ---------- local upload (the @pageindex/mcp connector's process_document) ----------


def get_signed_upload_url(owner, file_name: str, file_type: str = "application/pdf") -> dict:
    """A presigned PUT; `file_name` comes back as the key to pass to submit_document()."""
    if file_type != "application/pdf":
        raise ToolFailure("unsupported_type", "Only PDF documents can be uploaded.")
    up = files.create_upload(owner[1], file_name, file_type)
    return {"upload_url": up["url"], "file_name": up["storage_key"], "expires_in": up["expires_in"]}


def submit_document(owner, file_name: str, folder_id: str | None = None, engine=None) -> dict:
    display = file_name.rsplit("/", 1)[-1]
    try:
        reg = files.register_upload(
            owner[0],
            owner[1],
            display,
            file_name,
            folder_id=None if folder_id in (None, "", "root") else folder_id,
            created_by_user_id=owner[1],
            page_limit=accounts.page_limit(),
            engine=engine,
        )
    except files.UploadError as e:
        raise ToolFailure(e.code, str(e)) from e
    return {
        "success": True,
        "doc_name": reg["name"],
        "status": "queued",
        "page_count": reg["page_count"],
        "estimated_time": reg["estimated_time"],
        "next_steps": {
            "summary": "Document submitted for processing.",
            "options": [
                f'Call get_document(doc_name: "{reg["name"]}", wait_for_completion: true) to wait until it is ready.'
            ],
        },
    }


def call(owner, name: str, args: dict[str, Any], engine: Engine | None = None) -> Any:
    """Dispatch one tool call; unknown arguments are ignored, as JSON-schema clients may add them."""
    a = dict(args or {})
    if name == "browse_documents":
        keys = ("folder_id", "recursive", "sort", "query", "offset", "limit")
        return browse_documents(
            owner, **{k: a[k] for k in keys if k in a and a[k] is not None}, engine=engine
        )
    if name == "get_document":
        return get_document(
            owner, a["doc_name"], a.get("folder_id"), bool(a.get("wait_for_completion")), engine
        )
    if name == "get_document_structure":
        return get_document_structure(
            owner,
            a["doc_name"],
            a.get("folder_id"),
            a.get("part") or 1,
            bool(a.get("wait_for_completion")),
            engine,
        )
    if name == "get_page_content":
        return get_page_content(
            owner,
            a["doc_name"],
            str(a["pages"]),
            a.get("folder_id"),
            bool(a.get("wait_for_completion")),
            engine,
        )
    if name == "get_document_image":
        return get_document_image(owner, a["image_path"], engine)
    if name == "remove_document":
        names = a.get("doc_names")
        if not isinstance(names, list) or not names:
            raise ToolFailure("invalid_arguments", "doc_names must be a non-empty list of document names.")
        return remove_document(owner, [str(n) for n in names], a.get("folder_id"), engine)
    if name == "get_signed_upload_url":
        return get_signed_upload_url(owner, a["fileName"], a.get("fileType") or "application/pdf")
    if name == "submit_document":
        return submit_document(owner, a["file_name"], a.get("folder_id"), engine)
    raise ToolFailure("unknown_tool", f"Unknown tool {name!r}.")

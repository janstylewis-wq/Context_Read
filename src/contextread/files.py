"""Document upload and lookup. Upload is two-step, as in the recorded flow: hand out a presigned
PUT URL, the client uploads straight to storage, then registers the object to start indexing."""

import re
from typing import Any

import pypdfium2 as pdfium
from botocore.exceptions import ClientError
from sqlalchemy import Engine, select

from . import jobs, storage
from .db import session_scope
from .ids import new_id
from .models import Block, File, Folder, Page

SECONDS_PER_PAGE = 2
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class UploadError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def safe_name(file_name: str) -> str:
    name = re.sub(r"[^\w.\- ()]+", "_", file_name.strip()).strip(". ") or "document.pdf"
    return name[:200]


def _upload_key(owner_id: str, file_name: str) -> str:
    return f"{owner_id}/uploads/{new_id()}/{safe_name(file_name)}"


def create_upload(owner_id: str, file_name: str, content_type: str = "application/pdf") -> dict:
    key = _upload_key(owner_id, file_name)
    return {"storage_key": key, "url": storage.presign_put(key, content_type), "expires_in": 600}


def store_upload(owner_id: str, file_name: str, data: bytes) -> str:
    """Server-side upload (REST multipart) into the same key layout as presigned uploads."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("file_too_large", "PDFs may be up to 100 MB")
    key = _upload_key(owner_id, file_name)
    storage.put_bytes(key, data, "application/pdf")
    return key


def unique_name(session, owner_kind: str, owner_id: str, name: str) -> str:
    """A taken name gets _1.._99 before its extension, as the reference API does."""
    taken = set(
        session.scalars(
            select(File.file_name).where(
                File.owner_kind == owner_kind,
                File.owner_id == owner_id,
                File.file_name.like(f"{name.rsplit('.', 1)[0]}%"),
            )
        )
    )
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    stem, ext = (stem, dot + ext) if dot else (name, "")
    for n in range(1, 100):
        candidate = f"{stem}_{n}{ext}"
        if candidate not in taken:
            return candidate
    raise UploadError("name_taken", "Too many files with similar names. Please use a different file name.")


def register_upload(
    owner_kind: str,
    owner_id: str,
    file_name: str,
    storage_key: str,
    folder_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    created_by_user_id: str | None = None,
    engine: Engine | None = None,
    page_limit: int | None = None,
) -> dict:
    if not storage_key.startswith(f"{owner_id}/"):
        raise UploadError("forbidden", "storage key does not belong to this owner")
    try:
        size = storage.get_client().head_object(Bucket=storage.bucket(), Key=storage_key)["ContentLength"]
    except ClientError as e:
        raise UploadError("upload_missing", "no uploaded object at that key") from e
    if size > MAX_UPLOAD_BYTES:
        raise UploadError("file_too_large", "PDFs may be up to 100 MB")
    data = storage.get_bytes(storage_key)
    if not data.startswith(b"%PDF-"):
        raise UploadError("not_a_pdf", "the uploaded file is not a PDF")
    try:
        doc = pdfium.PdfDocument(data)
        page_count = len(doc)
        doc.close()
    except pdfium.PdfiumError as e:
        code = "encrypted_pdf" if "password" in str(e).lower() else "unreadable_pdf"
        raise UploadError(code, str(e)) from e

    with session_scope(engine) as s:
        if page_limit is not None:
            from .accounts import pages_used

            if pages_used(s, (owner_kind, owner_id)) + page_count > page_limit:
                raise UploadError(
                    "page_limit", f"This upload would pass the plan's {page_limit:,}-page limit."
                )
        if folder_id is not None:
            folder = s.get(Folder, folder_id)
            if folder is None or (folder.owner_kind, folder.owner_id) != (owner_kind, owner_id):
                raise UploadError("folder_not_found", "no such folder")
        f = File(
            owner_kind=owner_kind,
            owner_id=owner_id,
            file_name=unique_name(s, owner_kind, owner_id, safe_name(file_name)),
            storage_key=storage_key,
            folder_id=folder_id,
            meta=metadata,
            created_by_user_id=created_by_user_id,
            page_num=page_count,
            status="queued",
            config={"if_ocr": True, "enable_block_ref": True},
        )
        s.add(f)
        s.flush()
        file_id, stored_name = f.id, f.file_name
    jobs.enqueue("index_document", {"file_id": file_id}, engine=engine)
    return {
        "doc_id": file_id,
        "name": stored_name,
        "page_count": page_count,
        "estimated_time": page_count * SECONDS_PER_PAGE,
    }


def delete_file(file_id: str, engine: Engine | None = None) -> bool:
    """Delete the record (pages, blocks and retrievals cascade) and its stored objects."""
    with session_scope(engine) as s:
        f = s.get(File, file_id)
        if f is None:
            return False
        upload_key, prefix = f.storage_key, f"{f.owner_id}/{f.id}/"
        shared_upload = s.scalar(
            select(File.id).where(File.storage_key == upload_key, File.id != file_id).limit(1)
        )
        # a duplicate upload reuses this file's page images and thumbnail; keep them while it exists
        shared_images = s.scalar(
            select(Page.file_id).where(Page.image_key.like(f"{prefix}%"), Page.file_id != file_id).limit(1)
        )
        s.delete(f)
    if shared_images is None:
        storage.delete_prefix(prefix)
    if shared_upload is None:
        storage.delete_key(upload_key)
    return True


def file_record(file_id: str, engine: Engine | None = None) -> dict | None:
    with session_scope(engine) as s:
        f = s.get(File, file_id)
        if f is None:
            return None
        return {
            "id": f.id,
            "owner_kind": f.owner_kind,
            "owner_id": f.owner_id,
            "file_name": f.file_name,
            "status": f.status,
            "page_num": f.page_num,
            "description": f.description,
            "structure": f.structure,
            "md_tree": f.md_tree,
            "metadata": f.meta,
            "folder_id": f.folder_id,
            "has_block_ref": f.has_block_ref,
            "thumbnail_key": f.thumbnail_key,
            "doc_hash": f.doc_hash,
            "error_message": f.error_message,
            "config": f.config,
            "index_version": f.index_version,
            "created_at": f.created_at.isoformat(),
            "updated_at": f.updated_at.isoformat(),
        }


def page_content(file_id: str, page: int, engine: Engine | None = None) -> dict | None:
    with session_scope(engine) as s:
        p = s.get(Page, (file_id, page))
        if p is None:
            return None
        blocks = s.scalars(
            select(Block).where(Block.file_id == file_id, Block.page == page).order_by(Block.ord)
        )
        return {
            "page": p.page,
            "width": p.width,
            "height": p.height,
            "read": p.read_mode,
            "image_url": storage.presign_get(p.image_key) if p.image_key else None,
            "content": [
                {"page": b.page, "text": b.text, "type": b.type, "block_id": b.block_id, "bbox": b.bbox}
                for b in blocks
            ],
        }


def list_files(
    owner_kind: str, owner_id: str, limit: int = 50, offset: int = 0, engine: Engine | None = None
) -> list[dict]:
    with session_scope(engine) as s:
        rows = s.scalars(
            select(File)
            .where(File.owner_kind == owner_kind, File.owner_id == owner_id)
            .order_by(File.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [
            {
                "id": f.id,
                "file_name": f.file_name,
                "status": f.status,
                "page_num": f.page_num,
                "description": f.description,
                "has_block_ref": f.has_block_ref,
                "created_at": f.created_at.isoformat(),
            }
            for f in rows
        ]


# ---------- web app document management (owner-checked) ----------

_UNSET = object()


def owned(s, owner: tuple[str, str], file_id: str) -> File:
    f = s.get(File, file_id)
    if f is None or (f.owner_kind, f.owner_id) != owner:
        raise UploadError("document_not_found", "Document not found.")
    return f


def web_out(f: File) -> dict:
    return {
        "id": f.id,
        "fileName": f.file_name,
        "status": f.status,
        "pageNum": f.page_num,
        "folderId": f.folder_id,
        "description": f.description,
        "hasBlockRef": f.has_block_ref,
        "metadata": f.meta,
        "errorMessage": f.error_message,
        "createdAt": f.created_at.isoformat(),
        "updatedAt": f.updated_at.isoformat(),
    }


def list_web(
    owner: tuple[str, str],
    folder_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    engine: Engine | None = None,
) -> dict:
    """folder_id None: every document; 'root': documents outside any folder; else that folder."""
    with session_scope(engine) as s:
        q = select(File).where(File.owner_kind == owner[0], File.owner_id == owner[1])
        if folder_id == "root":
            q = q.where(File.folder_id.is_(None))
        elif folder_id is not None:
            q = q.where(File.folder_id == folder_id)
        rows = list(s.scalars(q.order_by(File.created_at.desc(), File.id).limit(limit + 1).offset(offset)))
        return {
            "items": [web_out(f) for f in rows[:limit]],
            "nextOffset": offset + limit if len(rows) > limit else None,
        }


def update_file(
    owner: tuple[str, str],
    file_id: str,
    file_name: str | None = None,
    folder_id=_UNSET,
    engine: Engine | None = None,
) -> dict:
    with session_scope(engine) as s:
        f = owned(s, owner, file_id)
        if file_name is not None:
            name = safe_name(file_name)
            if name != f.file_name:
                f.file_name = unique_name(s, owner[0], owner[1], name)
        if folder_id is not _UNSET:
            if folder_id is not None:
                folder = s.get(Folder, folder_id)
                if folder is None or (folder.owner_kind, folder.owner_id) != owner:
                    raise UploadError("folder_not_found", "no such folder")
            f.folder_id = folder_id
        s.flush()
        s.refresh(f)
        return web_out(f)


def thumbnails(owner: tuple[str, str], file_ids: list[str], engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        rows = s.scalars(
            select(File).where(
                File.id.in_(file_ids[:100]), File.owner_kind == owner[0], File.owner_id == owner[1]
            )
        )
        return {f.id: storage.presign_get(f.thumbnail_key) if f.thumbnail_key else None for f in rows}


def download_url(owner: tuple[str, str], file_id: str, engine: Engine | None = None) -> str:
    with session_scope(engine) as s:
        return storage.presign_get(owned(s, owner, file_id).storage_key)


def delete_all(owner: tuple[str, str], engine: Engine | None = None) -> int:
    with session_scope(engine) as s:
        ids = list(s.scalars(select(File.id).where(File.owner_kind == owner[0], File.owner_id == owner[1])))
    for file_id in ids:
        delete_file(file_id, engine=engine)
    return len(ids)

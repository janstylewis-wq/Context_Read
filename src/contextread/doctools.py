"""Document tools shared by chat, retrieval and (P5) the MCP server. Every call is scoped to one
owner; documents are addressed by id or by name."""

import copy
import json
import re

from sqlalchemy import Engine, select

from .db import session_scope
from .models import Block, File

MAX_PAGES_PER_CALL = 10


class ToolError(Exception):
    pass


def resolve(s, owner: tuple[str, str], ref: str) -> File:
    f = s.get(File, ref)
    if f is None or (f.owner_kind, f.owner_id) != owner:
        f = s.scalars(
            select(File).where(File.owner_kind == owner[0], File.owner_id == owner[1], File.file_name == ref)
        ).first()
    if f is None:
        raise ToolError(f"No document {ref!r}. Call browse_documents to see the available documents.")
    return f


def parse_pages(spec: str | int, page_count: int) -> list[int]:
    """'5-7', '3,8', '12' -> sorted page numbers within 1..page_count."""
    pages: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ToolError(f"Invalid pages {spec!r}; use forms like '5', '3,8' or '5-7'.")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        pages.update(range(min(a, b), max(a, b) + 1))
    valid = sorted(p for p in pages if 1 <= p <= page_count)
    if not valid:
        raise ToolError(f"Pages {spec!r} are outside the document (1-{page_count}).")
    return valid


def card(f: File) -> dict:
    return {
        "id": f.id,
        "name": f.file_name,
        "description": f.description,
        "status": f.status,
        "page_count": f.page_num,
        "folder_id": f.folder_id,
        "metadata": f.meta,
    }


def browse_documents(
    owner: tuple[str, str], folder_id: str | None = None, limit: int = 50, engine: Engine | None = None
) -> dict:
    with session_scope(engine) as s:
        q = select(File).where(File.owner_kind == owner[0], File.owner_id == owner[1])
        if folder_id and folder_id != "root":
            q = q.where(File.folder_id == folder_id)
        docs = [card(f) for f in s.scalars(q.order_by(File.created_at.desc()).limit(limit))]
    return {"documents": docs, "count": len(docs)}


def outline(nodes: list[dict]) -> list[dict]:
    """The stored tree as tools show it: id, title, pages, summary and children (no node text)."""
    out = []
    for n in nodes:
        item = {
            "node_id": n.get("node_id"),
            "title": n.get("title"),
            "start_index": n.get("start_index"),
            "end_index": n.get("end_index"),
        }
        if n.get("summary"):
            item["summary"] = n["summary"]
        if n.get("nodes"):
            item["nodes"] = outline(n["nodes"])
        out.append(item)
    return out


SUMMARY_CAPS = (160, 60, 0)  # summary lengths tried, longest first, when a structure must shrink to fit


def _size(structure: list[dict]) -> int:
    return len(json.dumps(structure, ensure_ascii=False))


def fit_structure(structure: list[dict], limit: int) -> tuple[list[dict], str | None]:
    """The structure within `limit` characters of JSON, and a note when it had to shrink. Titles and page
    ranges are always kept in full; only summaries shorten, deepest sections first, since a section's
    title and its parent's summary still say what it holds. A structure that fits comes back unchanged."""
    if _size(structure) <= limit:
        return structure, None
    fitted = copy.deepcopy(structure)
    by_depth: dict[int, list[dict]] = {}

    def walk(nodes: list[dict], depth: int) -> None:
        for n in nodes:
            by_depth.setdefault(depth, []).append(n)
            walk(n.get("nodes") or [], depth + 1)

    walk(fitted, 0)
    for cap in SUMMARY_CAPS:
        for depth in sorted(by_depth, reverse=True):
            for n in by_depth[depth]:
                if "summary" not in n:
                    continue
                if cap == 0:
                    del n["summary"]
                elif len(n["summary"]) > cap:
                    n["summary"] = n["summary"][:cap].rsplit(" ", 1)[0] + "…"
            if _size(fitted) <= limit:
                return fitted, (
                    "This document's structure is large, so some section summaries are shortened or left out; "
                    "every section's title and pages are listed."
                )
    return (
        fitted,
        "This document's structure is very large: summaries are left out; titles and pages are listed.",
    )


def get_document_structure(owner: tuple[str, str], doc: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        f = resolve(s, owner, doc)
        if f.status != "completed":
            return {"doc_name": f.file_name, "status": f.status, "structure": None}
        return {
            "doc_name": f.file_name,
            "doc_id": f.id,
            "page_count": f.page_num,
            "structure": outline(f.structure or []),
        }


def get_page_content(
    owner: tuple[str, str], doc: str, pages: str | int, engine: Engine | None = None
) -> dict:
    with session_scope(engine) as s:
        f = resolve(s, owner, doc)
        if f.status != "completed":
            return {"doc_name": f.file_name, "status": f.status, "content": []}
        wanted = parse_pages(pages, f.page_num or 0)
        truncated = len(wanted) > MAX_PAGES_PER_CALL
        wanted = wanted[:MAX_PAGES_PER_CALL]
        blocks = s.scalars(
            select(Block).where(Block.file_id == f.id, Block.page.in_(wanted)).order_by(Block.page, Block.ord)
        )
        content = [
            {"page": b.page, "text": b.text, "type": b.type, "block_id": b.block_id} for b in blocks if b.text
        ]
    result = {
        "doc_name": f.file_name,
        "doc_id": f.id,
        "total_pages": f.page_num,
        "returned_pages": f"{wanted[0]}-{wanted[-1]}" if len(wanted) > 1 else str(wanted[0]),
        "content": content,
    }
    if truncated:
        result["note"] = (
            f"Only the first {MAX_PAGES_PER_CALL} requested pages were returned; ask for the rest separately."
        )
    return result

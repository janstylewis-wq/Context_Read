"""Citation lookups for the chat viewer: block ids from <cite/> tags -> page and box, and one
page's image and blocks. Boxes are in thousandths of the page (0-1000), as in the reference
product, so the client can draw them over the page image at any size."""

from sqlalchemy import Engine, select

from . import doctools, storage
from .db import session_scope
from .models import Block, Page


def bbox_thousandths(bbox: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        round(x0 * 1000 / width),
        round(y0 * 1000 / height),
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
    ]


def resolve_citations(owner: tuple[str, str], items: list[dict], engine: Engine | None = None) -> dict:
    """items: [{doc, blockIds}] -> {results: [{doc, fileId, blocks: [{blockId, page, bbox}], missing}]}"""
    results = []
    with session_scope(engine) as s:
        for item in items:
            wanted = list(dict.fromkeys(item.get("blockIds") or []))
            try:
                f = doctools.resolve(s, owner, item["doc"])
            except doctools.ToolError:
                results.append({"doc": item["doc"], "fileId": None, "blocks": [], "missing": wanted})
                continue
            found = {
                b.block_id: b
                for b in s.scalars(select(Block).where(Block.file_id == f.id, Block.block_id.in_(wanted)))
            }
            pages = {
                p.page: p
                for p in s.scalars(
                    select(Page).where(Page.file_id == f.id, Page.page.in_({b.page for b in found.values()}))
                )
            }
            blocks = [
                {
                    "blockId": bid,
                    "page": found[bid].page,
                    "bbox": bbox_thousandths(
                        found[bid].bbox, pages[found[bid].page].width, pages[found[bid].page].height
                    ),
                }
                for bid in wanted
                if bid in found
            ]
            results.append(
                {
                    "doc": f.file_name,
                    "fileId": f.id,
                    "blocks": blocks,
                    "missing": [bid for bid in wanted if bid not in found],
                }
            )
    return {"results": results}


def citation_page(owner: tuple[str, str], doc: str, page: int, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        f = doctools.resolve(s, owner, doc)
        p = s.get(Page, (f.id, page))
        if p is None:
            raise doctools.ToolError(f"Page {page} is not in {f.file_name!r} (1-{f.page_num}).")
        blocks = s.scalars(select(Block).where(Block.file_id == f.id, Block.page == page).order_by(Block.ord))
        return {
            "docName": f.file_name,
            "fileId": f.id,
            "page": page,
            "pageCount": f.page_num,
            "imageUrl": storage.presign_get(p.image_key) if p.image_key else None,
            "aspect": p.height / p.width,
            "blocks": [
                {
                    "blockId": b.block_id,
                    "type": b.type,
                    "text": b.text,
                    "bbox": bbox_thousandths(b.bbox, p.width, p.height),
                }
                for b in blocks
            ],
        }

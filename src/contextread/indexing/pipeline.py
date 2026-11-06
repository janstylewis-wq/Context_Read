"""index_document job: PDF in object storage -> pages, blocks (with bboxes), tree, summaries.

Status moves queued -> processing -> completed | failed; the thumbnail is committed first so
the UI can show it while indexing runs. Re-running a job replaces the file's pages and blocks.
"""

import hashlib
import io
import logging
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

import pdfplumber
import pypdfium2 as pdfium
from sqlalchemy import Engine, delete, select

from .. import storage
from ..config import get_settings
from ..db import session_scope
from ..jobs import handler
from ..llm import BudgetedLLM
from ..models import Block as BlockRow
from ..models import File, Page, UsageEvent
from . import tree
from .ocr import OCR_SCALE, OcrUnavailable, ocr_blocks
from .pdf_blocks import PAGE_SCALE, Block, body_font_size, page_blocks
from .vlm import VLM_SCALE, VisionReader, assess_page, read_page

log = logging.getLogger(__name__)

# 4: flash expand splits sections the layout leaves whole; 5: long leaves split into windows;
# 6: a section is summarised from its whole text (subsections included), and only from that text;
# 7: a document with a table of contents (page numbers given) takes its outline from it
PIPELINE_VERSION = 7
THUMBNAIL_SCALE = 0.5
HEADINGS_TREE_SHARE = (
    0.5  # the title-block tree replaces flash's only when this share of pages has no text layer
)


class IndexingError(Exception):
    """A permanent failure (bad input); the job is not retried."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _encode(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _png(page: pdfium.PdfPage, scale: float) -> tuple[bytes, int, int]:
    image = page.render(scale=scale).to_pil()
    return _encode(image), image.width, image.height


def default_reader() -> VisionReader | None:
    settings = get_settings()
    if settings.ocr_vlm_model == "off":
        return None
    llm = BudgetedLLM(provider=settings.ocr_vlm_provider)
    return VisionReader(llm, settings.ocr_vlm_model, settings.ocr_vlm_concurrency)


def _read_unusable_page(reader: VisionReader | None, page_no: int, vlm_png: bytes | None, ocr_png: bytes):
    """(blocks, read_mode) for a page whose text layer cannot be used. The vision model reads it and places its
    blocks in one call when available; Tesseract is the fallback reader (no reader, budget cap reached, API
    errors). When both fail the page is left without text rather than failing the whole document."""
    if reader is not None and vlm_png is not None:
        try:
            return read_page(reader, page_no, vlm_png), "vlm"
        except Exception as e:
            log.warning("vision read failed for page %s, using Tesseract: %s", page_no, e)
    try:
        return ocr_blocks(ocr_png, page_no), "image"
    except (OcrUnavailable, subprocess.TimeoutExpired, OSError) as e:
        log.warning("page %s could not be read (%s); left without text", page_no, e)
        return [], "image"


def _thumbnail(page: pdfium.PdfPage) -> bytes:
    image = page.render(scale=THUMBNAIL_SCALE).to_pil().convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _open_pdf(data: bytes) -> pdfium.PdfDocument:
    if not data.startswith(b"%PDF-"):
        raise IndexingError("not_a_pdf", "file does not start with a PDF header")
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        msg = str(e).lower()
        if "password" in msg or "security" in msg:
            raise IndexingError("encrypted_pdf", "PDF is password-protected") from e
        raise IndexingError("unreadable_pdf", str(e)) from e
    if len(doc) == 0:
        raise IndexingError("unreadable_pdf", "PDF has no pages")
    return doc


def _update(engine: Engine | None, file_id: str, **values: Any) -> None:
    with session_scope(engine) as s:
        f = s.get(File, file_id)
        for k, v in values.items():
            setattr(f, k, v)


def _copy_from_duplicate(engine: Engine | None, file: File, source: File) -> None:
    with session_scope(engine) as s:
        s.execute(delete(Page).where(Page.file_id == file.id))
        s.execute(delete(BlockRow).where(BlockRow.file_id == file.id))
        for p in s.scalars(select(Page).where(Page.file_id == source.id)):
            s.add(
                Page(
                    file_id=file.id,
                    page=p.page,
                    width=p.width,
                    height=p.height,
                    image_key=p.image_key,
                    read_mode=p.read_mode,
                    text=p.text,
                )
            )
        for b in s.scalars(select(BlockRow).where(BlockRow.file_id == source.id)):
            s.add(
                BlockRow(
                    file_id=file.id,
                    block_id=b.block_id,
                    page=b.page,
                    ord=b.ord,
                    type=b.type,
                    text=b.text,
                    bbox=b.bbox,
                )
            )
        f = s.get(File, file.id)
        f.status, f.page_num, f.doc_hash = "completed", source.page_num, source.doc_hash
        f.structure, f.md_tree, f.description = source.structure, source.md_tree, source.description
        f.thumbnail_key, f.has_block_ref = source.thumbnail_key, source.has_block_ref
        f.config = {**(file.config or {}), "duplicate_of": source.id}
        f.index_version, f.error_message = source.index_version, None


def extract_pages(
    data: bytes, file: File, doc: pdfium.PdfDocument, reader: VisionReader | None = None
) -> tuple[list[dict], list[Block]]:
    """Render, store and block every page. Pages with a usable text layer use it; the others are
    read by the vision model (several at once) or by Tesseract. Without a `reader`, the default
    one is made when the first such page turns up."""
    prefix = f"{file.owner_id}/{file.id}"
    pages: list[dict] = []
    page_blocks_by_no: dict[int, list[Block]] = {}
    pending: dict[Future, dict] = {}
    workers = reader.concurrency if reader else get_settings().ocr_vlm_concurrency
    with pdfplumber.open(io.BytesIO(data)) as plumber, ThreadPoolExecutor(workers) as pool:
        body = body_font_size(plumber.pages)
        for i in range(len(doc)):
            page_no = i + 1
            pdf_page = doc[i]
            image = pdf_page.render(scale=PAGE_SCALE).to_pil()
            image_key = f"{prefix}/pages/{page_no}.png"
            storage.put_bytes(image_key, _encode(image), "image/png")
            plumber_page = plumber.pages[i]
            page = {"page": page_no, "width": image.width, "height": image.height, "image_key": image_key}
            pages.append(page)
            reason = assess_page(plumber_page, image)
            if reason is None:
                page["read_mode"] = "text"
                page_blocks_by_no[page_no] = page_blocks(plumber_page, page_no, body)
            else:
                if not pending and reader is None:
                    reader = default_reader()
                # pdfium is not thread-safe: render here, read in the pool
                ocr_png, _, _ = _png(pdf_page, OCR_SCALE)
                vlm_png = _png(pdf_page, VLM_SCALE)[0] if reader else None
                pending[pool.submit(_read_unusable_page, reader, page_no, vlm_png, ocr_png)] = page
                page["unusable_text_layer"] = reason
                while sum(not f.done() for f in pending) >= 2 * workers:
                    wait(pending, return_when=FIRST_COMPLETED)
            pdf_page.close()
            plumber_page.flush_cache()
        for future, page in pending.items():
            page_blocks_by_no[page["page"]], page["read_mode"] = future.result()

    blocks: list[Block] = []
    for page in pages:
        own = page_blocks_by_no[page["page"]]
        page["text"] = "\n\n".join(b.text for b in own if b.text)
        blocks.extend(own)
    return pages, blocks


def index_document(
    file_id: str,
    llm: BudgetedLLM | None = None,
    engine: Engine | None = None,
    reader: VisionReader | None = None,
    reuse_duplicate: bool = True,
) -> dict:
    """Index one uploaded document. A file this owner already has indexed is copied from that copy unless
    reuse_duplicate is False (re-indexing under changed settings)."""
    started = time.monotonic()
    with session_scope(engine) as s:
        file = s.get(File, file_id)
        if file is None:
            raise IndexingError("document_not_found", file_id)
    _update(engine, file_id, status="processing", error_message=None)
    try:
        data = storage.get_bytes(file.storage_key)
        doc_hash = hashlib.sha256(data).hexdigest()
        with session_scope(engine) as s:
            source = s.scalars(
                select(File)
                .where(
                    File.owner_kind == file.owner_kind,
                    File.owner_id == file.owner_id,
                    File.doc_hash == doc_hash,
                    File.status == "completed",
                    File.id != file.id,
                )
                .limit(1)
            ).first()
        if source is not None and reuse_duplicate:
            _copy_from_duplicate(engine, file, source)
            return {"file_id": file_id, "duplicate_of": source.id}

        doc = _open_pdf(data)
        page_count = len(doc)
        thumb_key = f"{file.owner_id}/{file.id}/thumbnail.jpg"
        storage.put_bytes(thumb_key, _thumbnail(doc[0]), "image/jpeg")
        _update(engine, file_id, thumbnail_key=thumb_key, page_num=page_count, doc_hash=doc_hash)

        pages, blocks = extract_pages(data, file, doc, reader)
        doc.close()
        unusable = {p["page"]: p.pop("unusable_text_layer") for p in pages if "unusable_text_layer" in p}
        with session_scope(engine) as s:
            s.execute(delete(BlockRow).where(BlockRow.file_id == file_id))
            s.execute(delete(Page).where(Page.file_id == file_id))
            s.flush()
            s.add_all(Page(file_id=file_id, **p) for p in pages)
            s.flush()
            s.add_all(
                BlockRow(
                    file_id=file_id,
                    block_id=b.block_id,
                    page=b.page,
                    ord=n,
                    type=b.type,
                    text=b.text,
                    bbox=b.bbox,
                )
                for n, b in enumerate(blocks)
            )

        llm = llm or BudgetedLLM()
        expand_model = get_settings().tree_expand_model
        expanding = bool(expand_model) and expand_model != "off"
        structure, doc_title, toc_source = tree.flash_structure(data)
        flash_found_none = toc_source in ("pages", "unreadable") or not structure
        if unusable and (flash_found_none or len(unusable) >= HEADINGS_TREE_SHARE * page_count):
            # flash reads the PDF's own text layer, which is missing on these pages; a few such pages (signature
            # pages, a scanned exhibit) leave flash's tree of the rest better than one built from title blocks
            from_headings = tree.heading_structure(blocks, page_count, llm)
            if from_headings:
                structure, toc_source = from_headings, "headings"
        texts = {p["page"]: p.get("text") or "" for p in pages}
        page_texts = [texts.get(n, "") for n in range(1, page_count + 1)]
        from_contents = tree.contents_structure(page_texts, llm, expand_model) if expanding else None
        if from_contents:
            structure, toc_source = from_contents, "contents"
        if toc_source in ("pages", "unreadable") or not structure:
            structure = tree.page_fallback_structure(page_count, page_leaves=not expanding)
            toc_source = "pages"
        if expanding:
            if toc_source != "contents":  # the contents already hold the headings expand would look for
                expanded = tree.expand_structure(structure, page_texts, llm, expand_model)
                if expanded and toc_source == "pages":
                    toc_source = "pages+expand"
            tree.split_long_leaves(
                structure, page_texts, llm, expand_model, headings=toc_source != "contents"
            )
        structure = tree.wrap_in_root(structure, doc_title, page_count)
        tree.relabel(structure)
        tree.attach_text(structure, blocks)

        summarized = tree.summarize_nodes(structure, llm)
        description = tree.describe(structure, doc_title, llm)
        node_count = len(tree.preorder(structure))
        incomplete = summarized < node_count or description is None

        _update(
            engine,
            file_id,
            status="completed",
            structure=structure,
            md_tree=tree.md_tree(structure),
            description=description,
            has_block_ref=True,
            index_version=PIPELINE_VERSION,
            error_message=("summaries incomplete: LLM budget cap reached" if incomplete else None),
            config={
                **(file.config or {}),
                "pipeline_version": PIPELINE_VERSION,
                "toc_source": toc_source,
                "doc_title": doc_title,
                "ocr_pages": sum(p["read_mode"] == "image" for p in pages),
                "vlm_pages": sum(p["read_mode"] == "vlm" for p in pages),
                "unusable_text_layer": {str(k): v for k, v in unusable.items()},
                "page_scale": PAGE_SCALE,
            },
        )
        with session_scope(engine) as s:
            s.add(
                UsageEvent(
                    owner_kind=file.owner_kind,
                    owner_id=file.owner_id,
                    kind="page",
                    qty=page_count,
                    file_id=file_id,
                )
            )
        stats = {
            "file_id": file_id,
            "pages": page_count,
            "blocks": len(blocks),
            "nodes": node_count,
            "summarized": summarized,
            "toc_source": toc_source,
            "seconds": round(time.monotonic() - started, 1),
        }
        log.info("indexed %s", stats)
        return stats
    except IndexingError as e:
        _update(engine, file_id, status="failed", error_message=str(e))
        raise
    except Exception as e:
        _update(
            engine, file_id, status="failed", error_message=f"internal_error: {type(e).__name__}: {e}"[:2000]
        )
        raise


@handler("index_document")
def _index_document_job(payload: dict[str, Any]) -> None:
    try:
        index_document(payload["file_id"])
    except IndexingError:
        log.warning("indexing failed permanently for %s", payload.get("file_id"))
        # recorded on the file; a bad input does not get better on retry

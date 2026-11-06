import io
import json
from pathlib import Path

import httpx
import pdfplumber
import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from contextread import config, files, jobs, storage
from contextread.api.main import app
from contextread.db import session_scope
from contextread.indexing import pipeline, tree
from contextread.indexing.pdf_blocks import PAGE_SCALE, body_font_size, item_text, page_blocks
from contextread.llm import budget
from contextread.models import Block, Page, UsageEvent

FIXTURES = Path(__file__).parent / "fixtures"
DISNEY = FIXTURES / "q1-fy25-earnings.pdf"


def upload(path_or_bytes, name, owner="user-test", engine=None) -> dict:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    up = files.create_upload(owner, name)
    assert httpx.put(up["url"], content=data, headers={"content-type": "application/pdf"}).status_code == 200
    return files.register_upload("user", owner, name, up["storage_key"], engine=engine)


def scanned_pdf(pages=(1, 3)) -> bytes:
    """An image-only PDF (no text layer) made by rasterizing pages of the Disney release."""
    doc = pdfium.PdfDocument(str(DISNEY))
    images = [doc[p - 1].render(scale=200 / 72).to_pil().convert("RGB") for p in pages]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:], resolution=200)
    return buf.getvalue()


# ---------- blocks ----------


def test_text_layer_blocks_match_reference_content():
    reference = json.loads((FIXTURES / "pageindex_reference_blocks.json").read_text())
    with pdfplumber.open(DISNEY) as pdf:
        body = body_font_size(pdf.pages)
        p1 = page_blocks(pdf.pages[0], 1, body)
        p3 = page_blocks(pdf.pages[2], 3, body)

    revenue_list = next(b for b in p1 if b.type == "list" and "Revenues increased 5%" in b.text)
    assert revenue_list.text.startswith("- Revenues increased 5% for Q1 to $24.7 billion")
    assert "- Income before income taxes increased 27%" in revenue_list.text

    segment_table = next(b for b in p3 if b.type == "table" and "Experiences" in b.text)
    assert "| Total revenues | $ 24,690 | $ 23,549 | 5 % |" in segment_table.text
    assert "| Experiences | 3,110 | 3,105 | — % |" in segment_table.text

    for page, ours in ((1, p1), (3, p3)):
        assert [b.block_id for b in ours] == [f"p{page}_{b.type}_{n}" for n, b in enumerate(ours, 1)]
        types = {b.type for b in ours}
        assert {"title", "text"} <= types and ({"list"} <= types or {"table"} <= types)
        assert {"references"} <= types  # footnotes are typed as the reference product types them
        # comparable granularity with the reference product (10 and 9 blocks)
        assert abs(len(ours) - len(reference[str(page)])) <= 3
        assert not any(b.text.strip().isdigit() for b in ours), "page numbers must be dropped"
        for b in ours:
            x0, y0, x1, y1 = b.bbox
            assert 0 <= x0 < x1 <= 612 * PAGE_SCALE + 1 and 0 <= y0 < y1 <= 792 * PAGE_SCALE + 1


# ---------- full pipeline ----------


def test_pipeline_indexes_disney_release(engine, clean_files, fake_llm, bucket):
    llm, client = fake_llm
    reg = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    assert reg["page_count"] == 22 and reg["estimated_time"] > 0
    stats = pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)

    record = files.file_record(reg["doc_id"], engine=engine)
    assert record["status"] == "completed" and record["error_message"] is None
    assert record["has_block_ref"] and record["page_num"] == 22 and record["description"]

    structure = record["structure"]
    nodes = tree.preorder(structure)
    titles = [n["title"] for n, _ in nodes]
    assert len(structure) == 1 and "WALT DISNEY" in structure[0]["title"].upper()  # document root
    assert 25 <= len(nodes) <= 40 and max(d for _, d in nodes) == 3
    assert [n["node_id"] for n, _ in nodes] == [f"{i:04d}" for i in range(len(nodes))]
    for want in (
        "Entertainment",
        "Sports",
        "Experiences",
        "Capital Expenditures",
        "NON-GAAP FINANCIAL MEASURES",
    ):
        assert any(want.lower() in t.lower() for t in titles), want
    experiences = next(n for n, _ in nodes if n["title"] == "Experiences")
    assert experiences["start_index"] == 9 and "Experiences" in experiences["text"]
    assert all(n.get("summary") for n, _ in nodes)
    assert sum(bool(n["text"]) for n, _ in nodes) >= len(nodes) - 3
    assert [m["start_index"] for m in record["md_tree"]] == [n["start_index"] for n, _ in nodes]

    # the first 20 pages are checked for a table of contents (this release has none, so the layout tree stays)
    contents = [c for c in client.calls if c["max_completion_tokens"] == tree.CONTENTS_MAX_TOKENS]
    assert len(contents) == tree.CONTENTS_CHECK_PAGES and stats["toc_source"] != "contents"
    # then one LLM call per node summary plus the description, all through the guard
    calls = [c for c in client.calls if c not in contents]
    assert len(calls) == len(nodes) + 1 == stats["summarized"] + 1
    # output stays bounded: the answer's own limit, or a reasoning model's floor (reasoning counts against it)
    reasoning = config.get_settings().is_reasoning(config.get_settings().llm_default_model)
    limit = budget.REASONING_MIN_TOKENS if reasoning else 150
    assert all(c["max_completion_tokens"] <= limit for c in calls)
    assert not reasoning or all("temperature" not in c for c in calls)

    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Page).where(Page.file_id == reg["doc_id"])) == 22
        n_blocks = s.scalar(select(func.count()).select_from(Block).where(Block.file_id == reg["doc_id"]))
        assert n_blocks >= 100
        usage = s.scalars(select(UsageEvent).where(UsageEvent.file_id == reg["doc_id"])).one()
        assert (usage.kind, usage.qty) == ("page", 22)

    page3 = files.page_content(reg["doc_id"], 3, engine=engine)
    assert page3["read"] == "text" and page3["width"] == 918 and page3["height"] == 1188
    assert any(b["block_id"].startswith("p3_table_") for b in page3["content"])
    assert httpx.get(page3["image_url"]).content[:8] == b"\x89PNG\r\n\x1a\n"
    assert storage.get_bytes(record["thumbnail_key"])[:3] == b"\xff\xd8\xff"  # JPEG

    # the same bytes uploaded again reuse the index without any LLM calls
    again = upload(DISNEY, "copy.pdf", engine=engine)
    calls_before = len(client.calls)
    pipeline.index_document(again["doc_id"], llm=llm, engine=engine)
    copy = files.file_record(again["doc_id"], engine=engine)
    assert copy["status"] == "completed" and copy["config"]["duplicate_of"] == reg["doc_id"]
    assert copy["structure"] == structure and len(client.calls) == calls_before
    assert files.page_content(again["doc_id"], 1, engine=engine)["content"]


def test_scanned_pdf_is_ocrd(engine, clean_files, fake_llm, bucket):
    llm, _ = fake_llm
    reg = upload(scanned_pdf(), "scanned.pdf", engine=engine)
    pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)
    record = files.file_record(reg["doc_id"], engine=engine)
    assert record["status"] == "completed" and record["config"]["ocr_pages"] == 2
    assert record["config"]["toc_source"] == "pages"
    assert [n["title"] for n, _ in tree.preorder(record["structure"])] == ["Page 1", "Page 2"]
    page1 = files.page_content(reg["doc_id"], 1, engine=engine)
    assert page1["read"] == "image"
    text = " ".join(b["text"] for b in page1["content"])
    assert "Revenues increased 5%" in text and "$24.7 billion" in text
    assert all(b["block_id"].startswith("p1_") for b in page1["content"])


def test_job_queue_runs_indexing(engine, clean_files, fake_llm, bucket, monkeypatch):
    llm, _ = fake_llm
    monkeypatch.setattr(pipeline, "BudgetedLLM", lambda: llm)
    monkeypatch.setattr(pipeline, "session_scope", lambda e=None: session_scope(engine))
    reg = upload(scanned_pdf(pages=(2,)), "one-page.pdf", engine=engine)
    assert jobs.run_one("w-test", engine) is True
    assert files.file_record(reg["doc_id"], engine=engine)["status"] == "completed"


def test_register_rejects_bad_uploads(engine, clean_files, bucket):
    up = files.create_upload("user-test", "notes.pdf")
    httpx.put(up["url"], content=b"just text", headers={"content-type": "application/pdf"})
    with pytest.raises(files.UploadError) as e:
        files.register_upload("user", "user-test", "notes.pdf", up["storage_key"], engine=engine)
    assert e.value.code == "not_a_pdf"
    with pytest.raises(files.UploadError) as e:
        files.register_upload("user", "someone-else", "notes.pdf", up["storage_key"], engine=engine)
    assert e.value.code == "forbidden"
    with pytest.raises(files.UploadError) as e:
        files.register_upload("user", "user-test", "x.pdf", "user-test/uploads/missing.pdf", engine=engine)
    assert e.value.code == "upload_missing"


def test_internal_api_requires_token(bucket):
    with TestClient(app) as client:
        assert (
            client.post("/internal/uploads", json={"owner_id": "u", "file_name": "a.pdf"}).status_code == 401
        )
        token = config.get_settings().internal_api_token
        ok = client.post(
            "/internal/uploads",
            json={"owner_id": "u", "file_name": "a b?.pdf"},
            headers={"x-internal-token": token},
        )
        assert ok.status_code == 200 and ok.json()["storage_key"].endswith("/a b_.pdf")
        missing = client.get("/internal/files/nope", headers={"x-internal-token": token})
        assert missing.status_code == 404


def test_list_items_keep_clause_labels():
    # agreements cite clauses by label ("Section 6.01(d)"), so only bullet glyphs are dropped
    assert item_text("(d) Capital Lease Obligations") == "(d) Capital Lease Obligations"
    assert item_text("2. Definitions") == "2. Definitions"
    assert item_text("• Indebtedness") == "Indebtedness"

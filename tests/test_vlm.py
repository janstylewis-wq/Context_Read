import json
import re

import pdfplumber
import pypdfium2 as pdfium
import pytest
from fakes import FakeClient, _Raw, completion
from test_indexing import DISNEY, FIXTURES, scanned_pdf, upload

from contextread import files
from contextread.indexing import pipeline, tree
from contextread.indexing.pdf_blocks import PAGE_SCALE, Block
from contextread.indexing.vlm import VLM_SCALE, assess_page, markdown_units, read_page
from contextread.llm import BudgetedLLM, BudgetExceeded

COURT = FIXTURES / "court-order-excerpt.pdf"
RECORDED = {int(k): v for k, v in json.loads((FIXTURES / "court_order_excerpt_vlm.json").read_text()).items()}


def assessments(pdf) -> list[str | None]:
    doc, pl = pdfium.PdfDocument(pdf), pdfplumber.open(pdf)
    return [assess_page(pl.pages[i], doc[i].render(scale=PAGE_SCALE).to_pil()) for i in range(len(doc))]


def test_assess_page_separates_clean_outlined_and_image_pages(tmp_path):
    assert set(assessments(str(DISNEY))) == {None}  # tables, charts and logos are not missing text
    assert assessments(str(COURT)) == ["outlined_text"] * 3
    scanned = tmp_path / "scanned.pdf"
    scanned.write_bytes(scanned_pdf(pages=(1,)))
    assert assessments(str(scanned)) == ["no_text"]


def test_markdown_units():
    md = (
        "```markdown\n# ANALYSIS\n\nAs noted above, there are\ntwo ripe motions.\n\n"
        "**I. Defendants’ Motion for Summary Judgment.**\n\n---\n\n"
        "| Item | 2025 |\n|---|---:|\n| Revenues | $ 24,690 |\n\n- first\n- second\n```"
    )
    assert markdown_units(md) == [
        ("title", "# ANALYSIS"),
        ("text", "As noted above, there are two ripe motions."),
        ("title", "### I. Defendants’ Motion for Summary Judgment."),
        ("table", "| Item | 2025 |\n|---|---:|\n| Revenues | $ 24,690 |"),
        ("list", "- first\n- second"),
    ]


def _png(width: int, height: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_read_page_places_blocks_from_the_models_boxes():
    class Reader:
        def read_layout(self, page_no, png):
            return [
                {"text": "# ANALYSIS", "bbox": [100, 50, 500, 80]},
                {"text": "## II. Motions\n\nThe Court GRANTS the motion.", "bbox": [100, 100, 900, 300]},
                {"text": "A footnote.", "bbox": [900, 10, 100, 20]},  # malformed: x1 < x0
                {"text": "No box at all."},
            ]

    blocks = read_page(Reader(), 5, _png(1224, 1584))  # a letter page rendered at VLM_SCALE
    w, h = 1224 * PAGE_SCALE / VLM_SCALE, 1584 * PAGE_SCALE / VLM_SCALE
    assert [b.block_id for b in blocks] == ["p5_title_1", "p5_title_2", "p5_text_3", "p5_text_4", "p5_text_5"]
    assert blocks[0].bbox == [
        round(100 * w / 1000),
        round(50 * h / 1000),
        round(500 * w / 1000),
        round(80 * h / 1000),
    ]
    assert blocks[1].bbox == blocks[2].bbox  # a heading and its paragraph read as one block share its box
    assert blocks[3].bbox == blocks[4].bbox == [0, 0, round(w), round(h)]  # no usable box: the whole page


class RecordedReader:
    concurrency = 3

    def __init__(self, fail_pages=()):
        self.fail_pages, self.calls = set(fail_pages), []

    def read_layout(self, page_no, png):
        self.calls.append(page_no)
        if page_no in self.fail_pages:
            raise BudgetExceeded("openai LLM budget $10.00 would be exceeded")
        # the recorded transcription, one block per paragraph, stacked down the page
        chunks = [c for c in RECORDED[page_no].split("\n\n") if c.strip()]
        return [{"text": c, "bbox": [100, 20 + 20 * i, 900, 35 + 20 * i]} for i, c in enumerate(chunks)]


class LevelsClient(FakeClient):
    """Answers the document-wide heading pass the way a model reading the numbering would."""

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format", {}).get("type") == "json_object":
            titles = re.findall(r"^\d+\. \[p\d+\] #+ (.*)$", kwargs["messages"][0]["content"], re.M)
            levels = [1 if t.isupper() else 2 if re.match(r"[IVX]+\. ", t) else 3 for t in titles]
            return _Raw(
                completion(json.dumps({"levels": {str(i): v for i, v in enumerate(levels, 1)}})),
                self._headers,
            )
        return _Raw(completion("A short factual summary."), self._headers)


@pytest.fixture
def levels_llm(clean_db):
    client = LevelsClient(headers={"x-litellm-response-cost": "0.00001"})
    return BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=False)


def test_outlined_text_pages_are_read_by_the_vision_model(engine, clean_files, levels_llm, bucket):
    reg = upload(COURT, "court-order.pdf", engine=engine)
    reader = RecordedReader()
    pipeline.index_document(reg["doc_id"], llm=levels_llm, engine=engine, reader=reader)
    record = files.file_record(reg["doc_id"], engine=engine)
    assert sorted(reader.calls) == [1, 2, 3]
    assert record["config"]["vlm_pages"] == 3 and record["config"]["toc_source"] == "headings"
    # headings that exist only as vector outlines in the PDF, nested across pages (depths relative
    # to ANALYSIS: flash may wrap them all under a document title)
    outline = [(depth, n["title"], n["start_index"]) for n, depth in tree.preorder(record["structure"])]
    analysis = next(i for i, (_, t, _) in enumerate(outline) if t == "ANALYSIS")
    top = outline[analysis][0]
    assert (top, "II. Rule 37(c), FRCP: Substantially Justified or Harmless.", 1) in outline
    assert (top, "III. Rule 56, FRCP: Summary Judgment.", 1) in outline
    assert outline[analysis + 1 :] == [
        (top + 1, "I. Defendants’ Motion for Summary Judgment.", 2),
        (top + 1, "II. Plaintiff’s Motion to Substitute Expert Witness.", 3),
    ]

    page3 = files.page_content(reg["doc_id"], 3, engine=engine)
    assert page3["read"] == "vlm"
    heading = next(b for b in page3["content"] if "Substitute Expert Witness" in b["text"])
    assert heading["type"] == "title"
    # placed from the model's box, not stretched over the whole page
    assert 0 < heading["bbox"][1] < heading["bbox"][3] and heading["bbox"][2] - heading["bbox"][0] < 0.9 * 918
    text = " ".join(b["text"] for b in page3["content"])
    assert "if the court concludes that she fails to meet Rule 16(b)(4)" in text  # outline-only words


def test_vision_failure_falls_back_to_tesseract(engine, clean_files, levels_llm, bucket):
    reg = upload(COURT, "court-order.pdf", engine=engine)
    pipeline.index_document(
        reg["doc_id"], llm=levels_llm, engine=engine, reader=RecordedReader(fail_pages={3})
    )
    modes = [files.page_content(reg["doc_id"], p, engine=engine)["read"] for p in (1, 2, 3)]
    assert modes == ["vlm", "vlm", "image"]
    page3 = " ".join(b["text"] for b in files.page_content(reg["doc_id"], 3, engine=engine)["content"])
    assert "Substitute Expert Witness" in page3


def test_heading_levels_fall_back_to_reader_levels(fake_llm):
    llm, _ = fake_llm  # replies with prose, not the JSON levels
    blocks = [
        Block(1, "title", "# PART ONE", [0, 0, 1, 1]),
        Block(2, "title", "## Background", [0, 0, 1, 1]),
        Block(4, "title", "# PART TWO", [0, 0, 1, 1]),
    ]
    structure = tree.heading_structure(blocks, 6, llm)
    assert [(n["title"], n["start_index"], n["end_index"]) for n, _ in tree.preorder(structure)] == [
        ("PART ONE", 1, 4),
        ("Background", 2, 4),
        ("PART TWO", 4, 6),
    ]

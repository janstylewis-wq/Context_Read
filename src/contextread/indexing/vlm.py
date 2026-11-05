"""Pages whose text layer cannot be trusted are read by a vision model.

A page goes to the vision model when it has no text layer, when it is one full-page image (a
scan with someone else's OCR layer on top), or when much of its visible text has no characters
behind it (text drawn as vector outlines). The model returns Markdown; Tesseract's word boxes
for the same page give each block its position, because the model's output has none.
"""

import base64
import io
import json
import logging
import re

from PIL import Image, ImageChops, ImageDraw

from ..llm import BudgetedLLM
from .pdf_blocks import BULLET, PAGE_SCALE, Block, item_text, number_blocks

log = logging.getLogger(__name__)

MIN_TEXT_LAYER_CHARS = 20
FULL_PAGE_IMAGE = 0.7  # share of the page an image must cover to count as a scan
GLYPH_PATHS = 50  # small vector shapes the size of letters
INK_WITHOUT_TEXT = 0.25  # share of dark pixels with no text-layer character under them
VLM_SCALE = 2.0
LAYOUT_MAX_TOKENS = 8000  # every word of a dense page plus a box per block

PROMPT = """Read this document page and split it into its layout blocks in reading order: each heading,
paragraph, list, table, footnote and caption is one block. Leave out running headers, running footers and page
numbers. For each block give:
- "text": the block's words exactly as printed, in Markdown: a heading starts with # marks by its visual rank
  (# for the largest), a table is a Markdown table, a list has one "- " item per line. Keep every word.
- "bbox": [x0, y0, x1, y1] on a 0-1000 scale (0,0 the page's top-left corner, 1000,1000 its bottom-right),
  tight around the block's ink.
Reply with JSON only: {"blocks": [{"text": "...", "bbox": [x0, y0, x1, y1]}]}"""


def assess_page(plumber_page, image: Image.Image) -> str | None:
    """Why this page's text layer cannot be used as it is, or None when it can.
    `image` is the page rendered at PAGE_SCALE."""
    chars = plumber_page.chars
    if len(chars) < MIN_TEXT_LAYER_CHARS:
        return "no_text"
    page_area = float(plumber_page.width * plumber_page.height) or 1.0
    for im in plumber_page.images:
        if (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) / page_area >= FULL_PAGE_IMAGE:
            return "scanned"
    glyphs = sum(1 for c in plumber_page.curves if 2 < c["bottom"] - c["top"] < 20 and c["x1"] - c["x0"] < 20)
    if glyphs >= GLYPH_PATHS:
        return "outlined_text"
    ink = image.convert("L").point(lambda v: 255 if v < 128 else 0)
    covered = Image.new("L", image.size, 0)
    draw, k = ImageDraw.Draw(covered), PAGE_SCALE
    for c in chars:
        draw.rectangle(
            [(c["x0"] - 2) * k, (c["top"] - 2) * k, (c["x1"] + 2) * k, (c["bottom"] + 2) * k], fill=255
        )
    for im in plumber_page.images:
        draw.rectangle([im["x0"] * k, im["top"] * k, im["x1"] * k, im["bottom"] * k], fill=255)
    dark = ink.histogram()[255]
    uncovered = ImageChops.subtract(ink, covered).histogram()[255]
    if dark and uncovered / dark >= INK_WITHOUT_TEXT:
        return "outlined_text"
    return None


class VisionReader:
    """Transcribes page images through the budget guard (cached, capped per provider)."""

    def __init__(self, llm: BudgetedLLM, model: str, concurrency: int = 8):
        self.llm, self.model, self.concurrency = llm, model, concurrency

    def read_layout(self, page_no: int, png: bytes) -> list[dict]:
        """The page's blocks in reading order, each {"text": Markdown, "bbox": [x0, y0, x1, y1] on 0-1000}."""
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        resp = self.llm.chat(
            self.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            max_tokens=LAYOUT_MAX_TOKENS,
            purpose="index.page_vlm",
            response_format={"type": "json_object"},
            **self._options(),
        )
        blocks = json.loads(resp.choices[0].message.content or "{}").get("blocks")
        if not isinstance(blocks, list):
            raise ValueError(f"vision model returned no blocks for page {page_no}")
        return [b for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)]

    def _options(self) -> dict:
        # both default to reasoning before answering: 2-8x the time for the same transcription
        if self.llm.provider == "deepinfra":
            return {"temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        return {"reasoning_effort": "low"}


# ---------- Markdown -> blocks ----------

_FENCE = re.compile(r"^```\w*\s*$")
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BOLD_LINE = re.compile(r"^\*\*([^*]+)\*\*$")
MAX_BOLD_HEADING = 120


def markdown_units(md: str) -> list[tuple[str, str]]:
    """(type, text) in reading order; headings keep their # level, like text-layer title blocks."""
    units: list[tuple[str, str]] = []
    para: list[str] = []
    table: list[str] = []
    items: list[str] = []

    def flush():
        if para:
            text = " ".join(para)
            bold = _BOLD_LINE.match(text)
            # a short line set entirely in bold is a heading the reader did not mark as one
            if bold and len(text) <= MAX_BOLD_HEADING:
                units.append(("title", "### " + bold.group(1).strip()))
            else:
                units.append(("text", text))
        if table:
            units.append(("table", "\n".join(table)))
        if items:
            units.append(("list", "\n".join(items)))
        para.clear()
        table.clear()
        items.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if _FENCE.match(line) or _RULE.match(line):
            continue
        if not line:
            flush()
            continue
        if m := _HEADING.match(line):
            flush()
            title = m.group(2).strip().strip("*").strip()
            if title:
                units.append(("title", f"{m.group(1)} {title}"))
        elif line.startswith("|"):
            if para or items:
                flush()
            table.append(line)
        elif BULLET.match(line):
            if para or table:
                flush()
            items.append("- " + item_text(line))
        else:
            if table or items:
                flush()
            para.append(line)
    flush()
    return units


def read_page(reader: VisionReader, page_no: int, vlm_png: bytes) -> list[Block]:
    """Blocks for one page from a single vision-model call: the text of each block and where it sits (its box
    on a 0-1000 scale, converted to PAGE_SCALE page pixels). A block whose box is missing or malformed takes the
    whole page, so a citation of it highlights the page rather than its lines."""
    width, height = Image.open(io.BytesIO(vlm_png)).size
    page_w, page_h = width * PAGE_SCALE / VLM_SCALE, height * PAGE_SCALE / VLM_SCALE
    blocks = []
    for item in reader.read_layout(page_no, vlm_png):
        box = _page_box(item.get("bbox"), page_w, page_h)
        for kind, text in markdown_units(item["text"]):
            blocks.append(Block(page_no, kind, text, box[:]))
    if not blocks:
        raise ValueError(f"vision model returned no text for page {page_no}")
    return number_blocks(blocks)


def _page_box(bbox, page_w: float, page_h: float) -> list[int]:
    whole = [0, 0, round(page_w), round(page_h)]
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
        return whole
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1000.0) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        return whole
    return [
        round(x0 * page_w / 1000),
        round(y0 * page_h / 1000),
        round(x1 * page_w / 1000),
        round(y1 * page_h / 1000),
    ]

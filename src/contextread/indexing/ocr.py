"""Blocks for pages without a text layer, from Tesseract's TSV output (block > paragraph > line
> word). Boxes are converted from the OCR render scale to PAGE_SCALE page pixels."""

import csv
import io
import re
import shutil
import subprocess

from .pdf_blocks import BULLET, PAGE_SCALE, Block, item_text, number_blocks

OCR_SCALE = 300 / 72  # render at 300 dpi for recognition
MIN_CONFIDENCE = 30


class OcrUnavailable(RuntimeError):
    pass


def tesseract_tsv(png: bytes, lang: str = "eng", timeout: int = 120) -> list[dict]:
    exe = shutil.which("tesseract")
    if exe is None:
        raise OcrUnavailable("tesseract is not installed")
    proc = subprocess.run(
        [exe, "stdin", "stdout", "-l", lang, "--psm", "3", "tsv"],
        input=png,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise OcrUnavailable(f"tesseract failed: {proc.stderr.decode(errors='replace')[:300]}")
    reader = csv.DictReader(
        io.StringIO(proc.stdout.decode("utf-8", errors="replace")), delimiter="\t", quoting=csv.QUOTE_NONE
    )
    return list(reader)


def ocr_blocks(png: bytes, page_no: int) -> list[Block]:
    rows = tesseract_tsv(png)
    k = PAGE_SCALE / OCR_SCALE
    paragraphs: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("level") != "5" or not (r.get("text") or "").strip():
            continue
        if float(r.get("conf") or -1) < MIN_CONFIDENCE:
            continue
        key = (r["block_num"], r["par_num"])
        x0, y0 = int(r["left"]), int(r["top"])
        x1, y1 = x0 + int(r["width"]), y0 + int(r["height"])
        p = paragraphs.setdefault(key, {"lines": {}, "box": [x0, y0, x1, y1]})
        p["lines"].setdefault(r["line_num"], []).append(r["text"])
        b = p["box"]
        p["box"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]

    blocks = []
    for p in paragraphs.values():
        lines = [" ".join(words) for _, words in sorted(p["lines"].items(), key=lambda kv: int(kv[0]))]
        text = "\n".join(lines) if all(BULLET.match(ln) for ln in lines) else " ".join(lines)
        kind = "list" if BULLET.match(lines[0]) else "text"
        if kind == "list":
            text = "\n".join("- " + item_text(ln) for ln in lines)
        blocks.append(
            Block(page_no, kind, re.sub(r"\s+\n", "\n", text).strip(), [round(v * k) for v in p["box"]])
        )
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return number_blocks(blocks)

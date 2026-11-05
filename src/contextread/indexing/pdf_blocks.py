"""Blocks from a PDF page's text layer: words -> lines -> blocks (title, text, list, table,
references, image), each with a bbox in page pixels at PAGE_SCALE (1.5 = 108 dpi, the
coordinate frame citations are resolved in)."""

import re
from collections import Counter
from dataclasses import dataclass, field

PAGE_SCALE = 1.5

BULLET = re.compile(r"^(?:[•●▪◦‣∙·\-–—*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
GLYPH_BULLETS = "•●▪◦‣∙·"
GLYPH = re.compile(r"^[•●▪◦‣∙·\-–—*]\s+")


def item_text(line: str) -> str:
    """A list item's text without its bullet glyph. Enumerators such as "(a)" or "1." are kept:
    legal documents refer to clauses by them ("Section 6.01(d)")."""
    return GLYPH.sub("", line)


FOOTNOTE = re.compile(r"^\(?\d{1,2}\)\s|^[*†‡]\s")
NUMERIC_CELL = re.compile(
    r"^[\s$€£(]*[-–—]?[\d.,]+\s?%?\s*\)?\s*(?:pts?|ppt|x|bps)?$|^[-–—]+$|^n/?m$|^\$$", re.I
)
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.I)


@dataclass
class Block:
    page: int
    type: str
    text: str
    bbox: list[int]
    block_id: str = ""


@dataclass
class _Line:
    words: list[dict]
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: float
    cells: list[str] = field(default_factory=list)
    cell_x0: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.cells)


def _is_bold(fontname: str) -> bool:
    return bool(re.search(r"bold|black|heavy|semibold", fontname, re.I))


def _lines(words: list[dict]) -> list[_Line]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(w["top"] - rows[-1][0]["top"]) <= 0.4 * max(w["size"], 1):
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        size = max(w["size"] for w in row)
        cells, cell_x0, current = [], [], [row[0]]
        for prev, w in zip(row, row[1:], strict=False):
            # a gap much wider than a space (~0.25 em) separates table cells
            if w["x0"] - prev["x1"] > 0.8 * size:
                cells.append(" ".join(x["text"] for x in current))
                cell_x0.append(current[0]["x0"])
                current = [w]
            else:
                current.append(w)
        cells.append(" ".join(x["text"] for x in current))
        cell_x0.append(current[0]["x0"])
        n_chars = sum(len(w["text"]) for w in row)
        bold = sum(len(w["text"]) for w in row if _is_bold(w.get("fontname", ""))) / max(n_chars, 1)
        out.append(
            _Line(
                row,
                min(w["x0"] for w in row),
                max(w["x1"] for w in row),
                min(w["top"] for w in row),
                max(w["bottom"] for w in row),
                size,
                bold,
                cells,
                cell_x0,
            )
        )
    return out


def _tabular(line: _Line) -> bool:
    numeric = sum(1 for c in line.cells if NUMERIC_CELL.match(c) and c != "$")
    return len(line.cells) >= 3 and numeric >= 2


def _heading(line: _Line, body_size: float) -> bool:
    words = line.text.split()
    if not words or len(words) > 14 or line.text.rstrip().endswith((".", ",", ";")):
        return False
    letters = [c for c in line.text if c.isalpha()]
    caps = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
    return line.size >= body_size * 1.15 or (line.bold > 0.8 and (caps or len(words) <= 8))


def _kind(line: _Line, body_size: float) -> str:
    if _tabular(line):
        return "table"
    if line.size < body_size * 0.9 and FOOTNOTE.match(line.text):
        return "references"
    if BULLET.match(line.text) or line.text[:1] in GLYPH_BULLETS:
        return "list"
    if _heading(line, body_size):
        return "title"
    return "text"


def _markdown(kind: str, lines: list[_Line], heading_level: int) -> str:
    if kind == "title":
        return "#" * heading_level + " " + " ".join(ln.text for ln in lines)
    if kind == "table":
        # a lone "$" is a currency column of its own in the PDF; keep it with its number
        cell_rows = [_join_currency(ln.cells) for ln in lines]
        width = max(len(c) for c in cell_rows)
        rows = [c + [""] * (width - len(c)) for c in cell_rows]
        md = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(md)
    if kind == "list":
        items: list[str] = []
        base = min(ln.x0 for ln in lines)
        for ln in lines:
            text = ln.text
            if BULLET.match(text) or text[:1] in GLYPH_BULLETS:
                text = item_text(text) if BULLET.match(text) else text[1:].lstrip()
                indent = "  " * min(3, round((ln.x0 - base) / (2 * ln.size)))
                items.append(f"{indent}- {text}")
            elif items:
                items[-1] += " " + text  # wrapped continuation of the previous item
            else:
                items.append(text)
        return "\n".join(items)
    return " ".join(ln.text for ln in lines)


def _join_currency(cells: list[str]) -> list[str]:
    out: list[str] = []
    for c in cells:
        if out and out[-1] in ("$", "€", "£"):
            out[-1] = f"{out[-1]} {c}"
        else:
            out.append(c)
    return out


def _merge_split_tables(groups: list[tuple[str, list[_Line]]]) -> list[tuple[str, list[_Line]]]:
    """table, short label line(s), table -> one table (section labels like 'Segment operating income:')."""
    out: list[tuple[str, list[_Line]]] = []
    i = 0
    while i < len(groups):
        kind, lines = groups[i]
        if (kind == "table" and out and out[-1][0] == "table") or (
            kind in ("text", "title")
            and out
            and out[-1][0] == "table"
            and i + 1 < len(groups)
            and groups[i + 1][0] == "table"
            and len(lines) <= 2
            and all(len(ln.text.split()) <= 8 for ln in lines)
        ):
            out[-1][1].extend(lines)
        else:
            out.append((kind, list(lines)))
        i += 1
    return out


def _bbox(lines: list[_Line]) -> list[int]:
    s = PAGE_SCALE
    return [
        round(min(ln.x0 for ln in lines) * s),
        round(min(ln.top for ln in lines) * s),
        round(max(ln.x1 for ln in lines) * s),
        round(max(ln.bottom for ln in lines) * s),
    ]


def body_font_size(pages) -> float:
    sizes: Counter = Counter()
    for page in pages:
        for ch in page.chars:
            sizes[round(ch["size"], 1)] += 1
    return sizes.most_common(1)[0][0] if sizes else 10.0


def page_blocks(page, page_no: int, body_size: float) -> list[Block]:
    """Blocks of one pdfplumber page in reading order (top to bottom)."""
    words = page.extract_words(extra_attrs=["size", "fontname"], x_tolerance=1.5, keep_blank_chars=False)
    margin = 0.06 * page.height
    lines = [
        ln
        for ln in _lines(words)
        # running page numbers in the top/bottom margin are not content
        if not (PAGE_NUMBER.match(ln.text) and (ln.top < margin or ln.bottom > page.height - margin))
    ]
    groups: list[tuple[str, list[_Line]]] = []
    for ln in lines:
        kind = _kind(ln, body_size)
        if groups:
            prev_kind, prev = groups[-1]
            last = prev[-1]
            gap = ln.top - last.bottom
            close = gap <= 0.9 * max(ln.size, last.size)
            same_style = abs(ln.size - last.size) <= 1.0 and (ln.bold > 0.5) == (last.bold > 0.5)
            if (
                prev_kind in ("list", "references")
                and kind == "text"
                and close
                and ln.x0 > prev[0].x0 + 0.5 * ln.size
            ):
                kind = prev_kind  # wrapped line of a bullet item or footnote
            if prev_kind == "table" and kind in ("text", "title") and close and len(ln.cells) <= 2:
                kind = "table"  # a row label without numbers inside a table
            if kind == "list" and prev_kind == "list" and gap <= 1.6 * ln.size:
                prev.append(ln)
                continue
            new_footnote = kind == "references" and FOOTNOTE.match(ln.text)
            if kind == prev_kind and close and (kind == "table" or same_style) and not new_footnote:
                prev.append(ln)
                continue
        groups.append((kind, [ln]))

    blocks = []
    for kind, glines in _merge_split_tables(groups):
        level = 1 if glines[0].size >= body_size * 1.25 else 2 if glines[0].bold > 0.5 else 3
        blocks.append(Block(page_no, kind, _markdown(kind, glines, level), _bbox(glines)))
    for img in page.images:
        x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
        if (x1 - x0) * (bottom - top) < 0.01 * page.width * page.height:
            continue  # rules, logos and other decoration
        s = PAGE_SCALE
        blocks.append(
            Block(page_no, "image", "", [round(x0 * s), round(top * s), round(x1 * s), round(bottom * s)])
        )
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return number_blocks(blocks)


def number_blocks(blocks: list[Block]) -> list[Block]:
    """PageIndex-style ids: p<page>_<type>_<n>, n counting every block on the page."""
    for n, b in enumerate(blocks, 1):
        b.block_id = f"p{b.page}_{b.type}_{n}"
    return blocks

"""Document tree: structure from PageIndex Flash (MIT layout analysis, plus its LLM expand step), node text from
our blocks, node summaries and the document description from the budget-guarded LLM."""

import asyncio
import contextvars
import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from ..config import get_settings
from ..llm import BudgetedLLM, BudgetExceeded
from .pdf_blocks import Block

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # flash imports litellm; no network fetch

log = logging.getLogger(__name__)

INTRO = " (intro)"
MAX_NODE_CHARS = 80_000  # ~20k tokens, the recorded max_token_num_each_node
PAGES_PER_FALLBACK_NODE = 10


_EXPAND_LLM: contextvars.ContextVar = contextvars.ContextVar("flash_expand_llm")
EXPAND_MAX_TOKENS = 6000


async def _guarded_completion(model, prompt, **_params):
    """Flash's model call for expand, sent through the budget guard of the document being indexed."""
    llm = _EXPAND_LLM.get()
    kwargs = _call_options(model)
    resp = await asyncio.to_thread(
        llm.chat,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=EXPAND_MAX_TOKENS,
        purpose="index.flash_expand",
        **kwargs,
    )
    return resp.choices[0].message.content


def flash_structure(pdf_bytes: bytes) -> tuple[list[dict], str | None, str]:
    """(structure, doc_title, toc_source) from layout statistics alone (no LLM)."""
    from pageindex.flash.api import page_index_flash

    result = page_index_flash(io.BytesIO(pdf_bytes), summary=False, optimize=False)
    return result.get("structure") or [], result.get("doc_title"), result.get("toc_source", "unreadable")


def expand_structure(structure: list[dict], page_texts: list[str], llm: BudgetedLLM, model: str) -> int:
    """Split every section over 5 pages into the subsections printed in it (flash's expand step, merge off),
    whatever built the tree: layout, title blocks or page groups. Documents that set their headings like body
    text (most contracts) otherwise keep whole articles, or the whole body, as one node. The model proposes
    headings with their pages; flash keeps only those printed on the page given. Returns how many sections
    were split; a failure (budget cap, model error) leaves the tree as it was."""
    from pageindex import tree_optimize
    from pageindex.flash.api import _page_lines

    if not model or model == "off" or not structure:
        return 0
    tree_optimize.llm_acompletion = _guarded_completion  # flash's only model hook for expand
    relabel(structure)  # expand tells sections apart by node_id
    token = _EXPAND_LLM.set(llm)
    try:
        outcome = asyncio.run(
            tree_optimize.optimize(
                structure,
                page_texts,
                _page_lines(page_texts),
                model=model,
                do_merge=False,
                do_expand=True,
                page_count=len(page_texts),
                concurrency=4,
            )
        )
        return outcome["expands"]
    except Exception as e:  # noqa: BLE001 - any expand failure keeps the tree it was given
        log.warning("tree expand failed (%s: %s); keeping the unexpanded tree", type(e).__name__, e)
        return 0
    finally:
        _EXPAND_LLM.reset(token)


def _call_options(model: str) -> dict:
    return {"reasoning_effort": "low"} if get_settings().is_reasoning(model) else {"temperature": 0}


# Long leaves (settings): a leaf over SECTION_HEADINGS_PAGES pages is read for the headings printed in it
# (expand's trigger), in HEADINGS_WINDOW_PAGES-page calls. One still over SECTION_SPLIT_PAGES pages (the
# recorded max_page_num_each_node) and SECTION_SPLIT_CHARS characters of real text (not a run of signature
# pages) is cut into SPLIT_PART_PAGES-page windows, each starting on the page the one before ends on, as the
# reference does. The reference needs 20k tokens (max_token_num_each_node), but its OCR titles such a leaf
# right; ours can carry a garbled title.
WINDOW_PAGE_CHARS = 6_000
HEADINGS_PROMPT = """Below are pages {start}-{end} of a document. The document's outline puts them inside the section
"{section}" (pages {s0}-{e0}), but that outline may have missed headings, so these pages can also hold whole
further sections.

{pages}

List every heading printed on these pages, in reading order, with the page it is printed on:
- Use each heading exactly as printed. Never invent, paraphrase or complete one.
- A heading printed over two lines (a number line, then its title: "SECTION 6" and "Conditions Precedent") is ONE
  heading: give both parts as one title, "SECTION 6 Conditions Precedent".
- Not headings: lettered or numbered clauses inside a section ("(a) Credit Documents.", "(iii) ..."), even with a
  title; defined terms in a list of definitions; cross-references ("as set forth in Section 7.1"); running headers
  and footers, page numbers and table labels.
- "level": 1 for the document's largest divisions (article, part, chapter, a top-level numbered section such as
  "SECTION 7" or "7."), 2 for sections within them ("7.1", "Section 7.01"), 3 for subsections below those.
- If no heading is printed on these pages, return an empty list. That is a valid and expected answer.

Reply with JSON only: {{"headings": [{{"title": "<heading as printed>", "page": <int>, "level": <1-3>}}]}}"""
WINDOW_PROMPT = """Below are pages {start}-{end} of the section "{section}" (pages {s0}-{e0}) of a document. The
section has no headings inside it: it is one long run of entries such as defined terms, glossary terms, schedule
items or numbered paragraphs, or one long stretch of text.

{pages}

Describe these pages so a reader of the document's outline knows whether what they look for is here:
- "title": at most 15 words naming what these pages cover by the entries printed on them, with the entry names
  exactly as printed. For a run of entries, the first and last entry that begin here, e.g.
  "Definitions: Investment to Material Subsidiary"; otherwise the main items or topics.
- "summary": at most 60 words: what these pages cover, naming the main entries and their key figures.
- "entries": every entry (defined term, item, numbered paragraph) that BEGINS on these pages, name only, in order;
  an empty list if the text is not a run of entries.

Reply with JSON only: {{"title": "...", "summary": "...", "entries": ["..."]}}"""


def _windows(start: int, end: int, step: int) -> list[tuple[int, int]]:
    return [(a, min(a + step, end)) for a in range(start, end, step)] or [(start, end)]


def _pages_block(page_texts: list[str], start: int, end: int) -> str:
    return "\n".join(
        f"<page_{p}>\n{page_texts[p - 1][:WINDOW_PAGE_CHARS]}\n</page_{p}>"
        for p in range(start, end + 1)
        if 0 < p <= len(page_texts)
    )


def _ask_json(llm: BudgetedLLM, model: str, prompt: str, purpose: str) -> dict:
    """One JSON reply; {} when the call fails or the budget cap is reached."""
    kwargs = _call_options(model)
    try:
        resp = llm.chat(
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=EXPAND_MAX_TOKENS,
            purpose=purpose,
            response_format={"type": "json_object"},
            **kwargs,
        )
        out = json.loads(resp.choices[0].message.content or "{}")
        return out if isinstance(out, dict) else {}
    except BudgetExceeded:
        log.warning("budget cap reached; %s skipped", purpose)
    except Exception as e:  # noqa: BLE001 - a failed call leaves that part of the tree as it was
        log.warning("%s failed (%s: %s)", purpose, type(e).__name__, e)
    return {}


def _is_long(node: dict, page_texts: list[str], pages: int | None = None, chars: int | None = None) -> bool:
    s = get_settings()
    pages = s.section_split_pages if pages is None else pages
    chars = s.section_split_chars if chars is None else chars
    s0, e0 = node["start_index"], node["end_index"]
    if node.get("nodes") or e0 - s0 + 1 <= pages:
        return False
    return sum(len(page_texts[p - 1]) for p in range(s0, e0 + 1) if 0 < p <= len(page_texts)) >= chars


def _nest(heads: list[tuple[int, int, str]], end: int) -> list[dict]:
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for level, page, title in heads:
        node = {"title": title, "start_index": page, "end_index": end, "nodes": []}
        while stack and stack[-1][0] >= level:
            stack.pop()
        (stack[-1][1]["nodes"] if stack else roots).append(node)
        stack.append((level, node))
    _close_ranges(roots, end)
    return roots


def split_long_leaves(
    structure: list[dict], page_texts: list[str], llm: BudgetedLLM, model: str, workers: int = 4
) -> int:
    """Split long leaves, as the reference splits every node over its max_page_num_each_node (10 pages) and
    max_token_num_each_node (20k tokens).
    1. Every leaf over SECTION_HEADINGS_PAGES pages: the headings printed in it, read in HEADINGS_WINDOW_PAGES-page calls and kept only
       where printed on the page given. Layout analysis misses headings set like body text, and takes a clause
       heading for a section, which leaves whole sections or articles inside the node before them; expand cannot
       recover those, as it asks only for that node's own subsections.
    2. Every leaf still over SECTION_SPLIT_PAGES pages and SECTION_SPLIT_CHARS characters (a definitions article, a
       glossary, a schedule: no headings in it): 5-page windows, each titled and summarised from its own pages by
       the entries it holds. Otherwise the outline shows one node whose summary covers only its first pages.
    Returns how many leaves were split; a failed call leaves that part as it was (a window keeps a page-range
    title)."""
    from pageindex.tree_optimize import heading_key

    if not model or model == "off":
        return 0
    s = get_settings()
    split: set[int] = set()

    # 1. headings printed inside long leaves
    leaves = [n for n, _ in preorder(structure) if _is_long(n, page_texts, s.section_headings_pages, chars=0)]
    jobs = [
        (leaf, a, b)
        for leaf in leaves
        for a, b in _windows(leaf["start_index"], leaf["end_index"], max(s.headings_window_pages - 1, 1))
    ]

    def headings_in(job: tuple[dict, int, int]) -> dict:
        leaf, a, b = job
        prompt = HEADINGS_PROMPT.format(
            start=a,
            end=b,
            section=leaf["title"],
            s0=leaf["start_index"],
            e0=leaf["end_index"],
            pages=_pages_block(page_texts, a, b),
        )
        return _ask_json(llm, model, prompt, "index.long_leaf_headings")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        replies = list(pool.map(headings_in, jobs))
    found: dict[int, dict] = {}
    for (leaf, a, b), out in zip(jobs, replies, strict=True):
        own_key = heading_key(leaf["title"].replace("(intro)", ""))
        for h in out.get("headings") or []:
            title, page, level = (h or {}).get("title"), (h or {}).get("page"), (h or {}).get("level")
            if not isinstance(title, str) or not isinstance(page, int) or not a <= page <= b:
                continue
            key, printed = (
                heading_key(title),
                heading_key(page_texts[page - 1] if page <= len(page_texts) else ""),
            )
            if not key or key not in printed or (key == own_key and page == leaf["start_index"]):
                continue
            if page == leaf["start_index"] and own_key and key.startswith(own_key):
                leaf["title"] = (
                    title.strip()
                )  # the leaf's own heading in full (layout kept only its first line)
                continue
            if page == leaf["start_index"] and 0 <= printed.find(key) < printed.find(own_key):
                continue  # printed above the leaf's own heading: it belongs to the section before
            level = min(max(level, 1), 3) if isinstance(level, int) else 2
            found.setdefault(id(leaf), {}).setdefault(
                (page, key), (page, printed.find(key), level, title.strip())
            )
    for leaf in leaves:
        heads = sorted(found.get(id(leaf), {}).values())
        if len(heads) < 2:
            continue
        s0, e0 = leaf["start_index"], leaf["end_index"]
        children = _nest([(level, page, title) for page, _, level, title in heads], e0)
        if children[0]["start_index"] > s0:  # the leaf's own text before the first heading found
            intro = leaf["title"].replace(INTRO, "").strip() + INTRO
            children.insert(0, {"title": intro, "start_index": s0, "end_index": children[0]["start_index"]})
        leaf["nodes"] = children
        split.add(id(leaf))

    # 2. leaves still long (no headings found, or a long section among them): titled windows
    jobs2 = []
    for node, _ in preorder(structure):
        if _is_long(node, page_texts):
            node["nodes"] = [
                {"title": f"{node['title']} (pages {a}-{b})", "start_index": a, "end_index": b}
                for a, b in _windows(node["start_index"], node["end_index"], max(s.split_part_pages - 1, 1))
            ]
            jobs2 += [(node, w) for w in node["nodes"]]
            split.add(id(node))

    def describe_window(job: tuple[dict, dict]) -> None:
        parent, w = job
        prompt = WINDOW_PROMPT.format(
            start=w["start_index"],
            end=w["end_index"],
            section=parent["title"],
            s0=parent["start_index"],
            e0=parent["end_index"],
            pages=_pages_block(page_texts, w["start_index"], w["end_index"]),
        )
        out = _ask_json(llm, model, prompt, "index.long_leaf_window")
        if isinstance(out.get("title"), str) and out["title"].strip():
            w["title"] = out["title"].strip()
        entries = [str(e).strip() for e in out.get("entries") or [] if str(e).strip()]
        summary = out["summary"].strip() if isinstance(out.get("summary"), str) else ""
        if entries:
            summary = f"{summary} Entries: {'; '.join(entries)}".strip()
        if summary:
            w["summary"] = summary

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(describe_window, jobs2))
    return len(split)


def page_fallback_structure(page_count: int, page_leaves: bool = True) -> list[dict]:
    """Page groups when nothing else gives a structure. Without page_leaves the groups are the leaves, so expand
    can split each into the sections that begin in it."""

    def page_node(p: int) -> dict:
        return {"title": f"Page {p}", "start_index": p, "end_index": p}

    if page_count <= PAGES_PER_FALLBACK_NODE:
        return [page_node(p) for p in range(1, page_count + 1)]
    groups = []
    for start in range(1, page_count + 1, PAGES_PER_FALLBACK_NODE):
        end = min(start + PAGES_PER_FALLBACK_NODE - 1, page_count)
        groups.append(
            {
                "title": f"Pages {start}–{end}",
                "start_index": start,
                "end_index": end,
                **({"nodes": [page_node(p) for p in range(start, end + 1)]} if page_leaves else {}),
            }
        )
    return groups


def wrap_in_root(structure: list[dict], title: str | None, page_count: int) -> list[dict]:
    """A single root titled after the document, as in the reference product's trees."""
    if not title or len(structure) <= 1:
        return structure
    return [{"title": title.strip(), "start_index": 1, "end_index": page_count, "nodes": structure}]


def preorder(structure: list[dict]) -> list[tuple[dict, int]]:
    out = []

    def walk(nodes, depth):
        for n in nodes:
            out.append((n, depth))
            walk(n.get("nodes") or [], depth + 1)

    walk(structure, 1)
    return out


def relabel(structure: list[dict]) -> None:
    for i, (node, _) in enumerate(preorder(structure)):
        node["node_id"] = f"{i:04d}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _anchor(node: dict, blocks: list[Block], start_from: int) -> int:
    """Index in the document's block sequence where this node's section starts."""
    page = node["start_index"]
    want = _norm(node["title"].replace("(intro)", ""))[:60]
    first_on_page = None
    for i in range(start_from, len(blocks)):
        b = blocks[i]
        if b.page < page:
            continue
        if b.page > page:
            break
        if first_on_page is None:
            first_on_page = i
        if want and b.type in ("title", "text") and want[:30] in _norm(b.text):
            return i
    if first_on_page is not None:
        return first_on_page
    return next((i for i, b in enumerate(blocks) if b.page >= page), len(blocks))


def attach_text(structure: list[dict], blocks: list[Block]) -> None:
    """node['text'] = markdown of the blocks from this node's heading to the next node's heading."""
    order = preorder(structure)
    anchors, cursor = [], 0
    for node, _ in order:
        a = max(_anchor(node, blocks, cursor), cursor)
        anchors.append(a)
        cursor = a
    for k, (node, _) in enumerate(order):
        end = anchors[k + 1] if k + 1 < len(order) else len(blocks)
        own = [b for b in blocks[anchors[k] : end] if b.page <= node["end_index"] and b.text]
        node["text"] = "\n\n".join(b.text for b in own)[:MAX_NODE_CHARS]


def md_tree(structure: list[dict]) -> list[dict]:
    return [
        {"title": n["title"], "level": depth, "text": n.get("text", ""), "start_index": n["start_index"]}
        for n, depth in preorder(structure)
    ]


LEVELS_PROMPT = (
    "These are the headings of one document in reading order, each with its page and the Markdown "
    "level a page-by-page reader gave it. That reader saw one page at a time, so its levels are not "
    "consistent across pages. Assign every heading its outline level (1 = top) so that sections nest "
    "correctly across the whole document: numbering schemes (I., A., 1., (a)) and wording show which "
    "headings are siblings. Use 0 for lines that are not section headings (running headers, captions, "
    "case captions, signature lines). Reply as JSON mapping every heading number to its level: "
    '{{"levels": {{"1": 1, "2": 2, ...}}}}.\n\n{items}'
)
_HASHES = re.compile(r"^(#{1,6})\s+")


def _heading_levels(heads: list[tuple[int, int, str]], llm: BudgetedLLM, model: str | None) -> list[int]:
    """Document-wide levels from one LLM pass; the readers' own # levels when that fails."""
    items = "\n".join(
        f"{i}. [p{page}] {'#' * level} {title}" for i, (page, level, title) in enumerate(heads, 1)
    )
    try:
        resp = llm.chat(
            model,
            [{"role": "user", "content": LEVELS_PROMPT.format(items=items)}],
            max_tokens=16 + 10 * len(heads),
            purpose="index.heading_levels",
            temperature=0,
            response_format={"type": "json_object"},
        )
        # keyed by number, so one miscounted entry cannot shift every level after it
        levels = json.loads(resp.choices[0].message.content or "{}").get("levels")
        if isinstance(levels, dict):
            out = [levels.get(str(i)) for i in range(1, len(heads) + 1)]
            if all(isinstance(v, int) for v in out):
                return [max(0, min(v, 6)) for v in out]
        log.warning("heading levels reply did not cover the %d headings; using reader levels", len(heads))
    except BudgetExceeded:
        log.warning("budget cap reached; using reader heading levels")
    except (json.JSONDecodeError, AttributeError):
        log.warning("heading levels reply was not JSON; using reader levels")
    return [level for _, level, _ in heads]


def heading_structure(
    blocks: list[Block], page_count: int, llm: BudgetedLLM, model: str | None = None
) -> list[dict] | None:
    """A tree from the title blocks themselves, for documents whose text layer flash cannot read."""
    heads = []
    for b in blocks:
        m = _HASHES.match(b.text or "") if b.type == "title" else None
        if m:
            heads.append((b.page, len(m.group(1)), b.text[m.end() :].strip()))
    if len(heads) < 2:
        return None
    levels = _heading_levels(heads, llm, model)
    kept = [(lvl, page, title) for lvl, (page, _, title) in zip(levels, heads, strict=True) if lvl > 0]
    if len(kept) < 2:
        return None
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for lvl, page, title in kept:
        node = {"title": title, "start_index": page, "end_index": page_count, "nodes": []}
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        (stack[-1][1]["nodes"] if stack else roots).append(node)
        stack.append((lvl, node))
    _close_ranges(roots, page_count)
    return roots


def _close_ranges(nodes: list[dict], end: int) -> None:
    """A section ends on the page where the next section at its level (or above) starts."""
    for i, node in enumerate(nodes):
        node["end_index"] = nodes[i + 1]["start_index"] if i + 1 < len(nodes) else end
        if node["nodes"]:
            _close_ranges(node["nodes"], node["end_index"])
        else:
            del node["nodes"]


SUMMARY_PROMPT = (
    "Summarize this section of a document in at most 60 words: what it covers and its key facts and figures, "
    "taken only from the section text below. Never invent or estimate a figure, name or date, and never write "
    "placeholders such as $X. If the text holds no more than a heading or a table-of-contents line, say in one "
    "short sentence what the section is from its title and subsections, with no facts. No preamble."
    "\n\nSection title: {title}\nPages {start}-{end}\n{children}\nSection text:\n{text}"
)
DESCRIPTION_PROMPT = (
    "Write a one-sentence description (at most 45 words) of this document, starting with "
    '"This document is". Name the organization and subject when known.\n\nTitle: {title}\nSections:\n{outline}'
)


def section_text(node: dict, limit: int) -> str:
    """The section's whole text, its subsections' included, up to `limit` characters. A parent's own text runs
    only from its heading to its first subsection's, often the heading alone; summarised from that, the model
    has nothing to state and invents figures. The reference takes a node's text from all its pages likewise."""
    parts, size = [], 0
    for n, _ in preorder([node]):
        t = n.get("text") or ""
        if t:
            parts.append(t)
            size += len(t) + 2
            if size >= limit:
                break
    return "\n\n".join(parts)[:limit]


def summarize_nodes(
    structure: list[dict], llm: BudgetedLLM, model: str | None = None, workers: int = 4
) -> int:
    """Fill node['summary'] where an earlier step left none; returns how many nodes have one (stops cleanly at the
    budget cap)."""
    nodes = [n for n, _ in preorder(structure) if not n.get("summary")]
    already = len(preorder(structure)) - len(nodes)

    limit = get_settings().summary_input_chars

    def one(node: dict) -> bool:
        children = node.get("nodes") or []
        child_line = ("Subsections: " + "; ".join(c["title"] for c in children)) if children else ""
        prompt = SUMMARY_PROMPT.format(
            title=node["title"],
            start=node["start_index"],
            end=node["end_index"],
            children=child_line,
            text=section_text(node, limit),
        )
        try:
            resp = llm.chat(
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=150,
                purpose="index.node_summary",
                temperature=0,
            )
        except BudgetExceeded:
            log.warning("budget cap reached; node %s left without a summary", node.get("node_id"))
            return False
        node["summary"] = (resp.choices[0].message.content or "").strip()
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return already + sum(pool.map(one, nodes))


def describe(
    structure: list[dict], title: str | None, llm: BudgetedLLM, model: str | None = None
) -> str | None:
    outline = "\n".join(
        f"{'  ' * (depth - 1)}- {n['title']}: {n.get('summary', '')[:300]}"
        for n, depth in preorder(structure)
        if depth <= 2
    )
    try:
        resp = llm.chat(
            model,
            [
                {
                    "role": "user",
                    "content": DESCRIPTION_PROMPT.format(
                        title=title or "(untitled)", outline=outline[: get_settings().summary_input_chars]
                    ),
                }
            ],
            max_tokens=120,
            purpose="index.doc_description",
            temperature=0,
        )
    except BudgetExceeded:
        log.warning("budget cap reached; document left without a description")
        return None
    return (resp.choices[0].message.content or "").strip()

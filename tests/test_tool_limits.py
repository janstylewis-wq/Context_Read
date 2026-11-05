"""Tool output never cut mid-text: a large structure shrinks summaries (titles and pages kept), and a large
page read drops whole pages with a note."""

import json

from contextread import doctools
from contextread.chat import agent
from contextread.config import Settings


def _tree(sections: int, children: int, summary_chars: int) -> list[dict]:
    return [
        {
            "node_id": f"{i:04d}",
            "title": f"ARTICLE {i}",
            "start_index": i * 10 + 1,
            "end_index": i * 10 + 10,
            "summary": "s" * summary_chars,
            "nodes": [
                {
                    "node_id": f"{i:04d}-{j}",
                    "title": f"Section {i}.{j:02d}",
                    "start_index": i * 10 + 1 + j,
                    "end_index": i * 10 + 2 + j,
                    "summary": "word " * (summary_chars // 5),
                }
                for j in range(children)
            ],
        }
        for i in range(sections)
    ]


def _titles(structure: list[dict]) -> list[tuple]:
    out = []
    for n in structure:
        out.append((n["title"], n["start_index"], n["end_index"]))
        out += _titles(n.get("nodes") or [])
    return out


def test_structure_that_fits_is_unchanged():
    tree = _tree(3, 4, 300)
    fitted, note = doctools.fit_structure(tree, 1_000_000)
    assert fitted is tree and note is None


def test_large_structure_keeps_every_title_and_page_range():
    tree = _tree(20, 30, 400)
    limit = len(json.dumps(tree)) // 3
    fitted, note = doctools.fit_structure(tree, limit)
    assert len(json.dumps(fitted, ensure_ascii=False)) <= limit and note
    assert _titles(fitted) == _titles(tree)
    # deepest sections shorten first: no section keeps a longer summary than its article
    assert all(len(n.get("summary", "")) <= len(a.get("summary", "")) for a in fitted for n in a["nodes"])
    assert any(len(a.get("summary", "")) > 0 for a in fitted)
    assert tree[0]["nodes"][0]["summary"].startswith("word")  # the stored tree is not modified


def test_agent_sees_a_tree_that_fits_whole():
    tree = _tree(10, 10, 300)
    seen = json.loads(agent._model_view({"doc_name": "d", "doc_id": "x", "structure": tree}))
    assert _titles(seen["structure"]) == _titles(tree) and "levels" not in seen and "doc_id" not in seen


def test_page_read_drops_whole_pages_with_a_note():
    content = [
        {"page": p, "text": "x" * 12_000, "type": "text", "block_id": f"p{p}_text_1"} for p in range(1, 11)
    ]
    seen = json.loads(agent._model_view({"doc_name": "d", "returned_pages": "1-10", "content": content}))
    pages = sorted({b["page"] for b in seen["content"]})
    assert pages == list(range(1, len(pages) + 1)) and len(pages) < 10
    assert all(b["text"] == "x" * 12_000 for b in seen["content"])  # no page is cut mid-text
    assert seen["returned_pages"] == f"1-{pages[-1]}" and f"Pages {pages[-1] + 1}-10" in seen["note"]


def test_plan_limits_default_to_unlimited(monkeypatch):
    monkeypatch.delenv("PAGE_LIMIT", raising=False)
    monkeypatch.delenv("CHAT_LIMIT", raising=False)
    s = Settings(_env_file=None)
    assert s.page_limit == 0 and s.chat_limit == 0

"""contextread eval: questions come from pages with real text, answers are judged, and a wrong answer is told
apart by whether its answer page was read."""

import contextlib
import json
from types import SimpleNamespace

from fakes import completion

from contextread import eval as ev
from contextread.chat.agent import ChatResult


class JSONReplies:
    """llm stand-in: answers each call with the next JSON object, recording the prompts."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, model, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return completion(json.dumps(self.replies.pop(0)))


def test_questions_come_from_pages_with_real_text(monkeypatch):
    texts = {1: "short", 2: "x" * 900, 3: "", 4: "y" * 2000}
    monkeypatch.setattr(ev, "_document", lambda owner, ref: ("doc-1", "report.pdf", texts))
    llm = JSONReplies(*[{"question": f"Q{i}?", "answer": f"A{i}"} for i in range(2)])
    qs = ev.make_questions(("user", "u"), ["doc-1"], llm, per_doc=5, workers=1)
    assert [q["page"] for q in qs] == [2, 4]
    assert qs[0] == {
        "doc_id": "doc-1",
        "doc_name": "report.pdf",
        "page": 2,
        "question": "Q0?",
        "answer": "A0",
    }
    assert "page 2 of the document" in llm.prompts[0]


def test_the_same_seed_picks_the_same_pages(monkeypatch):
    texts = {p: "z" * 1000 for p in range(1, 40)}
    monkeypatch.setattr(ev, "_document", lambda owner, ref: ("d", "d.pdf", texts))

    def pages(seed):
        llm = JSONReplies(*[{"question": "Q?", "answer": "A"}] * 3)
        return [
            q["page"] for q in ev.make_questions(("user", "u"), ["d"], llm, per_doc=3, seed=seed, workers=1)
        ]

    assert pages(0) == pages(0) and pages(0) != pages(1)


def test_pages_read_follow_what_a_read_returns():
    calls = [
        {"name": "get_document_structure", "arguments": {"doc_name": "d"}},
        {"name": "get_page_content", "arguments": {"doc_name": "d", "pages": "3,8"}},
        {"name": "get_page_content", "arguments": {"doc_name": "d", "pages": "20-40"}},  # first 10 returned
        {"name": "get_page_content", "arguments": {"doc_name": "d", "pages": "nonsense"}},
    ]
    assert ev.pages_read(calls, 10) == [3, 8, *range(20, 30)]


def _run(monkeypatch, question, answer_text, pages, verdict):
    monkeypatch.setattr(ev, "session_scope", contextlib.nullcontext)
    monkeypatch.setattr(ev.doctools, "resolve", lambda s, owner, ref: SimpleNamespace(id="doc-1"))
    calls = [{"name": "get_page_content", "arguments": {"pages": pages}}]
    monkeypatch.setattr(
        ev, "run_chat", lambda *a, **k: ChatResult(text=answer_text, tool_calls=calls, model="m")
    )
    return ev.run_questions(("user", "u"), [question], JSONReplies({"correct": verdict}), workers=1)[0]


def test_a_wrong_answer_is_a_missed_page_or_a_misread(monkeypatch):
    q = {"doc_name": "d.pdf", "page": 7, "question": "Q?", "answer": "A"}
    assert _run(monkeypatch, q, "A.", "7", True)["outcome"] == "right"
    assert _run(monkeypatch, q, "B.", "2-4", False)["outcome"] == "missed page"
    assert _run(monkeypatch, q, "B.", "6-8", False)["outcome"] == "misread"
    hand_written = {"doc_name": "d.pdf", "question": "Q?", "answer": "A"}  # no page given
    assert _run(monkeypatch, hand_written, "B.", "1", False)["outcome"] == "wrong"


def test_report_and_compare():
    def r(question, right, page, read, outcome):
        return {
            "doc_name": "d.pdf",
            "question": question,
            "answer": "A",
            "answer_given": "B",
            "page": page,
            "pages_read": read,
            "right": right,
            "steps": 2,
            "seconds": 5.0,
            "outcome": outcome,
        }

    before = [r("Q1?", True, 1, [1], "right"), r("Q2?", False, 9, [2], "missed page")]
    after = [r("Q1?", False, 1, [1], "misread"), r("Q2?", True, 9, [9], "right")]
    text = ev.report(before, 0.05)
    assert text.startswith("2 questions: right 1/2 (50%)") and "answer page read: 1/2" in text
    assert "[missed page] d.pdf p9, read 2: Q2?" in text and "spent $0.050" in text
    diff = ev.compare(before, after)
    assert "right 1 -> 1" in diff and "broke: d.pdf: Q1?" in diff and "fixed: d.pdf: Q2?" in diff

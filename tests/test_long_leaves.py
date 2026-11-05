"""Long leaves: headings printed inside them become children (checked against the page); a leaf with none is
cut into titled 5-page windows; short leaves are left alone."""

import json

from fakes import completion

from contextread.indexing import tree


class ScriptedLLM:
    """Answers each prompt kind with a function of the prompt."""

    def __init__(self, headings=None, window=None):
        self.headings, self.window, self.calls = headings, window, []

    def chat(self, model, messages, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append(kwargs.get("purpose"))
        if kwargs.get("purpose") == "index.long_leaf_headings":
            out = self.headings(prompt) if self.headings else {"headings": []}
        else:
            out = self.window(prompt) if self.window else {}
        return completion(json.dumps(out))


def _pages(n: int, printed: dict[int, str] | None = None) -> list[str]:
    return [(printed or {}).get(p, "") + "\n" + "body text " * 1_000 for p in range(1, n + 1)]


def test_headings_printed_inside_become_children():
    texts = _pages(
        40, {1: "ARTICLE VI\nIndemnity", 5: "SECTION 6.01 Notices", 20: "ARTICLE VII\nMiscellaneous"}
    )
    structure = [{"title": "ARTICLE VI", "start_index": 1, "end_index": 40}]

    def headings(prompt):
        found = [
            {"title": "ARTICLE VI Indemnity", "page": 1, "level": 1},
            {"title": "SECTION 6.01 Notices", "page": 5, "level": 2},
            {"title": "ARTICLE VII Miscellaneous", "page": 20, "level": 1},
            {"title": "ARTICLE IX Invented", "page": 25, "level": 1},  # not printed there: dropped
        ]
        return {"headings": [h for h in found if f"<page_{h['page']}>" in prompt]}

    split = tree.split_long_leaves(structure, texts, ScriptedLLM(headings=headings), "gpt-6-luna")
    leaf = structure[0]
    assert split >= 1 and leaf["title"] == "ARTICLE VI Indemnity"  # takes its full printed heading
    titles = [(n["title"], n["start_index"], n["end_index"]) for n, _ in tree.preorder(leaf["nodes"])]
    assert ("SECTION 6.01 Notices", 5, 20) in titles
    assert ("ARTICLE VII Miscellaneous", 20, 40) in [t[:3] for t in titles]
    assert not any("Invented" in t[0] for t in titles)


def test_leaf_without_headings_becomes_titled_windows():
    texts = _pages(28)
    structure = [{"title": "Definitions", "start_index": 1, "end_index": 28}]

    def window(prompt):
        start = int(prompt.split("Below are pages ")[1].split("-")[0])
        return {"title": f"Definitions: T{start} to U{start}", "summary": "Terms.", "entries": [f"T{start}"]}

    tree.split_long_leaves(structure, texts, ScriptedLLM(window=window), "gpt-6-luna")
    kids = structure[0]["nodes"]
    assert [(k["start_index"], k["end_index"]) for k in kids] == [
        (1, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (17, 21),
        (21, 25),
        (25, 28),
    ]
    assert kids[2]["title"] == "Definitions: T9 to U9" and kids[2]["summary"] == "Terms. Entries: T9"


def test_short_leaves_and_off_model_are_left_alone():
    texts = _pages(40)
    short = [{"title": "Section 2", "start_index": 1, "end_index": 5}]
    llm = ScriptedLLM()
    assert tree.split_long_leaves(short, texts, llm, "gpt-6-luna") == 0 and llm.calls == []
    long = [{"title": "Definitions", "start_index": 1, "end_index": 40}]
    assert tree.split_long_leaves(long, texts, llm, "off") == 0 and "nodes" not in long[0]


def test_failed_window_keeps_a_page_title():
    class Broken(ScriptedLLM):
        def chat(self, model, messages, **kwargs):
            if kwargs.get("purpose") == "index.long_leaf_window":
                raise RuntimeError("model down")
            return super().chat(model, messages, **kwargs)

    structure = [{"title": "Definitions", "start_index": 1, "end_index": 20}]
    tree.split_long_leaves(structure, _pages(20), Broken(), "gpt-6-luna")
    assert structure[0]["nodes"][0]["title"] == "Definitions (pages 1-5)"
    assert "summary" not in structure[0]["nodes"][0]


def test_mid_size_leaf_is_read_for_headings_but_not_cut_into_windows():
    texts = _pages(8, {3: "Section 2.07. Repayment of Loans"})
    structure = [{"title": "Swing Line Loans", "start_index": 1, "end_index": 8}]

    def headings(prompt):
        return {
            "headings": [
                {"title": "Section 2.07. Repayment of Loans", "page": 3, "level": 2},
                {"title": "Section 2.07. Repayment of Loans", "page": 3, "level": 2},  # repeated: kept once
            ]
        }

    llm = ScriptedLLM(headings=headings)
    tree.split_long_leaves(structure, texts, llm, "gpt-6-luna")
    assert "index.long_leaf_window" not in llm.calls
    assert "nodes" not in structure[0]  # one heading is not enough to split on

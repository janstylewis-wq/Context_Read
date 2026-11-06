"""A document with a table of contents takes its outline from it (PageIndex classic): entries placed on their real
pages, checked there; otherwise, or when too few entries are found on their page, the layout tree is kept."""

from fakes import completion
from test_indexing import DISNEY, upload

from contextread.indexing import pipeline, tree


class Classic:
    """pageindex.page_index_classic stand-in: a contents page found, three entries placed on real pages."""

    def __init__(self, accuracy=1.0):
        self.accuracy = accuracy

    def check_toc(self, page_list, opt):
        return {"toc_content": "ARTICLE I ..... 1", "toc_page_list": [1], "page_index_given_in_toc": "yes"}

    def process_toc_with_page_numbers(self, *a, **k):
        return [
            {"structure": "1", "title": "ARTICLE I Definitions", "physical_index": 3},
            {"structure": "1.1", "title": "Section 1.01 Defined Terms", "physical_index": 3},
            {"structure": "2", "title": "ARTICLE II The Credits", "physical_index": 9},
        ]

    def validate_and_truncate_physical_indices(self, items, n, start_index=1, logger=None):
        return items

    async def verify_toc(self, page_list, items, start_index=1, model=None):
        return self.accuracy, []

    async def check_title_appearance_in_start_concurrent(self, items, page_list, model=None, logger=None):
        return [{**i, "appear_start": "yes"} for i in items]


def _contents(monkeypatch, classic, pages=12):
    import pageindex

    monkeypatch.setattr(pageindex, "page_index_classic", classic, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pageindex.page_index_classic", classic)
    return tree.contents_structure(["text"] * pages, llm=None, model="m")


def test_the_outline_comes_from_the_contents(monkeypatch):
    out = _contents(monkeypatch, Classic())
    titles = [(n["title"], n["start_index"], n["end_index"]) for n, _ in tree.preorder(out)]
    assert titles[0] == ("Preface", 1, 2)  # the cover and the contents pages
    assert ("ARTICLE II The Credits", 9, 12) in titles
    assert all(set(n) <= {"title", "start_index", "end_index", "nodes"} for n, _ in tree.preorder(out))


def test_contents_too_few_of_whose_entries_are_found_are_not_used(monkeypatch):
    assert _contents(monkeypatch, Classic(accuracy=0.5)) is None


def test_the_contents_outline_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("CONTENTS_OUTLINE", "false")
    tree.get_settings.cache_clear()
    try:
        assert _contents(monkeypatch, Classic()) is None
    finally:
        monkeypatch.delenv("CONTENTS_OUTLINE")
        tree.get_settings.cache_clear()


def test_classic_calls_go_through_the_budget_guard():
    class LLM:
        def __init__(self):
            self.calls = []

        def chat(self, model, messages, **kw):
            self.calls.append((model, messages, kw))
            return completion('{"toc_detected": "yes"}')

    llm = LLM()
    token = tree._CONTENTS_LLM.set(llm)
    try:
        text, finish = tree._contents_completion("m", "Is this a contents page?", return_finish_reason=True)
    finally:
        tree._CONTENTS_LLM.reset(token)
    assert text == '{"toc_detected": "yes"}' and finish == "finished"
    assert llm.calls[0][2]["purpose"] == "index.contents"


def test_a_contents_tree_skips_expand_and_the_headings_pass(
    engine, clean_files, fake_llm, bucket, monkeypatch
):
    llm, _ = fake_llm
    seen = {}
    monkeypatch.setattr(
        tree, "contents_structure", lambda *a, **k: [{"title": "Results", "start_index": 1, "end_index": 22}]
    )
    monkeypatch.setattr(tree, "expand_structure", lambda *a, **k: seen.setdefault("expand", True) and 0)
    real = tree.split_long_leaves
    monkeypatch.setattr(
        tree,
        "split_long_leaves",
        lambda *a, **k: seen.setdefault("headings", k.get("headings")) and real(*a, **k),
    )
    reg = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    stats = pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)
    assert stats["toc_source"] == "contents" and "expand" not in seen and seen["headings"] is False

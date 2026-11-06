"""contextread show prints a document as the index holds it; contextread check names what is not ready."""

from fakes import completion
from test_indexing import DISNEY, upload

from contextread import diagnostics
from contextread.indexing import pipeline


def test_show_prints_the_tree_and_chosen_pages(engine, clean_files, fake_llm, bucket, capsys):
    llm, _ = fake_llm
    reg = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)
    diagnostics.show(("user", "user-test"), "q1-fy25-earnings.pdf", summaries=True, pages="13")
    out = capsys.readouterr().out
    assert "22 pages (read as: 22 text)" in out
    assert "- [0000] THE WALT DISNEY COMPANY REPORTS FIRST QUARTER EARNINGS FOR FISCAL 2025 (p1-22)" in out
    assert "A short factual summary." in out  # --summaries
    assert "===== page 13 (read as text;" in out and "===== page 14" not in out


class Models:
    """BudgetedLLM stand-in: one model name is refused, the rest answer."""

    cap = 10.0

    def __init__(self, refused):
        self.refused, self.calls = refused, []

    def chat(self, model, messages, **kwargs):
        self.calls.append((model, kwargs))
        if model == self.refused:
            raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
        return completion("ok")

    def spent(self):
        return 0.0


def test_check_tries_each_model_once_and_hints_at_reasoning_models(monkeypatch, capsys):
    from contextread import llm as llm_module
    from contextread.config import Settings

    fake = Models(refused="o3-mini")
    monkeypatch.setattr(llm_module, "BudgetedLLM", lambda **kw: fake)
    s = Settings(
        _env_file=None,
        APP_PG_DSN="postgresql://x@localhost/x",
        OPENAI_API_KEY="k",
        CHAT_MODEL="o3-mini",
        LLM_DEFAULT_MODEL="gpt-4o-mini",
        OCR_VLM_MODEL="gpt-4o-mini",
        TREE_EXPAND_MODEL="off",
    )
    assert diagnostics._models(s) is False
    out = capsys.readouterr().out
    assert [m for m, _ in fake.calls] == ["o3-mini", "gpt-4o-mini"]  # each distinct model called once
    assert "[FAIL] CHAT_MODEL: o3-mini" in out and "REASONING_MODELS" in out
    assert "[ok  ] OCR_VLM_MODEL: gpt-4o-mini answers (plain model)" in out
    assert "[note] TREE_EXPAND_MODEL: off" in out


def test_reindex_builds_the_tree_again_instead_of_copying_a_duplicate(engine, clean_files, fake_llm, bucket):
    llm, client = fake_llm
    first = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    pipeline.index_document(first["doc_id"], llm=llm, engine=engine)
    again = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    assert "duplicate_of" in pipeline.index_document(again["doc_id"], llm=llm, engine=engine)
    calls = len(client.calls)
    stats = pipeline.index_document(again["doc_id"], llm=llm, engine=engine, reuse_duplicate=False)
    assert "duplicate_of" not in stats and len(client.calls) > calls  # summaries written again

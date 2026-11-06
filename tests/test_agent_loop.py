"""The chat loop: a reply that only announces more reading is not taken as the answer; page reads tell the model
which pages it has read; model calls are cut off and retried instead of hanging."""

import json

from fakes import completion, tool_call_completion

from contextread.chat import agent
from contextread.config import Settings
from contextread.llm import budget


class Script:
    """llm.chat stand-in answering from a list, recording each conversation it was sent."""

    def __init__(self, *replies):
        self.replies, self.convos, self.models = list(replies), [], []

    def chat(self, model, convo, **kwargs):
        self.convos.append([dict(m) for m in convo])
        self.models.append((model, kwargs.get("reasoning_effort")))
        return self.replies.pop(0)


def _run(llm, monkeypatch, output=None, model=None):
    monkeypatch.setattr(agent, "_system_prompt", lambda *a, **k: "system")
    monkeypatch.setattr(agent, "run_tool", lambda *a, **k: output or {})
    return agent.run_chat(("user", "u"), [{"role": "user", "content": "Q?"}], llm, model=model)


def test_a_planning_note_is_not_taken_as_the_answer(monkeypatch):
    llm = Script(
        completion("Need pages maybe discussion tubes ovaries preceding/follow."),
        completion("They were normal."),
    )
    result = _run(llm, monkeypatch)
    assert result.text == "They were normal."
    assert llm.convos[1][-1] == {"role": "user", "content": agent.CONTINUE}


def test_a_short_real_answer_is_returned_as_is(monkeypatch):
    llm = Script(completion("No. He testified that he never analysed polypropylene."))
    assert _run(llm, monkeypatch).text.startswith("No. He testified") and len(llm.convos) == 1


def test_page_reads_tell_the_model_what_it_has_not_read(monkeypatch):
    output = {"doc_name": "brief.pdf", "total_pages": 11, "returned_pages": "1-4", "content": []}
    llm = Script(
        tool_call_completion("get_page_content", {"doc_name": "brief.pdf", "pages": "1-4"}),
        completion("Done."),
    )
    _run(llm, monkeypatch, output)
    tool_msg = next(m for m in llm.convos[1] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["coverage"] == "You have read pages 1-4 of 11. Not yet read: 5-11."


def test_coverage_of_a_long_document_is_a_count():
    read = {}
    agent._with_coverage(json.dumps({"doc_name": "d", "total_pages": 603, "returned_pages": "10-19"}), read)
    view = agent._with_coverage(
        json.dumps({"doc_name": "d", "total_pages": 603, "returned_pages": "200"}), read
    )
    assert json.loads(view)["coverage"] == "You have read pages 10-19, 200 of 603. 592 pages not yet read."


def test_model_calls_have_a_timeout_and_retries(monkeypatch):
    seen = {}
    monkeypatch.setattr(budget, "OpenAI", lambda **kw: seen.update(kw) or object())
    budget.BudgetedLLM(cache=False)
    assert seen["timeout"] == Settings().llm_timeout_seconds == 90.0 and seen["max_retries"] == 2


def _tree(n_nodes, summary_chars):
    return {
        "doc_name": "d",
        "structure": [
            {
                "node_id": f"{i:04d}",
                "title": f"Exhibit {i}",
                "start_index": i,
                "end_index": i,
                "summary": "x" * summary_chars,
            }
            for i in range(n_nodes)
        ],
    }


def _structure_then_answer():
    return Script(tool_call_completion("get_document_structure", {"doc_name": "d"}), completion("Answer."))


def test_chats_run_on_the_chat_model_at_low_effort(monkeypatch):
    llm = _structure_then_answer()
    result = _run(llm, monkeypatch, _tree(20, 200))
    chat = Settings().chat_model
    assert llm.models == [(chat, Settings().chat_effort)] * 2 and result.model == chat


def test_a_model_the_caller_chose_is_kept(monkeypatch):
    llm = _structure_then_answer()
    result = _run(llm, monkeypatch, _tree(20, 200), model="gpt-4.1-nano")
    assert llm.models == [("gpt-4.1-nano", None)] * 2 and result.model == "gpt-4.1-nano"


def test_a_reading_note_inside_a_short_reply_is_not_the_answer():
    assert agent._planning_note("Likely the finding was normal. Need read page 133? The testimony, perhaps.")
    assert agent._planning_note("The exhibit list continues; should check pages 40-42 for it.")
    # real answers that mention pages or needs are kept
    assert not agent._planning_note("The exhibit is listed on page 246 as ETH.MESH.03906527.")
    assert not agent._planning_note(
        "No. The plaintiff needs to show causation, and the court found she had not."
    )

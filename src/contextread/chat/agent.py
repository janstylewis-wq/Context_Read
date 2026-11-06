"""Document chat agent: an OpenAI tool-calling loop over the document tools, every LLM call
through the budget guard. Events (tool calls, final text) are reported through on_event so
callers can stream them."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine

from .. import doctools
from ..config import get_settings
from ..llm import BudgetedLLM

# limits (CHAT_MAX_TURNS, CHAT_STRUCTURE_CHARS, CHAT_READ_CHARS, CHAT_PAGES_PER_READ, CHAT_ANSWER_MAX_TOKENS,
# CHAT_EFFORT) are settings, read when a chat runs
# a reply with no tool call that only says what the model means to do next ("I'll check that page", "Need pages
# 5-9") is not an answer: the loop asks it to go on instead of returning it
PLANNING_NOTE_CHARS = 300
PLANNING_NOTE = re.compile(
    r"^\W*(?:need\b|let me\b|i(?:'|’)ll\b|i will\b|i need\b|i should\b|next\b|now i\b|checking\b|looking\b)"
    r"|\b(?:i(?:'|’)ll|i will|let me|i need to)\s+(?:check|read|look|open|review|see|get|fetch|search)\b"
    # anywhere in a short reply: a note about reading pages, "Need read page 133?", "should check pages 4-6"
    r"|\b(?:need(?:s)?|should|must)\s+(?:to\s+)?(?:read|check|look\s+at|open|see|review)\s+(?:the\s+)?(?:page|pages|p\.)\s*\d"
    r"|\bpages?\s+\d+(?:\s*-\s*\d+)?\s*\?",
    re.I,
)
CONTINUE = "Continue: call the tools to read what you need, then give your answer."
MAX_NUDGES = 2
COVERAGE_LIST_PAGES = 60  # up to this many pages, coverage names the unread pages; above it, a count

PAGE_TOOL_DESCRIPTION = (
    "The text of specific pages as blocks (page, type, block_id, text). At most {n} pages per call."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse_documents",
            "description": "List the user's documents (id, name, description, page count). Call this first when "
            "you do not know which document answers the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_id": {
                        "type": "string",
                        "description": "Limit to one folder; omit for the whole library.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": "The document's tree of sections: node_id, title, page range (start_index-end_index) "
            "and a summary per section. Use it to decide which pages to read.",
            "parameters": {
                "type": "object",
                "properties": {"doc_name": {"type": "string", "description": "Document name or id."}},
                "required": ["doc_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": PAGE_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "description": "Document name or id."},
                    "pages": {"type": "string", "description": "Pages such as '5', '3,8' or '5-7'."},
                },
                "required": ["doc_name", "pages"],
            },
        },
    },
]


def _tools(pages_per_read: int) -> list[dict]:
    tools = json.loads(json.dumps(TOOLS))
    for t in tools:
        if t["function"]["description"] == PAGE_TOOL_DESCRIPTION:
            t["function"]["description"] = PAGE_TOOL_DESCRIPTION.format(n=pages_per_read)
    return tools


SYSTEM = """You answer questions about the user's documents. Find the answer in the documents with the tools:
look at a document's structure, then read the pages that should hold the answer. Base every statement on
text you have read; if the documents do not answer the question, say so. Be concise and give exact figures.
When you need more pages, call the tool: never end a reply by saying what you will read next.
Section summaries are short and leave out most names and facts. Before you answer that a document does not
say something, read the pages you have not read that could hold it (for a short document, all of them)."""

CITATIONS = """After each statement taken from a document, add a citation tag naming the block it came from:
<cite doc="DOCUMENT NAME" page="PAGE" block="BLOCK_ID"/>. Use the exact document name, page and block_id
from get_page_content results. Cite only blocks you actually read."""


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    model: str | None = None  # the model that answered


def _system_prompt(owner, doc_ids: list[str], folder_id: str | None, enable_citations: bool, engine) -> str:
    parts = [SYSTEM]
    if doc_ids:
        names = []
        for ref in doc_ids:
            try:
                names.append(doctools.get_document_structure(owner, ref, engine)["doc_name"])
            except doctools.ToolError:
                continue
        if names:
            parts.append(
                "The user is asking about: "
                + ", ".join(f'"{n}"' for n in names)
                + ". Work with these directly (get_document_structure, then get_page_content); "
                "there is no need to call browse_documents."
            )
    if folder_id and folder_id != "root":
        parts.append(f"Prefer documents in folder {folder_id} (browse_documents with folder_id).")
    if enable_citations:
        parts.append(CITATIONS)
    return "\n\n".join(parts)


def run_tool(owner, name: str, args: dict, engine: Engine | None) -> Any:
    try:
        if name == "browse_documents":
            return doctools.browse_documents(owner, args.get("folder_id"), engine=engine)
        if name == "get_document_structure":
            return doctools.get_document_structure(owner, args["doc_name"], engine=engine)
        if name == "get_page_content":
            return doctools.get_page_content(owner, args["doc_name"], args["pages"], engine=engine)
        return {"error": f"unknown tool {name!r}"}
    except (doctools.ToolError, KeyError) as e:
        return {"error": str(e)}


def _model_view(output: Any) -> str:
    """Tool output as the model sees it. Ids that differ between identical documents are left out,
    so the same question on the same document makes the same request (and hits the cache).
    Nothing is cut mid-text: a structure too large keeps every title and page range with shorter
    summaries, and page content too large drops whole pages from the end with a note naming them."""
    s = get_settings()
    if isinstance(output, dict):
        output = {k: v for k, v in output.items() if k != "doc_id"}
        if isinstance(output.get("structure"), list):
            output["structure"], note = doctools.fit_structure(output["structure"], s.chat_structure_chars)
            if note:
                output["note"] = note
            return json.dumps(output, ensure_ascii=False)[: s.chat_structure_chars + 1000]
        if isinstance(output.get("content"), list):
            output = _fit_pages(output, s.chat_read_chars)
    return json.dumps(output, ensure_ascii=False)[: s.chat_read_chars]


def _fit_pages(output: dict, limit: int) -> dict:
    """Drop whole pages from the end until the result fits, and say which pages to ask for again."""
    content = output["content"]
    pages = list(dict.fromkeys(b["page"] for b in content))
    dropped: list[int] = []
    while (
        len(pages) > 1 and len(json.dumps({**output, "content": content}, ensure_ascii=False)) > limit - 300
    ):
        dropped.insert(0, pages.pop())
        content = [b for b in content if b["page"] in pages]
    if not dropped:
        return output
    shown = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
    later = f"{dropped[0]}-{dropped[-1]}" if len(dropped) > 1 else str(dropped[0])
    note = f"Pages {later} were not shown (too much text in one call); request them separately."
    return {
        **output,
        "content": content,
        "returned_pages": shown,
        "note": f"{output['note']} {note}" if output.get("note") else note,
    }


def _complete(llm, model, convo, purpose, options, stream, emit) -> tuple[str, list[dict], str, Any]:
    """One model turn: (text, tool calls as {id, name, arguments}, finish reason, usage).
    When streaming, text is emitted as it arrives."""
    if not stream:
        resp = llm.chat(
            model, convo, max_tokens=get_settings().chat_answer_max_tokens, purpose=purpose, **options
        )
        choice = resp.choices[0]
        calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or ""}
            for tc in choice.message.tool_calls or []
        ]
        return choice.message.content or "", calls, choice.finish_reason or "stop", resp.usage
    text: list[str] = []
    calls: dict[int, dict] = {}
    finish, usage, open_text = "stop", None, False
    max_tokens = get_settings().chat_answer_max_tokens
    for chunk in llm.chat_stream(model, convo, max_tokens=max_tokens, purpose=purpose, **options):
        usage = chunk.usage or usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            if not open_text:
                emit({"type": "text-start"})
                open_text = True
            emit({"type": "text-delta", "delta": delta.content})
            text.append(delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            call["id"] = tc.id or call["id"]
            if tc.function:
                call["name"] += tc.function.name or ""
                call["arguments"] += tc.function.arguments or ""
        finish = choice.finish_reason or finish
    if open_text:
        emit({"type": "text-end"})
    return "".join(text), [calls[k] for k in sorted(calls)], finish, usage


def run_chat(
    owner: tuple[str, str],
    messages: list[dict],
    llm: BudgetedLLM,
    doc_ids: list[str] | None = None,
    folder_id: str | None = None,
    enable_citations: bool = False,
    temperature: float | None = None,
    model: str | None = None,
    engine: Engine | None = None,
    on_event: Callable[[dict], None] | None = None,
    purpose: str = "chat",
    stream: bool = False,
    model_options: dict | None = None,
) -> ChatResult:
    """Events: step-start per model turn; tool_call and tool_result per tool use; text-start,
    text-delta, text-end while streaming; text with the final answer."""
    emit = on_event or (lambda _event: None)
    convo: list[dict] = [
        {
            "role": "system",
            "content": _system_prompt(owner, doc_ids or [], folder_id, enable_citations, engine),
        }
    ]
    convo += [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in messages
        if m.get("role") in ("user", "assistant", "system")
    ]
    result = ChatResult(text="")
    s = get_settings()
    options = dict(model_options or {})
    if model is None:
        model = s.chat_model
        if s.is_reasoning(model):
            options.setdefault("reasoning_effort", s.chat_effort)
    result.model = model
    max_turns, tool_list = s.chat_max_turns, _tools(doctools.pages_per_read())
    if temperature is not None:
        options["temperature"] = temperature
    read: dict[str, set[int]] = {}
    nudges = 0
    for turn in range(max_turns):
        emit({"type": "step-start"})
        # the last turn offers no tools, so the model has to answer with what it has read
        tools = {} if turn == max_turns - 1 else {"tools": tool_list}
        text, calls, finish, usage = _complete(llm, model, convo, purpose, {**tools, **options}, stream, emit)
        if usage:
            result.prompt_tokens += usage.prompt_tokens
            result.completion_tokens += usage.completion_tokens
        if not calls and turn < max_turns - 2 and nudges < MAX_NUDGES and _planning_note(text):
            nudges += 1
            convo += [{"role": "assistant", "content": text}, {"role": "user", "content": CONTINUE}]
            continue
        if not calls:
            result.text, result.finish_reason = text, finish
            emit({"type": "text", "text": text})
            return result
        convo.append(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            }
        )
        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            emit({"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": args})
            output = run_tool(owner, c["name"], args, engine)
            emit(
                {"type": "tool_result", "id": c["id"], "name": c["name"], "arguments": args, "output": output}
            )
            result.tool_calls.append({"name": c["name"], "arguments": args})
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": _with_coverage(_model_view(output), read),
                }
            )
    result.finish_reason = "length"
    return result


def _planning_note(text: str) -> bool:
    t = (text or "").strip()
    return not t or (len(t) <= PLANNING_NOTE_CHARS and bool(PLANNING_NOTE.search(t)))


def _ranges(pages) -> str:
    out, run = [], []
    for p in sorted(pages):
        if run and p == run[-1] + 1:
            run.append(p)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [p]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def _with_coverage(view: str, read: dict[str, set[int]]) -> str:
    """A page read also tells the model which pages of that document it has read so far, so it does not take
    the outline's summaries for the whole text and conclude something is absent after reading a few pages."""
    try:
        out = json.loads(view)
    except json.JSONDecodeError:
        return view
    total = out.get("total_pages") if isinstance(out, dict) else None
    shown = out.get("returned_pages") if isinstance(out, dict) else None
    if not isinstance(total, int) or not isinstance(shown, str) or not out.get("doc_name"):
        return view
    seen = read.setdefault(out["doc_name"], set())
    for part in shown.split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            seen.update(range(int(lo), int(hi or lo) + 1))
    unread = set(range(1, total + 1)) - seen
    note = f"You have read pages {_ranges(seen)} of {total}."
    if unread:
        note += (
            f" Not yet read: {_ranges(unread)}."
            if total <= COVERAGE_LIST_PAGES
            else f" {len(unread)} pages not yet read."
        )
    out["coverage"] = note
    return json.dumps(out, ensure_ascii=False)

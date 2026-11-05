"""Chat sessions for the web app: messages are stored as AI SDK UI messages, and a turn streams
AI SDK UI-message chunks (the recorded chat/sendMessage protocol). The agent runs in a worker
thread that saves the answer itself, so an answer already paid for is kept even when the browser
goes away mid-stream."""

import base64
import json
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, delete, func, select

from .. import accounts
from ..db import session_scope
from ..ids import new_id
from ..llm import BudgetedLLM
from ..models import ChatMessage, ChatSession, UsageEvent
from .agent import run_chat

log = logging.getLogger(__name__)

TITLE_CHARS = 80
THINKING_MODEL = "gpt-6-luna"


class ChatError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def _not_found() -> ChatError:
    return ChatError(404, "session_not_found", "Chat not found.")


def text_of(parts: list[dict]) -> str:
    return "\n".join(p.get("text", "") for p in parts or [] if p.get("type") == "text").strip()


def doc_ids_of(parts: list[dict]) -> list[str]:
    """Document ids from the JSON document cards the client attaches as file parts."""
    ids = []
    for p in parts or []:
        if p.get("type") != "file" or p.get("mediaType") != "application/json":
            continue
        url = p.get("url") or ""
        if not url.startswith("data:") or "," not in url:
            continue
        try:
            card = json.loads(base64.b64decode(url.split(",", 1)[1]))
        except (ValueError, json.JSONDecodeError):
            continue
        doc_id = card.get("__documentId") or card.get("id")
        if isinstance(doc_id, str):
            ids.append(doc_id)
    return ids


def _message_out(m: ChatMessage) -> dict:
    return {"id": m.id, "role": m.role, "parts": m.parts, "metadata": m.meta or {}}


def _session_out(c: ChatSession) -> dict:
    return {
        "id": c.id,
        "title": c.title,
        "docIds": (c.meta or {}).get("doc_ids", []),
        "createdAt": c.created_at.isoformat(),
        "updatedAt": c.updated_at.isoformat(),
    }


def _owned(s, user_id: str, session_id: str) -> ChatSession:
    c = s.get(ChatSession, session_id)
    if c is None or c.user_id != user_id:
        raise _not_found()
    return c


def list_sessions(user_id: str, limit: int = 50, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        rows = s.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        return {"sessions": [_session_out(c) for c in rows]}


def get_session(user_id: str, session_id: str, limit: int = 100, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        c = _owned(s, user_id, session_id)
        rows = s.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == c.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).all()
        return {**_session_out(c), "messages": [_message_out(m) for m in reversed(rows)]}


def update_session(user_id: str, session_id: str, title: str, engine: Engine | None = None) -> dict:
    title = (title or "").strip()
    if not title or len(title) > 200:
        raise ChatError(400, "invalid_title", "Titles are 1-200 characters.")
    with session_scope(engine) as s:
        c = _owned(s, user_id, session_id)
        c.title = title
        s.flush()
        s.refresh(c)
        return _session_out(c)


def delete_session(user_id: str, session_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        s.delete(_owned(s, user_id, session_id))
    return {"deleted": True}


# ---------- sending a message ----------


def _compact(name: str, output: Any) -> Any:
    """What the browser sees of a tool result: enough for the "Read pages 1-3" disclosure,
    without shipping page text it never displays."""
    if not isinstance(output, dict):
        return output
    if "error" in output:
        return {"error": output["error"]}
    if name == "get_page_content":
        return {
            "doc_name": output.get("doc_name"),
            "total_pages": output.get("total_pages"),
            "returned_pages": output.get("returned_pages"),
            "blocks": len(output.get("content") or []),
        }
    if name == "get_document_structure":
        return {"doc_name": output.get("doc_name"), "page_count": output.get("page_count")}
    if name == "browse_documents":
        return {"count": output.get("count"), "documents": [d["name"] for d in output.get("documents") or []]}
    return output


@dataclass
class ChatTurn:
    """One answer being produced. Created only after the turn passed its checks and the user
    message was saved; stream() runs the agent and yields UI-message chunks."""

    user_id: str
    session_id: str
    user_message_id: str
    assistant_message_id: str
    history: list[dict]
    doc_ids: list[str]
    thinking: bool
    llm: BudgetedLLM
    engine: Engine | None
    parts: list[dict] = field(default_factory=list)

    def _metadata(self) -> dict:
        return {
            "sessionId": self.session_id,
            "messageId": self.assistant_message_id,
            "userMessageId": self.user_message_id,
        }

    def _run(self, out: queue.Queue) -> None:
        texts: dict[str, dict] = {}
        n = 0

        def on_event(ev: dict) -> None:
            nonlocal n
            kind = ev["type"]
            if kind == "step-start":
                self.parts.append({"type": "step-start"})
            elif kind == "text-start":
                n += 1
                texts[str(n)] = part = {"type": "text", "text": "", "state": "done"}
                self.parts.append(part)
                ev = {**ev, "id": str(n)}
            elif kind == "text-delta":
                texts[str(n)]["text"] += ev["delta"]
                ev = {**ev, "id": str(n)}
            elif kind == "text-end":
                ev = {**ev, "id": str(n)}
            elif kind == "tool_result":
                ev = {**ev, "output": _compact(ev["name"], ev["output"])}
                self.parts.append(
                    {
                        "type": f"tool-{ev['name']}",
                        "toolCallId": ev["id"],
                        "state": "output-available",
                        "input": ev["arguments"],
                        "output": ev["output"],
                    }
                )
            out.put(ev)

        model, options = None, None
        if self.thinking:
            model, options = THINKING_MODEL, {"reasoning_effort": "low"}
        try:
            result = run_chat(
                ("user", self.user_id),
                self.history,
                self.llm,
                doc_ids=self.doc_ids,
                enable_citations=True,
                model=model,
                model_options=options,
                engine=self.engine,
                on_event=on_event,
                purpose="chat.web",
                stream=True,
            )
        except Exception as e:
            log.exception("chat turn failed in session %s", self.session_id)
            self._save({"error": f"{type(e).__name__}: {e}"})
            out.put({"type": "error", "errorText": "The answer could not be completed. Try again."})
            return
        meta = {
            "finishReason": result.finish_reason,
            "model": result.model,
            "promptTokens": result.prompt_tokens,
            "completionTokens": result.completion_tokens,
        }
        self._save(meta, count_usage=True)
        out.put({"type": "done", "finishReason": result.finish_reason})

    def _save(self, meta: dict, count_usage: bool = False) -> None:
        with session_scope(self.engine) as s:
            s.add(
                ChatMessage(
                    id=self.assistant_message_id,
                    session_id=self.session_id,
                    role="assistant",
                    parts=self.parts,
                    meta=meta,
                )
            )
            c = s.get(ChatSession, self.session_id)
            if c is not None:
                c.updated_at = func.now()
            if count_usage:
                s.add(
                    UsageEvent(
                        owner_kind="user", owner_id=self.user_id, kind="chat", qty=1, ref=self.session_id
                    )
                )

    def stream(self) -> Iterator[dict]:
        events: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(events,), daemon=True).start()
        meta = self._metadata()
        yield {"type": "start", "messageId": self.assistant_message_id, "messageMetadata": meta}
        step_open = False
        while True:
            ev = events.get()
            kind = ev["type"]
            if kind == "step-start":
                if step_open:
                    yield {"type": "finish-step"}
                yield {"type": "start-step"}
                step_open = True
            elif kind in ("text-start", "text-end"):
                yield {"type": kind, "id": ev["id"]}
            elif kind == "text-delta":
                yield {"type": "text-delta", "id": ev["id"], "delta": ev["delta"]}
            elif kind == "tool_call":
                yield {"type": "tool-input-start", "toolCallId": ev["id"], "toolName": ev["name"]}
                yield {
                    "type": "tool-input-available",
                    "toolCallId": ev["id"],
                    "toolName": ev["name"],
                    "input": ev["arguments"],
                }
            elif kind == "tool_result":
                yield {"type": "tool-output-available", "toolCallId": ev["id"], "output": ev["output"]}
            elif kind == "error":
                yield {"type": "error", "errorText": ev["errorText"]}
                return
            elif kind == "done":
                if step_open:
                    yield {"type": "finish-step"}
                yield {"type": "finish", "finishReason": ev["finishReason"], "messageMetadata": meta}
                return


def start_turn(
    user_id: str,
    message: dict | None,
    llm: BudgetedLLM,
    session_id: str | None = None,
    thinking: bool = False,
    trigger: str = "submit-message",
    message_id: str | None = None,
    engine: Engine | None = None,
) -> ChatTurn:
    """Check the turn, save the user message and return the turn to stream.
    submit-message with the id of an existing message edits it: that message and everything after
    it are replaced. regenerate-message drops the given assistant message (and anything after)
    and answers the last user message again."""
    if accounts.usage(user_id, engine=engine)["chat"]["available"] <= 0:
        raise ChatError(402, "chat_limit", "You have used all the chats in your plan.")
    if trigger not in ("submit-message", "regenerate-message"):
        raise ChatError(400, "invalid_trigger", f"Unknown trigger {trigger!r}.")
    if trigger == "submit-message" and (not message or not text_of(message.get("parts") or [])):
        raise ChatError(400, "empty_message", "Type a question first.")

    with session_scope(engine) as s:
        if session_id:
            c = _owned(s, user_id, session_id)
        else:
            if trigger == "regenerate-message":
                raise _not_found()
            title = text_of(message["parts"])[:TITLE_CHARS]
            c = ChatSession(user_id=user_id, title=title, meta={"doc_ids": []})
            s.add(c)
            s.flush()

        cut_from = message_id or (message or {}).get("id")
        if cut_from:
            anchor = s.get(ChatMessage, cut_from)
            if anchor is not None and anchor.session_id == c.id:
                s.execute(
                    delete(ChatMessage).where(
                        ChatMessage.session_id == c.id, ChatMessage.created_at >= anchor.created_at
                    )
                )

        user_message_id = None
        doc_ids = list((c.meta or {}).get("doc_ids", []))
        if trigger == "submit-message":
            parts = message.get("parts") or []
            user_message_id = message.get("id") or new_id()
            if s.get(ChatMessage, user_message_id) is not None:
                user_message_id = new_id()  # an id from another session; never overwrite it
            s.add(ChatMessage(id=user_message_id, session_id=c.id, role="user", parts=parts))
            doc_ids = list(dict.fromkeys(doc_ids + doc_ids_of(parts)))
            c.meta = {**(c.meta or {}), "doc_ids": doc_ids}
        s.flush()

        rows = s.scalars(
            select(ChatMessage).where(ChatMessage.session_id == c.id).order_by(ChatMessage.created_at)
        ).all()
        if not rows or rows[-1].role != "user":
            raise ChatError(400, "nothing_to_answer", "There is no question to answer.")
        user_message_id = user_message_id or rows[-1].id
        history = [{"role": m.role, "content": text_of(m.parts)} for m in rows if text_of(m.parts)]
        session_id = c.id

    return ChatTurn(
        user_id=user_id,
        session_id=session_id,
        user_message_id=user_message_id,
        assistant_message_id=new_id(),
        history=history,
        doc_ids=doc_ids,
        thinking=thinking,
        llm=llm,
        engine=engine,
    )

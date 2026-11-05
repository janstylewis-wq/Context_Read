import base64
import json

import pytest
from fakes import ScriptedClient
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_indexing import DISNEY, upload

from contextread import auth, config
from contextread.api import deps
from contextread.api.main import app
from contextread.db import session_scope
from contextread.indexing import pipeline
from contextread.llm import BudgetedLLM
from contextread.models import Block, ChatSession, File, UsageEvent

DOC = "q1-fy25-earnings.pdf"
ANSWER = (
    "Revenue was $24.7 billion, up 5%. "
    f'<cite doc="{DOC}" page="1" block="p1_text_3"/> Experiences grew 3%.'
    f'<cite doc="{DOC}" page="1" block="p1_list_6"/>'
)


@pytest.fixture(scope="module")
def indexed(engine, bucket, test_dsn):
    """One user with the Disney release indexed (fake LLM for summaries)."""
    with session_scope(engine) as s:
        s.execute(File.__table__.delete())
        s.execute(ChatSession.__table__.delete())
        s.execute(UsageEvent.__table__.delete())
    uid = auth.ensure_user("chat@web.test", engine=engine)
    stranger = auth.ensure_user("stranger@web.test", engine=engine)
    reg = upload(DISNEY, DOC, owner=uid, engine=engine)
    llm = BudgetedLLM(dsn=test_dsn, cap_usd=1, client=ScriptedClient(DOC, ANSWER), cache=False)
    pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)
    return {"uid": uid, "stranger": stranger, "doc_id": reg["doc_id"]}


@pytest.fixture
def api(indexed, clean_db):
    client = ScriptedClient(DOC, ANSWER)
    app.dependency_overrides[deps.get_llm] = lambda: BudgetedLLM(
        dsn=clean_db, cap_usd=1, client=client, cache=False
    )
    with TestClient(app, headers={"x-internal-token": config.get_settings().internal_api_token}) as c:
        yield c, client
    app.dependency_overrides.clear()


def card_part(doc_id: str) -> dict:
    card = {"name": DOC, "status": "completed", "page_count": 22, "__documentId": doc_id}
    url = "data:application/json;base64," + base64.b64encode(json.dumps(card).encode()).decode()
    return {"type": "file", "mediaType": "application/json", "filename": f"{DOC}.json", "url": url}


def user_message(text: str, doc_id: str | None = None, mid: str = "msg-user-1") -> dict:
    parts = ([card_part(doc_id)] if doc_id else []) + [{"type": "text", "text": text}]
    return {"id": mid, "role": "user", "parts": parts}


def send(c, body: dict) -> list[dict]:
    with c.stream("POST", "/internal/chat/send", json=body) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]


def test_turn_streams_ui_message_chunks_and_is_saved(api, indexed):
    c, llm_client = api
    uid = indexed["uid"]
    chunks = send(c, {"owner_id": uid, "message": user_message("What was revenue?", indexed["doc_id"])})
    types = [ch["type"] for ch in chunks]
    assert types[:6] == [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",
        "finish-step",
    ]
    assert types[6:8] == ["start-step", "text-start"] and types[-3:] == ["text-end", "finish-step", "finish"]
    assert "".join(ch["delta"] for ch in chunks if ch["type"] == "text-delta") == ANSWER
    output = next(ch["output"] for ch in chunks if ch["type"] == "tool-output-available")
    assert output == {"doc_name": DOC, "total_pages": 22, "returned_pages": "1", "blocks": output["blocks"]}
    assert output["blocks"] > 5  # the browser gets a count, not the page text
    meta = chunks[0]["messageMetadata"]
    assert chunks[0]["messageId"] == meta["messageId"] and meta["userMessageId"] == "msg-user-1"
    assert chunks[-1] == {"type": "finish", "finishReason": "stop", "messageMetadata": meta}
    # the selected document reached the system prompt; the answer streamed from a stream call
    system = llm_client.calls[0]["messages"][0]["content"]
    assert f'"{DOC}"' in system and "<cite" in system and llm_client.calls[-1]["stream"] is True

    sid = meta["sessionId"]
    session = c.get(f"/internal/chat/sessions/{sid}", params={"owner_id": uid}).json()
    assert session["title"] == "What was revenue?" and session["docIds"] == [indexed["doc_id"]]
    user, assistant = session["messages"]
    assert user["id"] == "msg-user-1" and assistant["id"] == meta["messageId"]
    assert [p["type"] for p in assistant["parts"]] == [
        "step-start",
        "tool-get_page_content",
        "step-start",
        "text",
    ]
    assert assistant["parts"][-1]["text"] == ANSWER
    assert assistant["metadata"]["finishReason"] == "stop"
    listed = c.get("/internal/chat/sessions", params={"owner_id": uid}).json()["sessions"]
    assert sid in [s["id"] for s in listed]
    assert c.get(f"/internal/users/{uid}/usage").json()["chat"]["total"] >= 1

    # a follow-up in the same session carries the earlier turn and keeps the document
    send(c, {"owner_id": uid, "session_id": sid, "message": user_message("And income?", mid="msg-user-2")})
    history = [m for m in llm_client.calls[-1]["messages"] if m["role"] in ("user", "assistant")]
    assert [m["content"] for m in history][:3] == ["What was revenue?", ANSWER, "And income?"]
    assert f'"{DOC}"' in llm_client.calls[-1]["messages"][0]["content"]


def test_regenerate_and_edit_replace_later_messages(api, indexed):
    c, _ = api
    uid = indexed["uid"]
    first = send(c, {"owner_id": uid, "message": user_message("Q one", indexed["doc_id"], mid="edit-1")})
    sid, aid = first[0]["messageMetadata"]["sessionId"], first[0]["messageId"]

    again = send(c, {"owner_id": uid, "session_id": sid, "trigger": "regenerate-message", "message_id": aid})
    assert again[0]["messageId"] != aid and again[0]["messageMetadata"]["userMessageId"] == "edit-1"
    msgs = c.get(f"/internal/chat/sessions/{sid}", params={"owner_id": uid}).json()["messages"]
    assert [m["id"] for m in msgs] == ["edit-1", again[0]["messageId"]]

    send(c, {"owner_id": uid, "session_id": sid, "message": user_message("Q one, edited", mid="edit-1")})
    msgs = c.get(f"/internal/chat/sessions/{sid}", params={"owner_id": uid}).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["parts"][-1]["text"] == "Q one, edited"


def test_refused_turns_are_plain_http_errors(api, indexed, engine):
    c, llm_client = api
    uid, stranger = indexed["uid"], indexed["stranger"]
    empty = c.post("/internal/chat/send", json={"owner_id": uid, "message": user_message("   ")})
    assert empty.status_code == 400
    first = send(c, {"owner_id": uid, "message": user_message("Mine", indexed["doc_id"], mid="own-1")})
    sid = first[0]["messageMetadata"]["sessionId"]
    calls = len(llm_client.calls)
    theirs = c.post(
        "/internal/chat/send", json={"owner_id": stranger, "session_id": sid, "message": user_message("x")}
    )
    assert theirs.status_code == 404
    assert c.get(f"/internal/chat/sessions/{sid}", params={"owner_id": stranger}).status_code == 404
    assert c.delete(f"/internal/chat/sessions/{sid}", params={"owner_id": stranger}).status_code == 404

    with session_scope(engine) as s:
        s.add(UsageEvent(owner_kind="user", owner_id=stranger, kind="chat", qty=100))
    limited = c.post("/internal/chat/send", json={"owner_id": stranger, "message": user_message("hi")})
    assert limited.status_code == 402 and limited.json()["detail"]["code"] == "chat_limit"
    assert len(llm_client.calls) == calls  # nothing was sent to the model

    renamed = c.patch(f"/internal/chat/sessions/{sid}", json={"owner_id": uid, "title": "Revenue"})
    assert renamed.json()["title"] == "Revenue"
    assert c.delete(f"/internal/chat/sessions/{sid}", params={"owner_id": uid}).json() == {"deleted": True}
    assert c.get(f"/internal/chat/sessions/{sid}", params={"owner_id": uid}).status_code == 404


def test_citations_resolve_to_page_boxes(api, indexed, engine):
    c, _ = api
    uid = indexed["uid"]
    with session_scope(engine) as s:
        ids = s.scalars(
            select(Block.block_id)
            .where(Block.file_id == indexed["doc_id"], Block.page == 3)
            .order_by(Block.ord)
        ).all()[:2]
    body = {"owner_id": uid, "items": [{"doc": DOC, "blockIds": [*ids, "p3_text_999"]}]}
    (result,) = c.post("/internal/citations/resolve", json=body).json()["results"]
    assert result["fileId"] == indexed["doc_id"] and result["missing"] == ["p3_text_999"]
    assert [b["blockId"] for b in result["blocks"]] == ids
    for b in result["blocks"]:
        assert b["page"] == 3 and all(0 <= v <= 1000 for v in b["bbox"]) and b["bbox"][0] < b["bbox"][2]

    page = c.get("/internal/citations/page", params={"owner_id": uid, "doc": DOC, "page": 3}).json()
    assert page["pageCount"] == 22 and page["imageUrl"].startswith("http") and 1.2 < page["aspect"] < 1.4
    assert [b["blockId"] for b in page["blocks"]][:2] == ids
    assert (
        c.get("/internal/citations/page", params={"owner_id": uid, "doc": DOC, "page": 99}).status_code == 404
    )

    # another user's documents do not resolve
    other = {"owner_id": indexed["stranger"], "items": [{"doc": DOC, "blockIds": ids}]}
    assert c.post("/internal/citations/resolve", json=other).json()["results"][0]["missing"] == ids

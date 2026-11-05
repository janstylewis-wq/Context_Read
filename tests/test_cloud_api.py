"""Conformance: PageIndex's own MIT-licensed SDK (pageindex==0.5.0), unmodified except for its
BASE_URL, drives our cloud REST API on a real HTTP server. The LLM is scripted (no spend)."""

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fakes import ScriptedClient
from pageindex import PageIndexClient
from pageindex.errors import PageIndexAPIError
from test_indexing import DISNEY

from contextread import auth, jobs
from contextread.api import deps
from contextread.api.main import app
from contextread.db import session_scope
from contextread.indexing import pipeline
from contextread.llm import BudgetedLLM
from contextread.models import File, Folder, MetadataSchema, Retrieval

ANSWER = 'Revenue was $24.7 billion.<cite doc="q1-fy25-earnings.pdf" page="1" block="p1_list_5"/>'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(test_dsn, engine, bucket):
    scripted = ScriptedClient("q1-fy25-earnings.pdf", ANSWER)
    llm = BudgetedLLM(dsn=test_dsn, cap_usd=1, client=scripted, cache=False)
    app.dependency_overrides[deps.get_db_engine] = lambda: engine
    app.dependency_overrides[deps.get_llm] = lambda: llm
    original = pipeline.BudgetedLLM
    pipeline.BudgetedLLM = lambda: llm
    from contextread import retrieval

    retrieval_original = retrieval.BudgetedLLM
    retrieval.BudgetedLLM = lambda: llm

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    stop_worker = threading.Event()

    def work():
        while not stop_worker.is_set():
            if not jobs.run_one("conformance-worker", engine):
                time.sleep(0.2)

    threading.Thread(target=work, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/healthz").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)
    with session_scope(engine) as s:
        for table in (Retrieval, File, MetadataSchema, Folder):
            s.execute(table.__table__.delete())
    user_id = auth.ensure_user("conformance@example.com", engine=engine)
    secret, key_id = auth.create_api_key(user_id, "cloud", engine=engine)
    yield {"base": base, "key": secret, "key_id": key_id, "user_id": user_id, "llm_client": scripted}
    stop_worker.set()
    srv.should_exit = True
    app.dependency_overrides.clear()
    pipeline.BudgetedLLM = original
    retrieval.BudgetedLLM = retrieval_original


@pytest.fixture(scope="module")
def client(server):
    c = PageIndexClient(api_key=server["key"])
    c.BASE_URL = server["base"]  # the only change: point the SDK at our server
    return c


@pytest.fixture(scope="module")
def doc_id(client):
    result = client.submit_document(str(DISNEY), wait=True)
    return result["doc_id"]


def test_rejects_missing_and_wrong_keys(server):
    r = httpx.get(f"{server['base']}/docs/")
    assert r.status_code == 401 and r.json()["detail"]["code"] == "invalid_api_key"
    bad = PageIndexClient(api_key="cr_live_not-a-real-key")
    bad.BASE_URL = server["base"]
    with pytest.raises(PageIndexAPIError) as e:
        bad.list_documents()
    assert e.value.status_code == 401


def test_submit_waits_and_document_metadata(client, doc_id):
    doc = client.get_document(doc_id)
    assert doc["status"] == "completed" and doc["pageNum"] == 22 and doc["name"] == "q1-fy25-earnings.pdf"
    assert set(doc) >= {"id", "name", "description", "status", "createdAt", "pageNum", "folderId", "metadata"}
    assert client.is_retrieval_ready(doc_id)


def test_tree(client, doc_id):
    tree = client.get_tree(doc_id, node_summary=True)
    assert tree["status"] == "completed" and tree["retrieval_ready"] is True
    root = tree["result"][0]
    assert {"title", "node_id", "start_index", "end_index", "summary", "text", "nodes"} <= set(root)
    bare = client.get_tree(doc_id, node_summary=False, include_text=False)["result"][0]
    assert "summary" not in bare and "text" not in bare and bare["nodes"]


def test_ocr_and_page_content(client, doc_id):
    pages = client.get_ocr(doc_id, format="page")["result"]
    assert [p["page_index"] for p in pages] == list(range(1, 23))
    assert "Revenues increased 5%" in pages[0]["markdown"]
    nodes = client.get_ocr(doc_id, format="node")["result"]
    assert {"title", "level", "page_index", "text"} <= set(nodes[0])
    raw = client.get_ocr(doc_id, format="raw")["result"]
    assert isinstance(raw, str) and "SUMMARIZED FINANCIAL RESULTS" in raw
    picked = client.get_page_content(doc_id, "3,5-6")
    assert [p["page_index"] for p in picked] == [3, 5, 6]


def test_block_and_images(client, doc_id):
    block = client.get_block(doc_id, "p3_table_7")
    assert block["page"] == 3 and block["block_type"] == "table" and "Experiences" in block["content"]
    assert all(0 <= v <= 1000 for v in block["bbox"]) and block["bbox"][0] < block["bbox"][2]
    with pytest.raises(PageIndexAPIError) as e:
        client.get_block(doc_id, "p3_text_999")
    assert e.value.status_code == 404
    url = client.get_page_image(doc_id, 2)
    assert httpx.get(url).content[:4] == b"\x89PNG"


def test_list_and_unique_names(client, doc_id):
    listing = client.list_documents()
    assert listing["total"] >= 1 and set(listing) == {"documents", "total", "limit", "offset"}
    assert listing["documents"][0]["id"] and "features" in listing["documents"][0]
    by_name = client._api.list_documents(name="q1-fy25-earnings.pdf")  # the SDK's CloudAPI method
    assert [d["id"] for d in by_name["documents"]] == [doc_id]
    again = client.submit_document(str(DISNEY))  # same name: stored as name_1, indexed from the duplicate
    assert again["name"] == "q1-fy25-earnings_1.pdf"
    client.delete_document(again["doc_id"])
    with pytest.raises(PageIndexAPIError):
        client.get_document(again["doc_id"])


def test_retrieval(client, doc_id):
    submitted = client.submit_query(doc_id, "What were total revenues?")
    rid = submitted["retrieval_id"]
    for _ in range(100):
        result = client.get_retrieval(rid)
        if result["status"] == "completed":
            break
        time.sleep(0.1)
    assert result["status"] == "completed" and result["doc_id"] == doc_id
    node = result["retrieved_nodes"][0]
    assert node["node_id"] == "0001" and node["relevant_contents"][0]["page_index"] == 1
    assert "Revenues increased 5%" in node["relevant_contents"][0]["relevant_content"]


def test_chat_completions_with_citations(client, doc_id, server):
    reply = client.chat("What was Q1 revenue?", doc_id=doc_id)
    text = reply if isinstance(reply, str) else getattr(reply, "text", None) or str(reply)
    assert "$24.7 billion" in text
    citations = client.get_citations(ANSWER, doc_id=doc_id)
    assert citations and citations[0]["page"] == 1 and citations[0]["block_id"] == "p1_list_5"
    assert citations[0]["doc_id"] == doc_id
    calls = server["llm_client"].calls
    assert any("tools" in c for c in calls), "the agent must offer document tools"


def test_chat_streaming_with_tool_events(server, doc_id):
    from pageindex.cloud_api import CloudAPI

    class _Owner:
        BASE_URL, api_key = server["base"], server["key"]

    api = CloudAPI(_Owner())
    chunks = list(
        api.chat_completions(
            [{"role": "user", "content": "Revenue?"}],
            stream=True,
            stream_metadata=True,
            doc_id=doc_id,
            enable_citations=True,
        )
    )
    kinds = [(c.get("block_metadata") or {}).get("type") for c in chunks]
    assert kinds[:3] == ["mcp_tool_use_start", "tool_use", "tool_use_stop"]
    text = "".join(
        (c["choices"][0]["delta"].get("content") or "") for c in chunks if kinds[chunks.index(c)] == "text"
    )
    assert text == ANSWER
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # the raw text stream carries tool-argument JSON in delta.content, as the reference server's does;
    # the SDK's chat(stream=True) separates the two using the block_metadata tags
    raw = "".join(api.chat_completions([{"role": "user", "content": "Revenue?"}], stream=True, doc_id=doc_id))
    assert raw.endswith(ANSWER) and '"pages": "1"' in raw


def test_sdk_chat_stream_shows_answer_and_tool_calls(client, doc_id):
    answer_only = "".join(client.chat("Revenue?", doc_id=doc_id, stream=True, show_process=False))
    assert answer_only == ANSWER
    with_process = "".join(client.chat("Revenue?", doc_id=doc_id, stream=True))
    assert "[tool_call] get_page_content" in with_process and with_process.rstrip().endswith(ANSWER)


def test_folders_and_metadata_schemas(client):
    root_folder = client.create_folder("Annual reports", description="10-Ks")["folder"]
    assert root_folder["name"] == "Annual reports" and root_folder["parent_folder_id"] is None
    child = client.create_folder("2024", parent_folder_id=root_folder["id"])["folder"]
    assert child["parent_folder_id"] == root_folder["id"]
    assert {f["id"] for f in client.list_folders(parent_folder_id="root")["folders"]} == {root_folder["id"]}
    assert client.list_folders()["total"] == 2

    fields = {"year": {"type": "number", "description": "Fiscal year"}, "ticker": {"type": "string"}}
    created = client.create_metadata_schema(root_folder["id"], fields)
    assert created == {"folder_id": root_folder["id"], "schema": {"fields": fields}}
    with pytest.raises(PageIndexAPIError) as e:
        client.create_metadata_schema(root_folder["id"], fields)
    assert e.value.status_code == 409
    inherited = client.get_metadata_schema(child["id"])
    assert inherited["workspace_id"] == root_folder["id"] and inherited["inherited"] is True

    doc = client.submit_document(str(DISNEY), folder_id=child["id"], metadata={"year": 2025, "ticker": "DIS"})
    assert client.get_document(doc["doc_id"])["metadata"] == {"year": 2025, "ticker": "DIS"}
    assert client.list_documents(folder_id=root_folder["id"], recursive=True)["total"] == 1
    with pytest.raises(PageIndexAPIError) as e:
        client.submit_document(str(DISNEY), folder_id=child["id"], metadata={"year": "twenty"})
    assert e.value.status_code == 400

    assert client.delete_metadata_schema(root_folder["id"]) == {
        "folder_id": root_folder["id"],
        "deleted": True,
    }
    with pytest.raises(PageIndexAPIError) as e:
        client.get_metadata_schema(child["id"])
    assert e.value.status_code == 404


def test_revoked_key_stops_working(server):
    auth.revoke_api_key(server["key_id"])
    r = httpx.get(f"{server['base']}/docs/", headers={"api_key": server["key"]})
    assert r.status_code == 401

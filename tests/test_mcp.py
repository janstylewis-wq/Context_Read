"""The MCP server through the official MCP client (streamable HTTP), against a real uvicorn server."""

import base64
import json
import socket
import threading
import time

import anyio
import httpx
import httpx2
import pytest
import uvicorn
from fakes import ScriptedClient
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from test_indexing import DISNEY, upload

from contextread import auth, folders
from contextread.api.main import app
from contextread.db import session_scope
from contextread.indexing import pipeline
from contextread.llm import BudgetedLLM
from contextread.mcp.server import CONTRACT, EXTRA_TOOLS
from contextread.models import File

DOC = "q1-fy25-earnings.pdf"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_server(engine, bucket, test_dsn):
    with session_scope(engine) as s:
        s.execute(File.__table__.delete())
    uid = auth.ensure_user("mcp@contextread.test", engine=engine)
    other = auth.ensure_user("mcp-other@contextread.test", engine=engine)
    key, _ = auth.create_api_key(uid, "cloud", engine=engine)
    other_key, _ = auth.create_api_key(other, "cloud", engine=engine)
    reports = folders.create_folder(("user", uid), "Reports", engine=engine)["folder"]
    llm = BudgetedLLM(dsn=test_dsn, cap_usd=1, client=ScriptedClient(DOC, "ok"), cache=False)
    in_folder = upload(DISNEY, DOC, owner=uid, engine=engine)
    at_root = upload(DISNEY, "copy at root.pdf", owner=uid, engine=engine)
    with session_scope(engine) as s:
        s.get(File, in_folder["doc_id"]).folder_id = reports["id"]
    for reg in (in_folder, at_root):
        pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/healthz").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)
    yield {"url": f"{base}/mcp", "key": key, "other_key": other_key, "folder": reports["id"], "uid": uid}
    srv.should_exit = True


def run(url: str, key: str, steps):
    """Open an MCP session with the API key and run `steps(client)`."""

    async def main():
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=60)
        async with http, Client(streamable_http_client(url, http_client=http)) as client:
            return await steps(client)

    return anyio.run(main)


def payload(result) -> dict:
    assert result.content and result.content[0].type == "text"
    return json.loads(result.content[0].text)


def test_tools_list_matches_the_reference_contract(mcp_server):
    async def steps(client):
        return (await client.list_tools()).tools

    tools = {t.name: t for t in run(mcp_server["url"], mcp_server["key"], steps)}
    expected = {**CONTRACT["tools"], **EXTRA_TOOLS}
    assert set(tools) == set(expected)
    for name, spec in expected.items():
        served = tools[name].model_dump(by_alias=True, exclude_none=True)
        assert served["description"] == spec["description"], name
        assert served["inputSchema"] == spec["schema"], name
        assert served["annotations"] == spec["annotations"], name


def test_requests_without_a_valid_key_are_refused(mcp_server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"accept": "application/json, text/event-stream"}
    missing = httpx.post(mcp_server["url"], json=body, headers=headers)
    assert missing.status_code == 401 and "Bearer" in missing.headers["www-authenticate"]
    wrong = httpx.post(
        mcp_server["url"], json=body, headers={**headers, "authorization": "Bearer cr_live_nope"}
    )
    assert wrong.status_code == 401


def test_browse_read_and_page_through_documents(mcp_server):
    folder = mcp_server["folder"]

    async def steps(client):
        root = payload(await client.call_tool("browse_documents", {}))
        inside = payload(await client.call_tool("browse_documents", {"folder_id": folder}))
        flat = payload(await client.call_tool("browse_documents", {"recursive": True, "limit": 1}))
        ranked = payload(
            await client.call_tool(
                "browse_documents", {"sort": "relevance", "query": "Disney Experiences revenue"}
            )
        )
        no_query = await client.call_tool("browse_documents", {"sort": "relevance"})
        doc = payload(await client.call_tool("get_document", {"doc_name": DOC}))
        outline = payload(await client.call_tool("get_document_structure", {"doc_name": DOC}))
        pages = payload(await client.call_tool("get_page_content", {"doc_name": DOC, "pages": "1-2"}))
        image = await client.call_tool(
            "get_document_image", {"image_path": pages["page_images"][0]["image_path"]}
        )
        return root, inside, flat, ranked, no_query, doc, outline, pages, image

    root, inside, flat, ranked, no_query, doc, outline, pages, image = run(
        mcp_server["url"], mcp_server["key"], steps
    )

    assert root["folders"] == [{"id": folder, "name": "Reports", "description": None, "document_count": 1}]
    assert [d["name"] for d in root["documents"]] == ["copy at root.pdf"]
    assert [d["name"] for d in inside["documents"]] == [DOC] and inside["documents"][0]["folder_id"] == folder
    assert (
        flat["total"] == 2
        and len(flat["documents"]) == 1
        and flat["next_offset"] == 1
        and flat["folders"] == []
    )
    assert {d["name"] for d in ranked["documents"]} == {DOC, "copy at root.pdf"}
    assert no_query.is_error and payload(no_query)["error"] == "query_required"

    assert doc["status"] == "completed" and doc["page_count"] == 22 and doc["folder_id"] == folder
    assert (
        doc["next_steps"]["options"][2]
        == f'Start with first few pages: get_page_content(doc_name: "{DOC}", pages: "1-3")'
    )

    assert outline["pagination"] == {"part": 1, "total_parts": 1, "has_more": False}
    assert outline["structure"][0]["nodes"] and "node_id" in outline["structure"][0]

    assert pages["returned_pages"] == "1-2" and pages["total_pages"] == 22
    first = pages["content"][0]
    assert set(first) == {"page", "text", "type", "block_id"} and first["block_id"].startswith("p1_")
    assert [p["page"] for p in pages["page_images"]] == [1, 2]

    assert not image.is_error and image.content[0].type == "image"
    assert base64.b64decode(image.content[0].data)[:8] == b"\x89PNG\r\n\x1a\n"
    assert image.content[1].text == f"Page 1 of {DOC}"


def test_errors_and_isolation_between_users(mcp_server):
    async def own(client):
        missing = await client.call_tool("get_document", {"doc_name": "nope.pdf"})
        bad_pages = await client.call_tool("get_page_content", {"doc_name": DOC, "pages": "90-91"})
        bad_image = await client.call_tool("get_document_image", {"image_path": "../../etc/passwd"})
        return missing, bad_pages, bad_image

    missing, bad_pages, bad_image = run(mcp_server["url"], mcp_server["key"], own)
    assert missing.is_error and payload(missing)["error"] == "document_not_found"
    assert bad_pages.is_error and payload(bad_pages)["error"] == "invalid_pages"
    assert bad_image.is_error and payload(bad_image)["error"] == "invalid_image_path"

    # a key from another account sees none of these documents
    async def other(client):
        listing = payload(await client.call_tool("browse_documents", {"recursive": True}))
        doc = await client.call_tool("get_document", {"doc_name": DOC})
        return listing, doc

    listing, doc = run(mcp_server["url"], mcp_server["other_key"], other)
    assert listing["total"] == 0 and doc.is_error


def test_remove_document(mcp_server, engine):
    async def steps(client):
        removed = payload(
            await client.call_tool("remove_document", {"doc_names": ["copy at root.pdf", "missing.pdf"]})
        )
        after = payload(await client.call_tool("browse_documents", {}))
        return removed, after

    removed, after = run(mcp_server["url"], mcp_server["key"], steps)
    assert removed["results"] == [
        {"doc_name": "copy at root.pdf", "status": "deleted"},
        {"doc_name": "missing.pdf", "status": "not_found"},
    ]
    assert after["documents"] == []

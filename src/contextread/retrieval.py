"""Tree-search retrieval (REST /retrieval/): the LLM reads the document outline and picks the
nodes that answer the query; the result carries those nodes' page text."""

import json
import logging
from typing import Any

from sqlalchemy import Engine, select

from . import doctools
from .db import session_scope
from .jobs import handler
from .llm import BudgetedLLM
from .models import File as FileRow
from .models import Page, Retrieval

log = logging.getLogger(__name__)

MAX_NODES = 5
MAX_PAGES_PER_NODE = 4
MAX_OUTLINE_CHARS = 400_000  # a long document's whole outline; larger ones shorten summaries, never titles

PROMPT = """You are given the outline of a document and a question. Choose the sections whose pages most
likely contain the answer (at most {max_nodes}, most relevant first).{thinking}
Return JSON: {{"node_ids": ["0003", ...]{thinking_key}}}

Question: {query}

Outline (node_id | pages | title | summary):
{outline}"""


def _outline(nodes: list[dict], depth: int = 0) -> list[str]:
    lines = []
    for n in nodes:
        summary = (n.get("summary") or "")[:240]
        lines.append(
            f"{'  ' * depth}{n['node_id']} | p{n['start_index']}-{n['end_index']} | {n['title']} | {summary}"
        )
        lines += _outline(n.get("nodes") or [], depth + 1)
    return lines


def _index(nodes: list[dict], out: dict | None = None) -> dict:
    out = {} if out is None else out
    for n in nodes:
        out[n["node_id"]] = n
        _index(n.get("nodes") or [], out)
    return out


def retrieve(
    file: FileRow, query: str, thinking: bool, llm: BudgetedLLM, engine: Engine | None = None
) -> list[dict]:
    structure = file.structure or []
    prompt = PROMPT.format(
        max_nodes=MAX_NODES + (3 if thinking else 0),
        thinking=" First reason step by step about where the answer would be." if thinking else "",
        thinking_key=', "thinking": "your reasoning"' if thinking else "",
        query=query,
        outline="\n".join(
            _outline(doctools.fit_structure(doctools.outline(structure), MAX_OUTLINE_CHARS)[0])
        )[:MAX_OUTLINE_CHARS],
    )
    resp = llm.chat(
        None,
        [{"role": "user", "content": prompt}],
        max_tokens=900 if thinking else 300,
        purpose="retrieval.tree_search",
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        chosen = json.loads(resp.choices[0].message.content or "{}").get("node_ids") or []
    except json.JSONDecodeError:
        chosen = []
    nodes = _index(structure)
    picked = [nodes[i] for i in dict.fromkeys(str(c) for c in chosen) if i in nodes][
        : MAX_NODES + (3 if thinking else 0)
    ]
    with session_scope(engine) as s:
        out = []
        for n in picked:
            pages = list(
                range(n["start_index"], min(n["end_index"], n["start_index"] + MAX_PAGES_PER_NODE - 1) + 1)
            )
            rows = s.scalars(
                select(Page).where(Page.file_id == file.id, Page.page.in_(pages)).order_by(Page.page)
            )
            out.append(
                {
                    "node_id": n["node_id"],
                    "title": n["title"],
                    "start_index": n["start_index"],
                    "end_index": n["end_index"],
                    "relevant_contents": [
                        {"page_index": p.page, "relevant_content": p.text or ""} for p in rows
                    ],
                }
            )
    return out


def run_retrieval(retrieval_id: str, llm: BudgetedLLM | None = None, engine: Engine | None = None) -> None:
    with session_scope(engine) as s:
        r = s.get(Retrieval, retrieval_id)
        r.status = "processing"
        file = s.get(FileRow, r.file_id)
        query, thinking = r.query, r.thinking
        s.expunge(file)
    try:
        if file.status != "completed":
            raise RuntimeError(f"document is {file.status}, not completed")
        result = retrieve(file, query, thinking, llm or BudgetedLLM(), engine)
    except Exception as e:
        with session_scope(engine) as s:
            r = s.get(Retrieval, retrieval_id)
            r.status, r.error_message = "failed", f"{type(e).__name__}: {e}"[:1000]
        raise
    with session_scope(engine) as s:
        r = s.get(Retrieval, retrieval_id)
        r.status, r.result = "completed", result


def retrieval_out(r: Retrieval) -> dict[str, Any]:
    out = {
        "retrieval_id": r.id,
        "doc_id": r.file_id,
        "status": r.status,
        "query": r.query,
        "retrieved_nodes": r.result if r.status == "completed" else None,
    }
    if r.error_message:
        out["error"] = r.error_message
    return out


@handler("retrieval")
def _retrieval_job(payload: dict[str, Any]) -> None:
    run_retrieval(payload["retrieval_id"])

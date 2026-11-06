"""Measure answers on your own documents: write questions whose answer page is known, put them to the chat
agent as a user would, and judge each answer against the expected one.

A questions file is a JSON list of {"doc_id" or "doc_name", "question", "answer", "page" (optional)}. `make`
writes one from pages of indexed documents; questions written by hand work the same way, and a "page" lets
the report tell a missed page (the tree led elsewhere) from a misread one (the page was read, the answer is
still wrong). Every model call goes through the budget guard."""

import contextlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from . import doctools
from .chat.agent import run_chat
from .config import get_settings
from .db import session_scope
from .llm import BudgetedLLM
from .models import File, Page

MIN_PAGE_CHARS = 800  # a page with less text rarely states a fact worth asking about
QUESTION_PAGE_CHARS = 7_000

QUESTION_PROMPT = """Below is page {page} of the document "{name}" ({pages} pages).

{text}

Write one question a person working with this document might ask, whose answer is stated on THIS page: a
specific fact such as a figure, a date, a name, a defined term, a decision, or a reason given. The question must
make sense without seeing the page (never mention page numbers or "this page") and must not be answerable from
the document's title alone. Give the short correct answer too.
Reply with JSON only: {{"question": "...", "answer": "..."}}"""

JUDGE_PROMPT = """Question: {question}
Expected answer: {expected}
Given answer: {given}

Does the given answer state the expected answer (same fact; wording may differ; extra detail is fine)? Reply with
JSON only: {{"correct": true|false}}"""


def _ask_json(llm: BudgetedLLM, model: str, prompt: str, purpose: str) -> dict:
    options = {"reasoning_effort": "low"} if get_settings().is_reasoning(model) else {"temperature": 0}
    resp = llm.chat(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=1000,
        purpose=purpose,
        response_format={"type": "json_object"},
        **options,
    )
    out = json.loads(resp.choices[0].message.content or "{}")
    return out if isinstance(out, dict) else {}


# ---------- make ----------


def _document(owner: tuple[str, str], ref: str) -> tuple[str, str, dict[int, str]]:
    """(id, name, {page: text}) of a completed document."""
    with session_scope() as s:
        f = doctools.resolve(s, owner, ref)
        if f.status != "completed":
            raise SystemExit(f"{f.file_name} is {f.status}, not indexed yet")
        texts = {p: t or "" for p, t in s.execute(select(Page.page, Page.text).where(Page.file_id == f.id))}
        return f.id, f.file_name, texts


def completed_documents(owner: tuple[str, str]) -> list[str]:
    with session_scope() as s:
        return list(
            s.scalars(
                select(File.id)
                .where(File.owner_kind == owner[0], File.owner_id == owner[1], File.status == "completed")
                .order_by(File.created_at)
            )
        )


def make_questions(
    owner: tuple[str, str],
    docs: list[str],
    llm: BudgetedLLM,
    per_doc: int = 5,
    model: str | None = None,
    seed: int = 0,
    workers: int = 4,
) -> list[dict]:
    """Up to `per_doc` questions per document, each from one randomly chosen page with real text."""
    model = model or get_settings().chat_model
    rng = random.Random(seed)
    jobs = []
    for ref in docs:
        doc_id, name, texts = _document(owner, ref)
        good = sorted(p for p, t in texts.items() if len(t) >= MIN_PAGE_CHARS)
        for page in sorted(rng.sample(good, min(per_doc, len(good)))):
            jobs.append((doc_id, name, page, len(texts), texts[page]))

    def one(job) -> dict | None:
        doc_id, name, page, pages, text = job
        prompt = QUESTION_PROMPT.format(page=page, name=name, pages=pages, text=text[:QUESTION_PAGE_CHARS])
        try:
            q = _ask_json(llm, model, prompt, "eval.question")
        except Exception as e:  # noqa: BLE001 - one failed page leaves the others
            print(f"  {name} p{page}: no question ({type(e).__name__}: {e})")
            return None
        if not (isinstance(q.get("question"), str) and isinstance(q.get("answer"), str)):
            return None
        return {
            "doc_id": doc_id,
            "doc_name": name,
            "page": page,
            "question": q["question"],
            "answer": q["answer"],
        }

    with ThreadPoolExecutor(workers) as pool:
        return [q for q in pool.map(one, jobs) if q]


# ---------- run ----------


def pages_read(tool_calls: list[dict], page_limit: int) -> list[int]:
    """Pages the agent asked to read (at most page_limit per call, as a read returns)."""
    read: set[int] = set()
    for c in tool_calls:
        if c["name"] != "get_page_content":
            continue
        with contextlib.suppress(Exception):  # a malformed page spec read nothing
            read |= set(doctools.parse_pages(c["arguments"].get("pages", ""), 100_000)[:page_limit])
    return sorted(read)


def outcome(r: dict) -> str:
    if r.get("error"):
        return "error"
    if r["right"]:
        return "right"
    if r.get("page") is None:
        return "wrong"
    return "misread" if r["page"] in r["pages_read"] else "missed page"


def run_questions(
    owner: tuple[str, str],
    questions: list[dict],
    llm: BudgetedLLM,
    judge_model: str | None = None,
    workers: int = 4,
) -> list[dict]:
    """Each question asked on its own document in a new chat, the answer judged against the expected one."""
    judge_model = judge_model or get_settings().chat_model
    limit = doctools.pages_per_read()
    with session_scope() as s:
        ids = {}
        for q in questions:
            ref = q.get("doc_id") or q.get("doc_name")
            if ref not in ids:
                ids[ref] = doctools.resolve(s, owner, ref).id

    def one(q: dict) -> dict:
        started = time.monotonic()
        base = {k: q.get(k) for k in ("doc_id", "doc_name", "page", "question", "answer")}
        base["doc_id"] = ids[q.get("doc_id") or q.get("doc_name")]
        try:
            res = run_chat(
                owner,
                [{"role": "user", "content": q["question"]}],
                llm,
                doc_ids=[base["doc_id"]],
                enable_citations=False,
                purpose="eval.chat",
            )
            seconds = time.monotonic() - started
            verdict = _ask_json(
                llm,
                judge_model,
                JUDGE_PROMPT.format(question=q["question"], expected=q["answer"], given=res.text or ""),
                "eval.judge",
            )
        except Exception as e:  # noqa: BLE001 - recorded, the run goes on
            return {**base, "right": False, "error": f"{type(e).__name__}: {e}", "pages_read": [], "steps": 0}
        return {
            **base,
            "answer_given": res.text,
            "right": bool(verdict.get("correct")),
            "pages_read": pages_read(res.tool_calls, limit),
            "steps": len(res.tool_calls),
            "seconds": round(seconds, 1),
            "model": res.model,
        }

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, questions))
    for r in results:
        r["outcome"] = outcome(r)
    return results


# ---------- report ----------


def report(results: list[dict], spent: float | None = None) -> str:
    n = len(results) or 1
    count = lambda o: sum(r["outcome"] == o for r in results)  # noqa: E731
    with_page = [r for r in results if r.get("page") is not None]
    timed = sorted(r["seconds"] for r in results if r.get("seconds") is not None)
    lines = [
        f"{len(results)} questions: right {count('right')}/{len(results)} ({100 * count('right') / n:.0f}%)",
    ]
    if with_page:
        hit = sum(r["page"] in r["pages_read"] for r in with_page)
        lines.append(f"answer page read: {hit}/{len(with_page)}")
    lines.append(
        f"steps per question: {sum(r['steps'] for r in results) / n:.1f}"
        + (f"; median time {timed[len(timed) // 2]:.1f}s" if timed else "")
        + (f"; spent ${spent:.3f}" if spent is not None else "")
    )
    wrong = [r for r in results if r["outcome"] != "right"]
    if wrong:
        lines.append("")
        lines.append(
            "Not right (missed page: the answer page was never opened, so look at the tree; misread: it was "
            "read and the answer is still wrong; check the expected answer too, a generated one can be off):"
        )
        for r in wrong:
            where = f"p{r['page']}" if r.get("page") is not None else "-"
            read = ",".join(map(str, r["pages_read"])) or "none"
            lines.append(f"  [{r['outcome']}] {r['doc_name']} {where}, read {read}: {r['question']}")
            if r.get("error"):
                lines.append(f"      error: {r['error']}")
            else:
                lines.append(f"      expected: {r['answer']}")
                lines.append(f"      given:    {(r.get('answer_given') or '').strip()[:300]}")
    return "\n".join(lines)


def compare(before: list[dict], after: list[dict]) -> str:
    """Questions whose result changed between two runs of the same questions file."""
    key = lambda r: (r.get("doc_name"), r["question"])  # noqa: E731
    old = {key(r): r for r in before}
    both = [(old[key(r)], r) for r in after if key(r) in old]
    right = lambda rs: sum(r["right"] for r in rs)  # noqa: E731
    lines = [
        f"{len(both)} questions in both runs: right {right(a for a, _ in both)} -> {right(b for _, b in both)}"
    ]
    for a, b in both:
        if a["right"] != b["right"]:
            lines.append(f"  {'fixed' if b['right'] else 'broke'}: {b['doc_name']}: {b['question']}")
    if len(lines) == 1:
        lines.append("  no question changed")
    return "\n".join(lines)

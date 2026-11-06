"""The `contextread` command.

contextread init               write .env and docker-compose.yml into this folder
contextread migrate            create or upgrade the database tables
contextread index FILE.pdf     index a PDF (billed, budget-guarded) and print its tree
contextread reindex DOC        index a document again under the current settings (same id)
contextread list               the indexed documents and their ids
contextread ask "Q" --doc ID   answer a question from one or more documents
contextread serve / worker     run the API server / the indexing worker
contextread create-key         create a user and an API key for the API and MCP server
contextread eval make/run/compare   measure answers on your own documents
contextread check              is this setup ready (settings, database, storage, models)
contextread show DOC           a document's tree (and --pages text) as the index holds it
"""

import argparse
import json
import logging
import secrets
import time
from importlib import resources
from pathlib import Path

import httpx

from . import auth, files, storage
from .indexing import pipeline, tree
from .llm import BudgetedLLM

# (setting, flag, what it does): the model names init asks for, with the package defaults offered
MODEL_SETTINGS = [
    (
        "CHAT_MODEL",
        "chat_model",
        "answers questions (the strongest model you can use; a reasoning model did best)",
    ),
    ("LLM_DEFAULT_MODEL", "summary_model", "writes section summaries (a small, cheap model)"),
    ("OCR_VLM_MODEL", "vision_model", "reads scanned pages (must accept images; off = Tesseract only)"),
    ("TREE_EXPAND_MODEL", "tree_model", "finds headings, reads tables of contents (off = layout only)"),
    (
        "REASONING_MODELS",
        "reasoning_models",
        "name prefixes of reasoning models, comma-separated (e.g. o3,o4,gpt-5)",
    ),
]


def _free_port(preferred: int) -> int:
    """preferred when nothing listens on it, otherwise the next free port above it."""
    import socket

    for port in range(preferred, preferred + 200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def _model_choices(args) -> dict[str, str]:
    """The model names to write: given as flags, asked for at a terminal (Enter keeps the default), or the
    defaults when neither (a script)."""
    import sys

    from .config import Settings

    fields = {
        "CHAT_MODEL": "chat_model",
        "LLM_DEFAULT_MODEL": "llm_default_model",
        "OCR_VLM_MODEL": "ocr_vlm_model",
        "TREE_EXPAND_MODEL": "tree_expand_model",
        "REASONING_MODELS": "reasoning_models",
    }
    defaults = {k: Settings.model_fields[f].default for k, f in fields.items()}
    ask = sys.stdin.isatty() and not args.yes
    if ask:
        print("Model names, as your provider calls them (Enter keeps the default shown):")
    chosen = {}
    for key, flag, what in MODEL_SETTINGS:
        value = getattr(args, flag)
        if value is None and ask:
            value = input(f"  {key}: {what}\n    [{defaults[key]}] ").strip()
        chosen[key] = value or defaults[key]
    return chosen


def _init(args) -> None:
    import re

    here = Path(args.dir)
    templates = resources.files("contextread") / "templates"
    targets = {here / ".env": "env.template", here / "docker-compose.yml": "docker-compose.yml"}
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"already exists: {', '.join(existing)} (use --force to overwrite)")
    env = (templates / "env.template").read_text()
    for placeholder in ("__PG_PASSWORD__", "__S3_SECRET__", "__INTERNAL_TOKEN__"):
        env = env.replace(placeholder, secrets.token_urlsafe(24))
    for key, value in _model_choices(args).items():  # the commented default line becomes the chosen value
        env = re.sub(rf"^# {key}=.*$", f"{key}={value}", env, count=1, flags=re.M)
    ports = {"PG_PORT": _free_port(5434), "S3_PORT": _free_port(9100)}
    for key, port in ports.items():
        env = re.sub(rf"^{key}=.*$", f"{key}={port}", env, count=1, flags=re.M)
    here.mkdir(parents=True, exist_ok=True)
    (here / ".env").write_text(env)
    (here / ".env").chmod(0o600)
    (here / "docker-compose.yml").write_text((templates / "docker-compose.yml").read_text())
    print(
        f"wrote {here / '.env'} and {here / 'docker-compose.yml'}"
        f" (Postgres on port {ports['PG_PORT']}, storage on {ports['S3_PORT']})\n"
        "next:\n"
        "  1. put your OPENAI_API_KEY (and OPENAI_BASE_URL for a gateway) in .env\n"
        "  2. docker compose up -d\n"
        "  3. contextread migrate\n"
        "  4. contextread check            (each model answers; the model names are in .env)\n"
        "  5. contextread index your.pdf\n"
        '  6. contextread ask "your question" --doc <id printed by index>'
    )


def _migrate(args) -> None:
    from . import migrate

    migrate.upgrade()
    print("database is at the latest schema")


def _serve(args) -> None:
    import uvicorn

    uvicorn.run("contextread.api.main:app", host=args.host, port=args.port, reload=args.reload)


def _worker(args) -> None:
    from .worker.main import main as worker_main

    worker_main()


def _list(args) -> None:
    for f in files.list_files("user", args.owner, limit=args.limit):
        print(f"{f['id']}  {f['status']:10} {f['page_num'] or '?':>5}p  {f['file_name']}")


def _ask(args) -> None:
    from .chat.agent import run_chat

    llm = BudgetedLLM()
    spent_before, started = llm.spent(), time.monotonic()
    result = run_chat(
        ("user", args.owner),
        [{"role": "user", "content": args.question}],
        llm,
        doc_ids=args.doc,
        enable_citations=args.cite,
    )
    print(result.text)
    pages = [c["arguments"].get("pages") for c in result.tool_calls if c["name"] == "get_page_content"]
    print(
        f"\n[{result.model}; {time.monotonic() - started:.1f}s; ${llm.spent() - spent_before:.4f}; "
        f"pages read: {', '.join(str(p) for p in pages) or 'none'}]"
    )


def _index(args) -> None:
    storage.ensure_bucket()
    data = Path(args.pdf).read_bytes()
    up = files.create_upload(args.owner, Path(args.pdf).name)
    httpx.put(up["url"], content=data, headers={"content-type": "application/pdf"}).raise_for_status()
    reg = files.register_upload("user", args.owner, Path(args.pdf).name, up["storage_key"])
    _run_index(reg["doc_id"], args.show_summaries)


def _reindex(args) -> None:
    from . import doctools
    from .db import session_scope

    with session_scope() as s:
        doc_id = doctools.resolve(s, ("user", args.owner), args.doc).id
    _run_index(doc_id, args.show_summaries, reuse_duplicate=False)


def _run_index(doc_id: str, show_summaries: bool, reuse_duplicate: bool = True) -> None:
    llm = BudgetedLLM()
    spent_before = llm.spent()
    started = time.monotonic()
    stats = pipeline.index_document(doc_id, llm=llm, reuse_duplicate=reuse_duplicate)
    seconds = time.monotonic() - started
    record = files.file_record(doc_id)
    nodes = tree.preorder(record["structure"] or [])
    print(
        json.dumps(
            {
                "doc_id": doc_id,
                **stats,
                "seconds": round(seconds, 1),
                "spend_usd": round(llm.spent() - spent_before, 6),
                "total_spend_usd": round(llm.spent(), 6),
                "cap_usd": llm.cap,
                "description": record["description"],
                "error_message": record["error_message"],
            },
            indent=1,
        )
    )
    for node, depth in nodes:
        print(
            f"{'  ' * (depth - 1)}- [{node['node_id']}] {node['title'][:70]} (p{node['start_index']}-{node['end_index']})"
        )
    if show_summaries:
        for node, _ in nodes[:5]:
            print(f"\n[{node['node_id']}] {node['title']}\n  {node.get('summary', '')}")


def _write_new(path: Path, data, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def _eval_make(args) -> None:
    from . import eval as ev

    owner = ("user", args.owner)
    docs = args.doc or ev.completed_documents(owner)
    if not docs:
        raise SystemExit("no indexed documents (contextread index FILE.pdf first)")
    if Path(args.out).exists() and not args.force:  # checked before any billed call
        raise SystemExit(f"{args.out} exists (use --force to overwrite)")
    llm = BudgetedLLM()
    before = llm.spent()
    questions = ev.make_questions(owner, docs, llm, per_doc=args.per_doc, model=args.model, seed=args.seed)
    _write_new(Path(args.out), questions, force=True)
    print(
        f"wrote {len(questions)} questions from {len(docs)} documents to {args.out} (${llm.spent() - before:.4f})\n"
        f"read them through, fix or drop any that are unclear, then: contextread eval run {args.out}"
    )


def _eval_run(args) -> None:
    from . import eval as ev

    questions = json.loads(Path(args.questions).read_text())
    earlier = json.loads(Path(args.against).read_text()) if args.against else None
    out = Path(args.out or f"results-{time.strftime('%Y%m%d-%H%M%S')}.json")
    if out.exists() and not args.force:  # checked before any billed call
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    llm = BudgetedLLM(cache=None if args.cache else False)
    before = llm.spent()
    print(
        f"asking {len(questions)} questions ({'cache on' if args.cache else 'cache off: fresh answers'}) ..."
    )
    results = ev.run_questions(("user", args.owner), questions, llm, judge_model=args.judge_model)
    _write_new(out, results, force=True)
    print(ev.report(results, llm.spent() - before))
    print(f"\nresults in {out}")
    if earlier is not None:
        print("\n" + ev.compare(earlier, results))


def _eval_compare(args) -> None:
    from . import eval as ev

    print(ev.compare(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text())))


def _check(args) -> None:
    from . import diagnostics

    if not diagnostics.check(models=not args.no_models):
        raise SystemExit(1)


def _show(args) -> None:
    from . import diagnostics

    diagnostics.show(("user", args.owner), args.doc, summaries=args.summaries, pages=args.pages)


def _create_key(args) -> None:
    user_id = auth.ensure_user(args.email)
    secret, key_id = auth.create_api_key(user_id, args.product, args.name)
    print(
        json.dumps(
            {
                "user_id": user_id,
                "key_id": key_id,
                "api_key": secret,
                "note": "shown once; store it now (only its hash is kept)",
            },
            indent=1,
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="contextread", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(required=True)

    i = sub.add_parser("init", help="write .env and docker-compose.yml into a folder")
    i.add_argument("dir", nargs="?", default=".")
    i.add_argument("--force", action="store_true", help="overwrite existing files")
    i.add_argument("--yes", action="store_true", help="do not ask; use the given or default model names")
    for key, flag, what in MODEL_SETTINGS:
        i.add_argument(f"--{flag.replace('_', '-')}", dest=flag, help=f"{key}: {what}")
    i.set_defaults(func=_init)

    m = sub.add_parser("migrate", help="create or upgrade the database tables")
    m.set_defaults(func=_migrate)

    p = sub.add_parser("index", help="index a PDF (billed, budget-guarded) and print its tree")
    p.add_argument("pdf")
    p.add_argument("--owner", default="dev-user")
    p.add_argument("--show-summaries", action="store_true")
    p.set_defaults(func=_index)

    ri = sub.add_parser("reindex", help="index a document again under the current settings (same id)")
    ri.add_argument("doc", help="document id or name")
    ri.add_argument("--owner", default="dev-user")
    ri.add_argument("--show-summaries", action="store_true")
    ri.set_defaults(func=_reindex)

    ls = sub.add_parser("list", help="the indexed documents and their ids")
    ls.add_argument("--owner", default="dev-user")
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(func=_list)

    a = sub.add_parser("ask", help="answer a question from indexed documents")
    a.add_argument("question")
    a.add_argument("--doc", action="append", required=True, help="document id (repeat for several)")
    a.add_argument("--owner", default="dev-user")
    a.add_argument("--cite", action="store_true", help="add block citations to the answer")
    a.set_defaults(func=_ask)

    sv = sub.add_parser("serve", help="run the API server (REST, chat, MCP)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=_serve)

    w = sub.add_parser("worker", help="run the indexing worker (documents uploaded through the API)")
    w.set_defaults(func=_worker)

    k = sub.add_parser("create-key", help="create a user (if needed) and an API key")
    k.add_argument("--email", required=True)
    k.add_argument("--product", choices=["cloud", "search"], default="cloud")
    k.add_argument("--name")
    k.set_defaults(func=_create_key)

    ck = sub.add_parser("check", help="is this setup ready: settings, database, storage, models")
    ck.add_argument("--no-models", action="store_true", help="skip the model calls (one tiny call per model)")
    ck.set_defaults(func=_check)

    sh = sub.add_parser("show", help="a document's tree as the index holds it, and the text of chosen pages")
    sh.add_argument("doc", help="document id or name")
    sh.add_argument("--summaries", action="store_true", help="each section's summary under its title")
    sh.add_argument("--pages", help="also print these pages' text, e.g. 12 or 10-14")
    sh.add_argument("--owner", default="dev-user")
    sh.set_defaults(func=_show)

    ev = sub.add_parser("eval", help="measure answers on your own documents")
    evs = ev.add_subparsers(required=True)
    em = evs.add_parser("make", help="write questions from pages of indexed documents (answer page known)")
    em.add_argument("--doc", action="append", help="document id or name (repeat); default: all indexed")
    em.add_argument("--per-doc", type=int, default=5, help="questions per document")
    em.add_argument("--out", default="questions.json")
    em.add_argument("--model", help="model that writes the questions (default CHAT_MODEL)")
    em.add_argument("--seed", type=int, default=0, help="which pages are picked")
    em.add_argument("--owner", default="dev-user")
    em.add_argument("--force", action="store_true", help="overwrite --out")
    em.set_defaults(func=_eval_make)
    er = evs.add_parser("run", help="ask each question in a new chat and judge the answers")
    er.add_argument("questions", help="questions file (from eval make, or written by hand)")
    er.add_argument("--out", help="results file (default results-<time>.json)")
    er.add_argument("--against", help="an earlier results file: also list the questions that changed")
    er.add_argument("--judge-model", help="model that judges the answers (default CHAT_MODEL)")
    er.add_argument(
        "--cache", action="store_true", help="reuse cached answers (default: fresh, as a user gets)"
    )
    er.add_argument("--owner", default="dev-user")
    er.add_argument("--force", action="store_true", help="overwrite --out")
    er.set_defaults(func=_eval_run)
    ec = evs.add_parser("compare", help="the questions whose result changed between two runs")
    ec.add_argument("before")
    ec.add_argument("after")
    ec.set_defaults(func=_eval_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

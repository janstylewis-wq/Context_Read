"""The `contextread` command.

contextread init               write .env and docker-compose.yml into this folder
contextread migrate            create or upgrade the database tables
contextread index FILE.pdf     index a PDF (billed, budget-guarded) and print its tree
contextread list               the indexed documents and their ids
contextread ask "Q" --doc ID   answer a question from one or more documents
contextread serve / worker     run the API server / the indexing worker
contextread create-key         create a user and an API key for the API and MCP server
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


def _init(args) -> None:
    here = Path(args.dir)
    templates = resources.files("contextread") / "templates"
    targets = {here / ".env": "env.template", here / "docker-compose.yml": "docker-compose.yml"}
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"already exists: {', '.join(existing)} (use --force to overwrite)")
    env = (templates / "env.template").read_text()
    for placeholder in ("__PG_PASSWORD__", "__S3_SECRET__", "__INTERNAL_TOKEN__"):
        env = env.replace(placeholder, secrets.token_urlsafe(24))
    here.mkdir(parents=True, exist_ok=True)
    (here / ".env").write_text(env)
    (here / ".env").chmod(0o600)
    (here / "docker-compose.yml").write_text((templates / "docker-compose.yml").read_text())
    print(
        f"wrote {here / '.env'} and {here / 'docker-compose.yml'}\n"
        "next:\n"
        "  1. put your OPENAI_API_KEY (and OPENAI_BASE_URL for a gateway) in .env\n"
        "  2. docker compose up -d\n"
        "  3. contextread migrate\n"
        "  4. contextread index your.pdf\n"
        '  5. contextread ask "your question" --doc <id printed by index>'
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
    llm = BudgetedLLM()
    spent_before = llm.spent()
    started = time.monotonic()
    stats = pipeline.index_document(reg["doc_id"], llm=llm)
    seconds = time.monotonic() - started
    record = files.file_record(reg["doc_id"])
    nodes = tree.preorder(record["structure"] or [])
    print(
        json.dumps(
            {
                "doc_id": reg["doc_id"],
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
    if args.show_summaries:
        for node, _ in nodes[:5]:
            print(f"\n[{node['node_id']}] {node['title']}\n  {node.get('summary', '')}")


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
    i.set_defaults(func=_init)

    m = sub.add_parser("migrate", help="create or upgrade the database tables")
    m.set_defaults(func=_migrate)

    p = sub.add_parser("index", help="index a PDF (billed, budget-guarded) and print its tree")
    p.add_argument("pdf")
    p.add_argument("--owner", default="dev-user")
    p.add_argument("--show-summaries", action="store_true")
    p.set_defaults(func=_index)

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

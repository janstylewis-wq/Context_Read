"""`contextread check` (is this setup ready: settings, database, storage, models) and `contextread show`
(a document as the index holds it: its tree, and the text of chosen pages)."""

import shutil
from collections import Counter

from sqlalchemy import select, text

from . import doctools, files
from .config import get_settings
from .db import session_scope
from .indexing import tree
from .models import Page


def _line(ok: bool | None, what: str, detail: str = "") -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
    print(f"[{mark}] {what}" + (f": {detail}" if detail else ""))
    return ok is not False


def check(models: bool = True) -> bool:
    """Print one line per check; False when something needed is not working."""
    try:
        s = get_settings()
    except Exception as e:  # noqa: BLE001 - a missing required setting
        return _line(False, "settings", f"{e.__class__.__name__}: {str(e).splitlines()[0]} (is .env here?)")
    good = _line(True, "settings", "loaded")
    good &= _database()
    good &= _storage()
    if models:
        good &= _models(s)
    if not shutil.which("tesseract"):
        _line(None, "tesseract", "not installed; a page the vision model fails on is left without text")
    return good


def _database() -> bool:
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    from . import migrate
    from .db import get_engine

    try:
        with get_engine().connect() as c:
            c.execute(text("select 1"))
            current = MigrationContext.configure(c).get_current_revision()
    except Exception as e:  # noqa: BLE001
        return _line(
            False, "database", f"{type(e).__name__}: {str(e).splitlines()[0]} (docker compose up -d?)"
        )
    head = ScriptDirectory.from_config(migrate.alembic_config()).get_current_head()
    if current != head:
        return _line(
            False, "database", f"schema {current or 'empty'}, latest {head}: run contextread migrate"
        )
    return _line(True, "database", "reachable, schema up to date")


def _storage() -> bool:
    from . import storage

    try:
        storage.ensure_bucket()
    except Exception as e:  # noqa: BLE001
        return _line(False, "storage", f"{type(e).__name__}: {str(e).splitlines()[0]}")
    return _line(True, "storage", f"bucket {storage.bucket()} ready")


def _models(s) -> bool:
    from .llm import BudgetedLLM

    if not s.openai_api_key:
        return _line(False, "models", "OPENAI_API_KEY is not set in .env")
    roles = {
        "CHAT_MODEL": s.chat_model,
        "LLM_DEFAULT_MODEL": s.llm_default_model,
        "OCR_VLM_MODEL": s.ocr_vlm_model,
        "TREE_EXPAND_MODEL": s.tree_expand_model,
    }
    llm, good, tried = BudgetedLLM(cache=False), True, {}
    for role, model in roles.items():
        if model == "off":
            _line(None, role, "off")
            continue
        if model not in tried:
            # a call shaped as the package makes them: temperature for a plain model, effort for a reasoning one
            options = {} if s.is_reasoning(model) else {"temperature": 0}
            try:
                llm.chat(
                    model,
                    [{"role": "user", "content": "Reply with: ok"}],
                    max_tokens=5,
                    purpose="check",
                    **options,
                )
                tried[model] = None
            except Exception as e:  # noqa: BLE001
                tried[model] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        error = tried[model]
        kind = "reasoning" if s.is_reasoning(model) else "plain"
        if error:
            hint = ""
            if "temperature" in error or "reasoning" in error:
                hint = " (a reasoning model? add its name prefix to REASONING_MODELS)"
            good = _line(False, role, f"{model}: {error}{hint}") and good
        else:
            _line(True, role, f"{model} answers ({kind} model)")
    print(f"       model spend so far: ${llm.spent():.4f} of ${llm.cap:.2f} (LLM_BUDGET_USD)")
    return good


def show(owner: tuple[str, str], ref: str, summaries: bool = False, pages: str | None = None) -> None:
    with session_scope() as s:
        f = doctools.resolve(s, owner, ref)
        modes = Counter(s.scalars(select(Page.read_mode).where(Page.file_id == f.id)))
        wanted = doctools.parse_pages(pages, f.page_num or 0) if pages else []
        page_rows = (
            s.execute(
                select(Page.page, Page.read_mode, Page.text)
                .where(Page.file_id == f.id, Page.page.in_(wanted))
                .order_by(Page.page)
            ).all()
            if wanted
            else []
        )
    record = files.file_record(f.id)
    how = ", ".join(f"{n} {m}" for m, n in sorted(modes.items())) or "none"
    print(f"{record['file_name']}  ({f.id})")
    print(
        f"status {record['status']}; {record['page_num']} pages (read as: {how}); index version {record['index_version']}"
    )
    if record.get("error_message"):
        print(f"error: {record['error_message']}")
    if record.get("description"):
        print(f"description: {record['description']}")
    print()
    for node, depth in tree.preorder(record["structure"] or []):
        indent = "  " * (depth - 1)
        print(f"{indent}- [{node['node_id']}] {node['title']} (p{node['start_index']}-{node['end_index']})")
        if summaries and node.get("summary"):
            print(f"{indent}    {node['summary']}")
    for page, mode, body in page_rows:
        print(
            f"\n===== page {page} (read as {mode}; {len(body or '')} characters) =====\n{body or '(no text)'}"
        )

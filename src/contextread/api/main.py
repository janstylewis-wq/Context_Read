from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .. import storage
from ..config import get_settings
from ..db import get_engine
from ..llm import BudgetedLLM
from ..mcp.server import mcp_app
from . import cloud, internal


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.ensure_bucket()
    async with mcp_app.lifespan():
        yield


app = FastAPI(title="ContextRead core", lifespan=lifespan)
app.include_router(internal.router)
app.include_router(cloud.router)


@app.get("/")
def root():
    """People who open the API in a browser land here; point them at the app."""
    settings = get_settings()
    base = settings.public_api_url.rstrip("/")
    return {
        "service": "ContextRead API",
        "app": settings.web_url,
        "mcp": f"{base}/mcp",
        "api_docs": f"{base}/docs",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    checks = {}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
    try:
        storage.get_client().head_bucket(Bucket=storage.bucket())
        checks["storage"] = "ok"
    except Exception as e:
        checks["storage"] = f"error: {type(e).__name__}"
    try:
        llm = BudgetedLLM()
        checks["llm_budget"] = {"spent_usd": round(llm.spent(), 6), "cap_usd": llm.cap}
    except Exception as e:
        checks["llm_budget"] = f"error: {type(e).__name__}"
    ok = checks["database"] == "ok" and checks["storage"] == "ok" and isinstance(checks["llm_budget"], dict)
    return JSONResponse({"ok": ok, "checks": checks}, status_code=200 if ok else 503)


# MCP (streamable HTTP at /mcp, plus its OAuth protected-resource metadata). Mounted last so every
# route above takes precedence.
app.mount("", mcp_app)

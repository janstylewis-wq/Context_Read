"""Request dependencies; tests override get_db_engine and get_llm on the app."""

from fastapi import Depends, Header, HTTPException
from sqlalchemy import Engine

from .. import auth
from ..db import get_engine
from ..llm import BudgetedLLM


def get_db_engine() -> Engine:
    return get_engine()


def get_llm() -> BudgetedLLM:
    return BudgetedLLM()


def cloud_principal(
    # the SDK sends the key in a header literally named "api_key" (underscore, not hyphen)
    api_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
    engine: Engine = Depends(get_db_engine),
) -> auth.Principal:
    secret = api_key
    if not secret and authorization and authorization.lower().startswith("bearer "):
        secret = authorization[7:].strip()
    principal = auth.authenticate(secret, "cloud", engine)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_api_key", "message": "The API key is missing, wrong or revoked."},
        )
    return principal

import pytest
from fakes import FakeClient, not_found_error
from test_budget import MSG, rows

from contextread.llm import BudgetedLLM
from contextread.llm.cache import ResponseCache


def test_repeat_request_is_served_from_cache(clean_db, tmp_path):
    client = FakeClient(headers={"x-litellm-response-cost": "0.001"})
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=ResponseCache(tmp_path))
    first = llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    second = llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert second.choices[0].message.content == first.choices[0].message.content
    assert len(client.calls) == 1
    assert len(rows(clean_db)) == 1  # the cache hit reserved nothing


def test_different_request_misses_and_bypass_forces_live(clean_db, tmp_path):
    client = FakeClient()
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=ResponseCache(tmp_path))
    llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    llm.chat("gpt-4o-mini", MSG, max_tokens=6)
    llm.chat("gpt-4o-mini", MSG, max_tokens=5, cache=False)
    assert len(client.calls) == 3


def test_failures_are_not_cached(clean_db, tmp_path):
    cache = ResponseCache(tmp_path)
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=FakeClient(error=not_found_error()), cache=cache)
    with pytest.raises(Exception, match="model not found"):
        llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert not any(tmp_path.rglob("*.json"))

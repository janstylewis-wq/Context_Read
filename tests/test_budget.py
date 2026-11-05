import psycopg
import pytest
from fakes import FakeClient, completion, not_found_error

from contextread.llm import BudgetedLLM, BudgetExceeded

MSG = [{"role": "user", "content": "Reply with the single word: ok"}]


def rows(dsn):
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            "SELECT status, cost_usd::float, cost_source FROM llm_spend ORDER BY id"
        ).fetchall()


def test_refuses_before_sending_when_over_cap(clean_db):
    client = FakeClient()
    llm = BudgetedLLM(dsn=clean_db, cap_usd=0.0000001, client=client, cache=False)
    with pytest.raises(BudgetExceeded):
        llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert client.calls == []
    assert rows(clean_db) == []


def test_settles_with_gateway_reported_cost(clean_db):
    client = FakeClient(headers={"x-litellm-response-cost": "0.000123"})
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=False)
    llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert rows(clean_db) == [("done", 0.000123, "gateway")]
    assert client.calls[0]["max_completion_tokens"] == 5


def test_falls_back_to_price_table(clean_db):
    client = FakeClient(resp=completion(prompt_tokens=1_000_000, completion_tokens=0))
    llm = BudgetedLLM(dsn=clean_db, cap_usd=100, client=client, cache=False)
    llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert rows(clean_db) == [("done", 0.15, "price-table")]


def test_rejected_request_costs_nothing(clean_db):
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=FakeClient(error=not_found_error()), cache=False)
    with pytest.raises(Exception, match="model not found"):
        llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    assert rows(clean_db) == [("failed", 0.0, "http-404")]


def test_spend_accumulates_toward_cap(clean_db):
    client = FakeClient(headers={"x-litellm-response-cost": "0.4"})
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=False)
    llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    llm.chat("gpt-4o-mini", MSG, max_tokens=5)
    with pytest.raises(BudgetExceeded):
        # 0.8 spent; the pessimistic estimate for this call pushes past 1.0
        llm.chat("unknown-model", MSG, max_tokens=10_000)
    assert len(client.calls) == 2


def test_providers_have_separate_caps(clean_db):
    gateway = BudgetedLLM(
        dsn=clean_db, cap_usd=1, client=FakeClient(headers={"x-litellm-response-cost": "0.9"}), cache=False
    )
    gateway.chat("gpt-4o-mini", MSG, max_tokens=5)
    qwen_client = FakeClient(resp=completion(model="Qwen/Qwen3.5-9B", estimated_cost=0.0002))
    qwen = BudgetedLLM(dsn=clean_db, cap_usd=0.5, client=qwen_client, cache=False, provider="deepinfra")
    qwen.chat("Qwen/Qwen3.5-9B", MSG, max_tokens=5)  # the gateway's $0.90 does not count here
    assert (gateway.spent(), qwen.spent()) == (0.9, 0.0002)
    assert rows(clean_db)[-1] == ("done", 0.0002, "provider")
    with pytest.raises(ValueError, match="provider"):
        BudgetedLLM(dsn=clean_db, client=FakeClient(), provider="nope")


def test_image_parts_are_estimated_by_tiles_not_base64_length(clean_db):
    page = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 3_000_000}}
    msg = [{"role": "user", "content": [{"type": "text", "text": "Transcribe."}, page]}]
    client = FakeClient(resp=completion(model="Qwen/Qwen3.5-9B"))
    # 3 MB of base64 read as text would reserve ~$0.10; as one image it is ~$0.002
    llm = BudgetedLLM(dsn=clean_db, cap_usd=0.01, client=client, cache=False, provider="deepinfra")
    llm.chat("Qwen/Qwen3.5-9B", msg, max_tokens=4000)
    assert len(client.calls) == 1


def test_stream_settles_from_usage_and_replays_from_cache(clean_db, tmp_path):
    from contextread.llm.cache import ResponseCache

    client = FakeClient(resp=completion("streamed answer", prompt_tokens=1_000_000, completion_tokens=0))
    llm = BudgetedLLM(dsn=clean_db, cap_usd=100, client=client, cache=ResponseCache(tmp_path))
    text = "".join(
        c.choices[0].delta.content or ""
        for c in llm.chat_stream("gpt-4o-mini", MSG, max_tokens=5)
        if c.choices
    )
    assert text == "streamed answer"
    assert rows(clean_db) == [("done", 0.15, "price-table")]
    assert client.calls[0]["stream"] is True and client.calls[0]["stream_options"] == {"include_usage": True}
    again = "".join(
        c.choices[0].delta.content or ""
        for c in llm.chat_stream("gpt-4o-mini", MSG, max_tokens=5)
        if c.choices
    )
    assert again == text and len(client.calls) == 1 and len(rows(clean_db)) == 1


def test_abandoned_stream_settles_at_the_estimate(clean_db):
    llm = BudgetedLLM(dsn=clean_db, cap_usd=1, client=FakeClient(resp=completion("x" * 100)), cache=False)
    stream = llm.chat_stream("gpt-4o-mini", MSG, max_tokens=5)
    next(stream)
    stream.close()  # the reader went away mid-answer
    ((status, cost, source),) = rows(clean_db)
    assert status == "failed" and cost > 0 and source == "estimate"


def test_real_client_must_use_canonical_ledger(clean_db):
    with pytest.raises(ValueError, match="canonical"):
        BudgetedLLM(dsn=clean_db)


@pytest.mark.live
def test_live_call_is_billed_to_main_ledger():
    llm = BudgetedLLM()  # real gateway client, canonical ledger
    before = llm.spent()
    resp = llm.chat("gpt-4o-mini", MSG, max_tokens=5, purpose="p0-live-test", cache=False)
    assert resp.choices[0].message.content
    after = llm.spent()
    assert 0 < after - before < 0.001
    assert after <= llm.cap

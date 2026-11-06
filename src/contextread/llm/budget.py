"""Hard USD spending cap for every LLM call made by ContextRead.

Each call first reserves its worst-case cost in the llm_spend ledger under an
advisory lock, so concurrent workers cannot jointly overshoot the cap. After
the call the reservation is replaced by the actual cost: the gateway's reported
``x-litellm-response-cost`` (or DeepInfra's ``usage.estimated_cost``) when present,
otherwise token usage x PRICES. Each provider has its own cap: a call that could push
that provider's spend past LLM_BUDGET_USD (OpenAI gateway) or DEEPINFRA_BUDGET_USD
(DeepInfra) is refused before anything is sent. Cache hits (llm/cache.py) send nothing and cost nothing.
"""

import json
import os

import psycopg
from openai import APIStatusError, OpenAI

from ..config import get_settings
from .cache import ResponseCache

# USD per 1M tokens (input, output). Unknown models fall back to DEFAULT_PRICE,
# deliberately pessimistic; override with LLM_PRICES='{"model": [in, out]}'.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-6-luna": (1.25, 10.0),  # unpublished; a ceiling for reservations, the gateway reports actual cost
    "Qwen/Qwen3.5-9B": (0.10, 0.15),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": (0.15, 0.60),
    "Qwen/Qwen3-VL-235B-A22B-Instruct": (0.20, 0.88),
}
PROVIDERS = ("openai", "deepinfra")
# images are priced by pixel tiles, not by their base64 length; this bounds a full page
IMAGE_TOKENS = 16_384
DEFAULT_PRICE = (15.0, 60.0)
_LOCK_KEY = 7_340_001


class BudgetExceeded(RuntimeError):
    pass


def _prices():
    prices = dict(PRICES)
    prices.update({k: tuple(v) for k, v in json.loads(os.environ.get("LLM_PRICES", "{}")).items()})
    return prices


def _price(model):
    return _prices().get(model, DEFAULT_PRICE)


def _cost(model, input_tokens, output_tokens):
    p_in, p_out = _price(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


# which models reason: REASONING_MODELS (settings), model name prefixes
REASONING_EFFORT = "low"
REASONING_MIN_TOKENS = 4000  # reasoning tokens count against the output limit


def _adapt(model, max_tokens, kwargs):
    """Request parameters for the model. A reasoning model takes reasoning_effort, not temperature, and
    spends output tokens on reasoning first, so an output limit sized for the answer alone (a 60-word summary)
    would leave it nothing to answer with. Callers pass what suits a plain model; this fits it to the one used."""
    if not get_settings().is_reasoning(model):
        return max_tokens, kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature", "top_p")}
    kwargs.setdefault("reasoning_effort", REASONING_EFFORT)
    return max(max_tokens, REASONING_MIN_TOKENS), kwargs


def _estimate_input_tokens(messages):
    # ~3 chars per token over-counts typical English, which is the safe direction
    chars = images = 0
    for m in messages:
        content = m.get("content", "")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "image_url":
                images += 1
            else:
                chars += len(json.dumps(part))
    return chars // 3 + images * IMAGE_TOKENS + 16 * len(messages)


def _psycopg_dsn(url):
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


class BudgetedLLM:
    def __init__(self, dsn=None, cap_usd=None, client=None, cache=None, provider="openai"):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown LLM provider {provider!r}")
        settings = get_settings()
        self.provider = provider
        canonical = _psycopg_dsn(settings.database_url)
        self.dsn = _psycopg_dsn(dsn) if dsn else canonical
        if client is None and self.dsn != canonical:
            # real, billed calls must land in the one ledger the cap is enforced on
            raise ValueError("a real LLM client must use the canonical APP_PG_DSN ledger")
        default_cap = settings.llm_budget_usd if provider == "openai" else settings.deepinfra_budget_usd
        self.cap = float(cap_usd if cap_usd is not None else default_cap)
        if client is None and provider == "deepinfra":
            client = OpenAI(
                api_key=settings.deepinfra_api_key,
                base_url=settings.deepinfra_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=2,
            )
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        # cache: None = per LLM_CACHE setting, False = disabled, or a ResponseCache
        if cache is None:
            cache = ResponseCache(settings.llm_cache_dir) if settings.llm_cache_enabled else False
        self.cache = cache or None

    def spent(self):
        with psycopg.connect(self.dsn) as conn:
            return self._spent(conn)

    def _spent(self, conn):
        return float(
            conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_spend WHERE provider = %s", (self.provider,)
            ).fetchone()[0]
        )

    def _reserve(self, model, purpose, estimate):
        with psycopg.connect(self.dsn) as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
            spent = self._spent(conn)
            if spent + estimate > self.cap:
                raise BudgetExceeded(
                    f"{self.provider} LLM budget ${self.cap:.2f} would be exceeded: spent ${spent:.4f}, "
                    f"this call may cost up to ${estimate:.4f}"
                )
            return conn.execute(
                "INSERT INTO llm_spend (model, purpose, status, cost_usd, cost_source, provider) "
                "VALUES (%s, %s, 'reserved', %s, 'estimate', %s) RETURNING id",
                (model, purpose, estimate, self.provider),
            ).fetchone()[0]

    def _settle(self, row_id, status, cost, input_tokens=None, output_tokens=None, source=None):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                "UPDATE llm_spend SET status=%s, cost_usd=%s, input_tokens=%s, output_tokens=%s, "
                "cost_source=%s WHERE id=%s",
                (status, cost, input_tokens, output_tokens, source, row_id),
            )

    def chat(self, model=None, messages=(), max_tokens=1024, purpose=None, cache=True, **kwargs):
        """chat.completions.create with the cap enforced. max_tokens is required
        so the worst-case output cost is bounded. cache=False forces a live call."""
        model = model or get_settings().llm_default_model
        max_tokens, kwargs = _adapt(model, max_tokens, kwargs)
        messages = list(messages)
        key = None
        if cache and self.cache is not None:
            key = self.cache.key(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        estimate = _cost(model, _estimate_input_tokens(messages), max_tokens)
        row_id = self._reserve(model, purpose, estimate)
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=model, messages=messages, max_completion_tokens=max_tokens, **kwargs
            )
        except APIStatusError as e:
            # requests rejected before generation are not billed; anything else might have been
            unbilled = e.status_code in (400, 401, 403, 404, 422)
            self._settle(row_id, "failed", 0 if unbilled else estimate, source=f"http-{e.status_code}")
            raise
        except Exception:
            self._settle(row_id, "failed", estimate, source="estimate")
            raise
        resp = raw.parse()
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        header = raw.headers.get("x-litellm-response-cost")
        reported = (usage.model_extra or {}).get("estimated_cost") if usage else None
        if header:
            cost, source = float(header), "gateway"
        elif reported is not None:
            cost, source = float(reported), "provider"
        elif usage:
            cost, source = _cost(model, in_tok, out_tok), "price-table"
        else:
            cost, source = estimate, "estimate"
        self._settle(row_id, "done", cost, in_tok, out_tok, source)
        if key is not None:
            self.cache.put(key, resp)
        return resp

    def chat_stream(self, model=None, messages=(), max_tokens=1024, purpose=None, cache=True, **kwargs):
        """Streaming chat.completions.create with the cap enforced: yields the chunks as they arrive.
        The cost settles from the final usage chunk; a stream abandoned midway settles at the
        reserved estimate, since the provider may have billed it."""
        model = model or get_settings().llm_default_model
        max_tokens, kwargs = _adapt(model, max_tokens, kwargs)
        messages = list(messages)
        key = None
        if cache and self.cache is not None:
            key = self.cache.key(model=model, messages=messages, max_tokens=max_tokens, stream=True, **kwargs)
            hit = self.cache.get_stream(key)
            if hit is not None:
                yield from hit
                return

        estimate = _cost(model, _estimate_input_tokens(messages), max_tokens)
        row_id = self._reserve(model, purpose, estimate)
        settled = False
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            chunks, usage = [], None
            for chunk in stream:
                chunks.append(chunk)
                usage = chunk.usage or usage
                yield chunk
            if usage:
                in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
                reported = (usage.model_extra or {}).get("estimated_cost")
                cost, source = (
                    (float(reported), "provider")
                    if reported is not None
                    else (_cost(model, in_tok, out_tok), "price-table")
                )
                self._settle(row_id, "done", cost, in_tok, out_tok, source)
            else:
                self._settle(row_id, "done", estimate, source="estimate")
            settled = True
            if key is not None:
                self.cache.put_stream(key, chunks)
        except APIStatusError as e:
            unbilled = e.status_code in (400, 401, 403, 404, 422)
            self._settle(row_id, "failed", 0 if unbilled else estimate, source=f"http-{e.status_code}")
            settled = True
            raise
        finally:
            if not settled:  # an error mid-stream, or the caller stopped reading
                self._settle(row_id, "failed", estimate, source="estimate")

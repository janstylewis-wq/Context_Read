"""Reasoning models get reasoning_effort instead of temperature, and room in the output limit to reason."""

from contextread.llm.budget import REASONING_MIN_TOKENS, _adapt


def test_reasoning_model_drops_temperature_and_gets_room_to_reason():
    max_tokens, kwargs = _adapt(
        "gpt-6-luna", 150, {"temperature": 0, "response_format": {"type": "json_object"}}
    )
    assert max_tokens == REASONING_MIN_TOKENS
    assert kwargs == {"reasoning_effort": "low", "response_format": {"type": "json_object"}}


def test_explicit_effort_and_larger_limit_are_kept():
    assert _adapt("gpt-6-luna", 9000, {"reasoning_effort": "medium"}) == (
        9000,
        {"reasoning_effort": "medium"},
    )


def test_plain_model_is_unchanged():
    assert _adapt("gpt-4o-mini", 150, {"temperature": 0}) == (150, {"temperature": 0})

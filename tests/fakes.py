"""A stand-in for the OpenAI client's chat.completions.with_raw_response surface."""

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def completion(
    text="ok", prompt_tokens=10, completion_tokens=1, model="gpt-4o-mini", **usage_extra
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                **usage_extra,
            },
        }
    )


def as_chunks(resp: ChatCompletion, piece: int = 12) -> list[ChatCompletionChunk]:
    """A completion as the chunks a streaming call returns: text in pieces, each tool call in one
    delta, then a usage-only chunk (stream_options include_usage)."""
    choice = resp.choices[0]
    base = {"id": resp.id, "object": "chat.completion.chunk", "created": 0, "model": resp.model}
    deltas = []
    text = choice.message.content or ""
    for i in range(0, len(text), piece):
        deltas.append({"content": text[i : i + piece]})
    for i, tc in enumerate(choice.message.tool_calls or []):
        fn = {"name": tc.function.name, "arguments": tc.function.arguments}
        deltas.append({"tool_calls": [{"index": i, "id": tc.id, "type": "function", "function": fn}]})
    chunks = [
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
        )
        for d in deltas
    ]
    chunks.append(
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.finish_reason}]}
        )
    )
    chunks.append(
        ChatCompletionChunk.model_validate({**base, "choices": [], "usage": resp.usage.model_dump()})
    )
    return chunks


class _Raw:
    def __init__(self, resp, headers):
        self._resp, self.headers = resp, headers

    def parse(self):
        return self._resp


class FakeClient:
    def __init__(self, resp=None, headers=None, error=None):
        self.calls = []
        self._resp, self._headers, self._error = resp or completion(), headers or {}, error
        self.chat = self
        self.completions = self
        self.with_raw_response = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._reply(self._resp, kwargs)

    def _reply(self, resp, kwargs):
        return iter(as_chunks(resp)) if kwargs.get("stream") else _Raw(resp, self._headers)


def tool_call_completion(name: str, arguments: dict) -> ChatCompletion:
    import json

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
    )


class ScriptedClient(FakeClient):
    """Plays the roles the server asks of the LLM: chat agent (one page read, then a cited answer),
    retrieval (JSON node ids) and summaries (fixed text)."""

    def __init__(self, doc_name: str, answer: str, node_ids=("0001",)):
        super().__init__(headers={"x-litellm-response-cost": "0.00001"})
        self.doc_name, self.answer, self.node_ids = doc_name, answer, list(node_ids)

    def create(self, **kwargs):
        import json

        self.calls.append(kwargs)
        messages = kwargs["messages"]
        if kwargs.get("response_format", {}).get("type") == "json_object":
            resp = completion(json.dumps({"node_ids": self.node_ids}))
        elif "tools" in kwargs and not any(m.get("role") == "tool" for m in messages):
            resp = tool_call_completion("get_page_content", {"doc_name": self.doc_name, "pages": "1"})
        elif any(m.get("role") == "tool" for m in messages):
            resp = completion(self.answer)
        else:
            resp = completion("A short factual summary.")
        return self._reply(resp, kwargs)


def not_found_error() -> openai.NotFoundError:
    request = httpx.Request("POST", "http://gateway.test/v1/chat/completions")
    return openai.NotFoundError("model not found", response=httpx.Response(404, request=request), body=None)

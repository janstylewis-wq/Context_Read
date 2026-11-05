"""On-disk cache of chat completions, keyed by the full request, so re-running dev
and test code does not spend budget. Enabled by LLM_CACHE (default on); never
caches failures."""

import hashlib
import json
import os
from pathlib import Path

from openai.types.chat import ChatCompletion, ChatCompletionChunk


class ResponseCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @staticmethod
    def key(**request) -> str:
        blob = json.dumps(request, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> ChatCompletion | None:
        path = self._path(key)
        if not path.exists():
            return None
        return ChatCompletion.model_validate_json(path.read_text())

    def put(self, key: str, response: ChatCompletion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json())
        os.replace(tmp, path)  # atomic: concurrent writers never leave a torn file

    def get_stream(self, key: str) -> list[ChatCompletionChunk] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return [ChatCompletionChunk.model_validate(c) for c in json.loads(path.read_text())]

    def put_stream(self, key: str, chunks: list[ChatCompletionChunk]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps([c.model_dump(mode="json") for c in chunks]))
        os.replace(tmp, path)

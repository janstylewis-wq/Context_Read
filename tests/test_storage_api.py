import httpx
from fastapi.testclient import TestClient

from contextread import storage
from contextread.api.main import app
from contextread.config import get_settings


def test_presigned_put_then_get_roundtrip(bucket):
    key = "tests/roundtrip.pdf"
    data = b"%PDF-1.4 test bytes"
    put = httpx.put(
        storage.presign_put(key, "application/pdf"), content=data, headers={"content-type": "application/pdf"}
    )
    assert put.status_code == 200
    assert storage.get_bytes(key) == data
    assert httpx.get(storage.presign_get(key)).content == data


def test_health_and_readiness(bucket):
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True}
        ready = client.get("/readyz")
        body = ready.json()
        assert ready.status_code == 200, body
        assert body["checks"]["database"] == "ok" and body["checks"]["storage"] == "ok"
        assert body["checks"]["llm_budget"]["cap_usd"] == get_settings().llm_budget_usd

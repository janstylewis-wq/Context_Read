import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from test_indexing import DISNEY

from contextread import accounts, config
from contextread.api.main import app
from contextread.db import session_scope
from contextread.models import AuthCode, File, User


@pytest.fixture
def mail_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_MAIL_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def fresh(engine):
    with session_scope(engine) as s:
        s.execute(AuthCode.__table__.delete())
        s.execute(File.__table__.delete())
        s.execute(User.__table__.delete().where(User.email.like("%@acct.test")))


def last_mail(mail_dir) -> dict:
    return json.loads(sorted(mail_dir.glob("*.json"))[-1].read_text())


def _age_codes(engine, seconds=60):
    """Make earlier codes look older than the resend cooldown."""
    with session_scope(engine) as s:
        s.execute(
            update(AuthCode).values(
                created_at=AuthCode.created_at - __import__("datetime").timedelta(seconds=seconds)
            )
        )


def test_email_code_signs_up_then_signs_in(engine, fresh, mail_dir):
    assert accounts.request_login_code("New.User@Acct.test")["sent"]
    mail = last_mail(mail_dir)
    assert mail["to"] == "new.user@acct.test" and mail["code"] in mail["subject"]
    user = accounts.verify_login_code("new.user@acct.test", mail["code"])
    assert user["email"] == "new.user@acct.test" and user["emailVerified"] and not user["hasPassword"]
    with pytest.raises(accounts.AccountError, match="wrong or has expired"):
        accounts.verify_login_code("new.user@acct.test", mail["code"])  # single use
    _age_codes(engine)
    accounts.request_login_code("new.user@acct.test")
    again = accounts.verify_login_code("new.user@acct.test", last_mail(mail_dir)["code"])
    assert again["id"] == user["id"]


def test_resend_cooldown_and_attempt_limit(engine, fresh, mail_dir):
    accounts.request_login_code("guess@acct.test")
    with pytest.raises(accounts.AccountError) as e:
        accounts.request_login_code("guess@acct.test")
    assert e.value.status == 429
    code = last_mail(mail_dir)["code"]
    wrong = f"{(int(code) + 1) % 1_000_000:06d}"
    for _ in range(accounts.MAX_ATTEMPTS):
        with pytest.raises(accounts.AccountError):
            accounts.verify_login_code("guess@acct.test", wrong)
    with pytest.raises(accounts.AccountError):
        accounts.verify_login_code("guess@acct.test", code)  # locked after too many guesses


def test_magic_link(engine, fresh, mail_dir):
    accounts.request_login_code("link@acct.test")
    link = last_mail(mail_dir)["link"]
    token = httpx.URL(link).params["token"]
    assert link.startswith(config.get_settings().web_url + "/magic-link?")
    assert accounts.verify_magic_link("link@acct.test", token)["email"] == "link@acct.test"
    with pytest.raises(accounts.AccountError):
        accounts.verify_magic_link("link@acct.test", token)


def test_password_set_login_and_reset(engine, fresh, mail_dir):
    accounts.request_login_code("pw@acct.test")
    user = accounts.verify_login_code("pw@acct.test", last_mail(mail_dir)["code"])
    with pytest.raises(accounts.AccountError, match="at least"):
        accounts.set_password(user["id"], "short")
    assert accounts.set_password(user["id"], "correct horse 1")["hasPassword"]
    with pytest.raises(accounts.AccountError, match="current password"):
        accounts.set_password(user["id"], "another pass 2", current_password="nope")
    assert accounts.password_login("pw@acct.test", "correct horse 1")["id"] == user["id"]
    for email, pw in (("pw@acct.test", "wrong pass"), ("nobody@acct.test", "whatever1")):
        with pytest.raises(accounts.AccountError) as e:
            accounts.password_login(email, pw)
        assert e.value.code == "invalid_credentials"  # the same answer for a missing account

    before = len(list(mail_dir.glob("*.json")))
    assert accounts.request_password_reset("nobody@acct.test") == {"sent": True}
    assert len(list(mail_dir.glob("*.json"))) == before  # no mail, but the same response
    accounts.request_password_reset("pw@acct.test")
    token = httpx.URL(last_mail(mail_dir)["link"]).params["token"]
    accounts.reset_password("pw@acct.test", token, "brand new pass")
    assert accounts.password_login("pw@acct.test", "brand new pass")["id"] == user["id"]


def test_oauth_links_to_existing_user(engine, fresh, mail_dir):
    accounts.request_login_code("oauth@acct.test")
    user = accounts.verify_login_code("oauth@acct.test", last_mail(mail_dir)["code"])
    linked = accounts.oauth_login("github", "gh-123", "oauth@acct.test", "Octo", "https://img")
    assert linked["id"] == user["id"]
    assert accounts.oauth_login("github", "gh-123", "changed@acct.test")["id"] == user["id"]


def test_internal_routes_profile_usage_limits_and_ownership(engine, fresh, mail_dir, bucket):
    token = config.get_settings().internal_api_token
    h = {"x-internal-token": token}
    with TestClient(app) as c:
        assert c.post("/internal/auth/request-code", json={"email": "bad"}, headers=h).status_code == 400
        assert (
            c.post("/internal/auth/request-code", json={"email": "web@acct.test"}, headers=h).status_code
            == 200
        )
        user = c.post(
            "/internal/auth/verify-code",
            headers=h,
            json={"email": "web@acct.test", "code": last_mail(mail_dir)["code"]},
        ).json()
        uid = user["id"]
        assert (
            c.patch(f"/internal/users/{uid}", headers=h, json={"name": "Web User"}).json()["name"]
            == "Web User"
        )
        assert c.get(f"/internal/users/{uid}/usage", headers=h).json()["page"] == {
            "total": 0,
            "limit": 1000,
            "available": 1000,
        }

        up = c.post("/internal/uploads", headers=h, json={"owner_id": uid, "file_name": "disney.pdf"}).json()
        httpx.put(up["url"], content=DISNEY.read_bytes(), headers={"content-type": "application/pdf"})
        over = c.post(
            "/internal/files",
            headers=h,
            json={
                "owner_id": uid,
                "file_name": "disney.pdf",
                "storage_key": up["storage_key"],
                "page_limit": 10,
            },
        )
        assert over.status_code == 402 and over.json()["detail"]["code"] == "page_limit"
        reg = c.post(
            "/internal/files",
            headers=h,
            json={
                "owner_id": uid,
                "file_name": "disney.pdf",
                "storage_key": up["storage_key"],
                "page_limit": 1000,
            },
        ).json()
        assert c.get(f"/internal/users/{uid}/usage", headers=h).json()["page"]["total"] == 22

        folder = c.post("/internal/folders", headers=h, json={"owner_id": uid, "name": "Reports"}).json()
        child = c.post(
            "/internal/folders", headers=h, json={"owner_id": uid, "name": "2025", "parent_id": folder["id"]}
        ).json()
        moved = c.patch(
            f"/internal/files/{reg['doc_id']}",
            headers=h,
            json={"owner_id": uid, "move": True, "folder_id": child["id"], "file_name": "Q1.pdf"},
        ).json()
        assert moved["folderId"] == child["id"] and moved["fileName"] == "Q1.pdf"
        tree = c.get("/internal/folders", headers=h, params={"owner_id": uid}).json()["folders"]
        assert tree[0]["name"] == "Reports" and tree[0]["children"][0]["fileCount"] == 1
        cycle = c.patch(
            f"/internal/folders/{folder['id']}",
            headers=h,
            json={"owner_id": uid, "move": True, "parent_id": child["id"]},
        )
        assert cycle.status_code == 400

        # another user's id cannot see or change this document
        stranger = {"owner_id": "someone-else"}
        assert c.get(f"/internal/files/{reg['doc_id']}", headers=h, params=stranger).status_code == 404
        assert (
            c.patch(
                f"/internal/files/{reg['doc_id']}", headers=h, json={**stranger, "file_name": "x.pdf"}
            ).status_code
            == 404
        )
        assert c.delete(f"/internal/files/{reg['doc_id']}", headers=h, params=stranger).status_code == 404

        listing = c.get(
            "/internal/files", headers=h, params={"owner_id": uid, "folder_id": child["id"]}
        ).json()
        assert [f["id"] for f in listing["items"]] == [reg["doc_id"]]
        assert c.delete(f"/internal/folders/{folder['id']}", headers=h, params={"owner_id": uid}).json() == {
            "deleted": 2,
            "documentsMoved": 1,
        }
        assert c.delete(f"/internal/users/{uid}/files", headers=h).json() == {"deleted": 1}


def test_no_plan_limits(monkeypatch):
    monkeypatch.setenv("PAGE_LIMIT", "0")
    monkeypatch.setenv("CHAT_LIMIT", "0")
    config.get_settings.cache_clear()
    try:
        assert accounts.page_limit() is None and accounts.chat_limit() is None
    finally:
        monkeypatch.undo()
        config.get_settings.cache_clear()
    assert accounts.page_limit() == 1000 and accounts.chat_limit() == 100

"""contextread init: the model names are asked for (or given as flags, or the defaults in a script), free ports
are chosen, and the compose project is named after the folder, so two folders never share a database."""

import argparse
import re
import socket

from contextread import cli
from contextread.config import Settings


def _args(tmp_path, **flags):
    ns = {"dir": str(tmp_path), "force": False, "yes": False}
    ns.update({flag: None for _, flag, _ in cli.MODEL_SETTINGS})
    ns.update(flags)
    return argparse.Namespace(**ns)


def _env(tmp_path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Z0-9_]+)=(.*)$", (tmp_path / ".env").read_text(), flags=re.M))


def test_init_asks_for_the_model_names_at_a_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["o4-mini", "", "gpt-4o", "off", "o3,o4"])  # Enter keeps the summary model's default
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli._init(_args(tmp_path))
    env = _env(tmp_path)
    assert (
        env["CHAT_MODEL"] == "o4-mini"
        and env["OCR_VLM_MODEL"] == "gpt-4o"
        and env["TREE_EXPAND_MODEL"] == "off"
    )
    assert env["LLM_DEFAULT_MODEL"] == "gpt-4o-mini" and env["REASONING_MODELS"] == "o3,o4"
    assert len(env["APP_PG_PASSWORD"]) > 20 and "__" not in env["S3_SECRET_ACCESS_KEY"]


def test_flags_and_scripts_are_not_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(AssertionError("asked")))
    cli._init(_args(tmp_path, chat_model="my-model"))
    env = _env(tmp_path)
    assert (
        env["CHAT_MODEL"] == "my-model"
        and env["LLM_DEFAULT_MODEL"] == Settings.model_fields["llm_default_model"].default
    )


def test_a_busy_port_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        assert cli._free_port(taken) != taken


def test_the_compose_project_is_named_after_its_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    cli._init(_args(tmp_path))
    compose = (tmp_path / "docker-compose.yml").read_text()
    assert not re.search(r"^name:", compose, flags=re.M) and "container_name" not in compose


def test_the_deepinfra_key_takes_its_standard_name(monkeypatch):
    monkeypatch.delenv("deep_infra", raising=False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "k1")
    assert Settings(_env_file=None, APP_PG_DSN="postgresql://x@h/d").deepinfra_api_key == "k1"
    monkeypatch.delenv("DEEPINFRA_API_KEY")
    monkeypatch.setenv("deep_infra", "k2")
    assert Settings(_env_file=None, APP_PG_DSN="postgresql://x@h/d").deepinfra_api_key == "k2"

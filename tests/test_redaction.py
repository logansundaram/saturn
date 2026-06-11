"""redaction.py — the outbound secret-stripping guard on the cloud boundary."""

import pytest

import redaction
from config import get_config


@pytest.fixture(autouse=True)
def reset_mode(monkeypatch):
    """Pin the mode to a known value around each test (default off)."""
    monkeypatch.setitem(get_config()._data["runtime"], "redaction", "off")
    yield


def _set_mode(monkeypatch, m):
    monkeypatch.setitem(get_config()._data["runtime"], "redaction", m)


def test_detects_common_secret_kinds():
    text = (
        "anthropic sk-ant-abcdefghij1234567890ABCD, "
        "openai sk-proj-abcdefghij1234567890ABCD, "
        "tavily tvly-abcdefghij1234567890, "
        "email a.b@example.com, "
        "bearer Bearer abcdefghij1234567890XYZ"
    )
    kinds = {f.kind for f in redaction.scan(text)}
    assert {"anthropic-key", "openai-key", "tavily-key", "email", "bearer-token"} <= kinds


def test_anthropic_key_not_double_counted_as_openai():
    """An sk-ant- key must register once (anthropic), not also as an openai sk- key."""
    finds = redaction.scan("key sk-ant-abcdefghij1234567890ABCD here")
    kinds = [f.kind for f in finds]
    assert kinds.count("anthropic-key") == 1
    assert "openai-key" not in kinds


def test_redact_replaces_with_placeholder():
    new, finds = redaction.redact("token sk-ant-abcdefghij1234567890ABCD done")
    assert "sk-ant-" not in new
    assert "[REDACTED:anthropic-key]" in new
    assert len(finds) == 1


def test_preview_never_exposes_full_secret():
    secret = "sk-ant-abcdefghij1234567890ABCDEFGH"
    f = redaction.scan(f"x {secret} y")[0]
    assert secret not in f.preview
    assert "…" in f.preview


def test_no_false_positive_on_ordinary_prose():
    assert redaction.scan("The quick brown fox jumps over 12 lazy dogs.") == []


def test_mode_off_is_passthrough(monkeypatch):
    from langchain.messages import HumanMessage

    _set_mode(monkeypatch, "off")
    msgs = [HumanMessage(content="sk-ant-abcdefghij1234567890ABCD")]
    out, n = redaction.process_messages(msgs)
    assert n == 0
    assert out is msgs  # untouched


def test_warn_counts_but_does_not_modify(monkeypatch):
    from langchain.messages import HumanMessage

    _set_mode(monkeypatch, "warn")
    original = "leak sk-ant-abcdefghij1234567890ABCD"
    msgs = [HumanMessage(content=original)]
    out, n = redaction.process_messages(msgs)
    assert n == 1
    assert out[0].content == original  # warn never alters the payload


def test_redact_mode_modifies_copy_not_original(monkeypatch):
    from langchain.messages import HumanMessage

    _set_mode(monkeypatch, "redact")
    original = "leak sk-ant-abcdefghij1234567890ABCD"
    msg = HumanMessage(content=original)
    out, n = redaction.process_messages([msg])
    assert n == 1
    assert "[REDACTED:anthropic-key]" in out[0].content
    assert msg.content == original  # the scratchpad message object is not mutated


def test_active_reflects_mode(monkeypatch):
    _set_mode(monkeypatch, "off")
    assert not redaction.active()
    _set_mode(monkeypatch, "warn")
    assert redaction.active()
    _set_mode(monkeypatch, "redact")
    assert redaction.active()


def test_unknown_mode_falls_back_to_off(monkeypatch):
    _set_mode(monkeypatch, "banana")
    assert redaction.mode() == "off"

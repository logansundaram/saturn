"""
/config persist-by-default (2026-07-07) with the trust-key exemption (2026-07-10): a security
posture set through the generic setter applies for the SESSION and never writes config.yaml
without an explicit --save — the same fail-closed convention the canonical toggles
(/policy open, /privacy airgap) keep via the opt-IN save parser — runtime.redaction stays a
trust key even though its command front end was cut 2026-07-16 (/config is its only door now).
Ordinary settings keep the persist-by-default inversion.
"""

import pytest

from commands._framework import CommandContext
from commands.config import _TRUST_KEYS, _config


@pytest.fixture
def ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


@pytest.fixture
def recording_persist(monkeypatch):
    """Capture config.persist calls instead of writing the real config.yaml."""
    import config

    saved: list[str] = []
    monkeypatch.setattr(config, "persist", lambda key: saved.append(key) or config._CONFIG_PATH)
    return saved


@pytest.fixture(autouse=True)
def _restore_runtime():
    """Snapshot the live runtime section so posture edits can't leak across tests."""
    from config import get_config

    runtime = get_config()._data.setdefault("runtime", {})
    snap = dict(runtime)
    yield
    runtime.clear()
    runtime.update(snap)


def _out(capsys) -> str:
    return capsys.readouterr().out


@pytest.mark.parametrize("key,value", [
    ("runtime.quarantine", "off"),
    ("runtime.airgap", "false"),
    ("runtime.auto_approve", "destructive"),
    ("runtime.redaction", "off"),
])
def test_trust_key_set_is_session_only_by_default(ctx, capsys, recording_persist, key, value):
    from config import get_config

    _config(ctx, [key, value])
    out = _out(capsys)
    assert str(get_config().get(key)).lower() == value  # applied live for the session
    assert recording_persist == [], "a loosened posture must never persist silently"
    assert "session only" in out and "--save" in out


def test_trust_key_persists_with_explicit_save(ctx, capsys, recording_persist):
    _config(ctx, ["runtime.auto_approve", "side_effecting", "--save"])
    assert recording_persist == ["runtime.auto_approve"]


def test_ordinary_setting_still_persists_by_default(ctx, capsys, recording_persist):
    _config(ctx, ["runtime.max_iterations", "9"])
    assert recording_persist == ["runtime.max_iterations"]


def test_every_trust_key_is_a_real_config_leaf():
    """A typo'd guard key would silently lose its protection — pin each one to a live leaf in
    the shipped config."""
    from config import get_config

    missing = object()
    for key in _TRUST_KEYS:
        assert get_config().get(key, missing) is not missing, key

"""
/config persist-by-default with the trust-key exemption: a security posture set through the
generic setter applies for the SESSION and never writes config.yaml without an explicit --save —
the same fail-closed convention the canonical toggles (/policy open, /policy airgap) keep via the
opt-IN save parser. Ordinary settings persist by default.
"""

import pytest

from commands.config import _TRUST_KEYS, _config


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

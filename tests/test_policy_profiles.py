"""
Policy profiles (policy.export_profile / apply_profile) — the round trip, the replace semantics,
validation failing closed, and the YAML `off`-is-False footgun.
"""

import pytest

from trust import policy
from config import get_config


@pytest.fixture
def clean_policy(isolated_paths, monkeypatch):
    """Isolated permissions.json + session-scoped runtime knobs restored afterward."""
    cfg = get_config()
    runtime = cfg._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "auto_approve", "read_only")
    monkeypatch.setitem(runtime, "airgap", False)
    monkeypatch.setitem(runtime, "redaction", "off")
    return isolated_paths


def test_export_round_trips_through_apply(clean_policy):
    policy.set_risk_override("web_search", "read_only")
    policy.add_shell_allow("git status")
    profile = policy.export_profile()
    assert profile["saturn_policy"] == policy.PROFILE_VERSION
    assert profile["risk_overrides"] == {"web_search": "read_only"}
    assert profile["shell_allow"] == ["git status"]

    # wipe, then re-apply the export — the posture comes back exactly
    policy.clear_risk_override("web_search")
    policy.remove_shell_allow("git status")
    assert policy.risk_overrides() == {}
    overrides = policy.apply_profile(profile)
    assert overrides == {"web_search": "read_only"}
    assert policy.risk_overrides() == {"web_search": "read_only"}
    assert policy.shell_allow() == ["git status"]


def test_apply_replaces_not_merges(clean_policy):
    policy.set_risk_override("web_search", "read_only")
    policy.add_shell_allow("git status")
    profile = {
        "saturn_policy": 1,
        "auto_approve": "side_effecting",
        "risk_overrides": {"web_extract": "read_only"},
        "shell_allow": ["git log"],
    }
    policy.apply_profile(profile)
    # the previous posture is gone, not layered under
    assert policy.risk_overrides() == {"web_extract": "read_only"}
    assert policy.shell_allow() == ["git log"]
    assert policy.tier() == "side_effecting"


def test_apply_rejects_non_profiles(clean_policy):
    with pytest.raises(ValueError):
        policy.apply_profile({"random": "mapping"})
    with pytest.raises(ValueError):
        policy.apply_profile("not a mapping")
    with pytest.raises(ValueError):
        policy.apply_profile({"saturn_policy": 999})


def test_apply_rejects_invalid_tier_and_changes_nothing(clean_policy):
    # An invalid tier is a HARD error like an invalid auto_approve — silently dropping it would
    # let a typo'd override (often one meant to RAISE a tool's tier) vanish from a profile that
    # reports success, leaving the gate weaker than the operator believes.
    policy.set_risk_override("web_search", "read_only")
    profile = {
        "saturn_policy": 1,
        "risk_overrides": {"web_extract": "read_only", "evil": "nuclear"},
    }
    with pytest.raises(ValueError):
        policy.apply_profile(profile)
    assert policy.risk_overrides() == {"web_search": "read_only"}  # durable file untouched


def test_apply_normalizes_blank_prefixes(clean_policy):
    profile = {
        "saturn_policy": 1,
        "risk_overrides": {"web_search": "read_only"},
        "shell_allow": ["  git   status  ", "", "   "],
    }
    overrides = policy.apply_profile(profile)
    assert overrides == {"web_search": "read_only"}
    assert policy.shell_allow() == ["git status"]    # whitespace normalized, blanks dropped


def test_yaml_bare_off_redaction_coerces(clean_policy):
    # A hand-written `redaction: off` parses as boolean False in YAML — apply must read it as "off".
    profile = {"saturn_policy": 1, "redaction": False, "airgap": True}
    policy.apply_profile(profile)
    cfg = get_config()
    assert cfg.get("runtime.redaction") == "off"
    assert cfg.get("runtime.airgap") is True


def test_apply_rejects_invalid_threshold_and_changes_nothing(clean_policy):
    # A present-but-invalid auto_approve must be a hard error, not a silent skip: skipping would
    # replace permissions.json while leaving whatever threshold the machine had (possibly a
    # gate-open --yolo residue) in force — a half-applied security posture.
    policy.set_risk_override("web_search", "read_only")
    before = policy.tier()
    with pytest.raises(ValueError):
        policy.apply_profile({"saturn_policy": 1, "auto_approve": "everything",
                              "risk_overrides": {"web_extract": "read_only"}})
    assert policy.tier() == before
    assert policy.risk_overrides() == {"web_search": "read_only"}  # durable file untouched


def test_apply_absent_threshold_is_fine(clean_policy):
    before = policy.tier()
    policy.apply_profile({"saturn_policy": 1, "shell_allow": ["git status"]})
    assert policy.tier() == before
    assert policy.shell_allow() == ["git status"]


def test_apply_rejects_invalid_redaction(clean_policy):
    with pytest.raises(ValueError):
        policy.apply_profile({"saturn_policy": 1, "redaction": "scramble"})
    assert get_config().get("runtime.redaction") == "off"


def test_apply_write_failure_leaves_live_knobs_untouched(clean_policy, monkeypatch):
    # permissions.json is written FIRST (the likeliest failure is disk I/O): if it fails, no live
    # knob may have moved — the docstring's "never half-applies" must hold under an OSError too.
    def boom(data):
        raise OSError("disk full")

    monkeypatch.setattr(policy, "_save", boom)
    before = policy.tier()
    with pytest.raises(OSError):
        policy.apply_profile({"saturn_policy": 1, "auto_approve": "destructive", "airgap": True})
    assert policy.tier() == before
    assert get_config().get("runtime.airgap") is False


def test_export_command_artifact_round_trips(clean_policy, tmp_path, monkeypatch):
    # The /policy export file (now also reachable via -o/--out/--output) must be a valid profile
    # end-to-end: written through the command path, applied through apply_policy_file.
    from core import llms
    import commands.policy  # noqa: F401 — registers /policy
    from commands._framework import CommandContext, dispatch
    from commands.policy import apply_policy_file

    monkeypatch.setattr(llms, "reset_models", lambda: None)
    policy.add_shell_allow("git status")
    dest = tmp_path / "exported.yaml"
    ctx = CommandContext(state={}, make_initial_state=dict, db_path="")
    dispatch(f"/policy export -o {dest}", ctx)
    assert dest.exists()

    policy.remove_shell_allow("git status")
    summary = apply_policy_file(str(dest))
    assert "policy applied" in summary
    assert policy.shell_allow() == ["git status"]


def test_apply_policy_file_drops_model_cache(clean_policy, tmp_path, monkeypatch):
    # A profile can flip runtime.airgap — apply_policy_file must drop the llms model cache
    # (mirroring /privacy airgap) or a cloud model cached while the boundary was open keeps
    # serving calls from llms._DERIVED_CACHE after the profile sealed it.
    from core import llms
    from commands.policy import apply_policy_file

    calls = []
    monkeypatch.setattr(llms, "reset_models", lambda: calls.append(1))
    prof = tmp_path / "prof.yaml"
    prof.write_text("saturn_policy: 1\nairgap: true\n", encoding="utf-8")
    apply_policy_file(str(prof))
    assert calls, "applying a profile must reset the model cache"

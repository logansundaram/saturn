"""
The /policy command namespace (commands/policy.py) — the levers (`risk`, `allow`, `open`) share
ONE handler with their legacy top-level spellings (/risk, /allow, /autoapprove — zero drift),
bare /autoapprove and bare /policy open are pure readouts (the gate-flip footgun is gone),
`/policy allow add` is the unambiguous add verb (removal verbs route to removal only when the
target resolves), the shared --save grammar is case-insensitive any-position, and `--save` with
no value persists the CURRENT value (mutating nothing live). Offline: no LLM, no network —
registry imports with mcp.servers empty.
"""

import pytest

from trust import policy

# Importing each module registers its command; the full `commands` package (every module) is
# deliberately NOT imported — these tests need only the policy-family commands.
# commands.policy registers the whole family, including the top-level /risk, /allow,
# /autoapprove spellings (one file owns every view of the gate policy).
import commands.policy  # noqa: F401
import commands.privacy  # noqa: F401
from commands._framework import CommandContext, dispatch


@pytest.fixture
def ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


@pytest.fixture
def gate(isolated_paths, monkeypatch):
    """Isolated permissions.json + session runtime knobs pinned to defaults and restored after."""
    from config import get_config

    cfg = get_config()
    runtime = cfg._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "auto_approve", "read_only")
    monkeypatch.setitem(runtime, "airgap", False)
    monkeypatch.setitem(runtime, "dry_run", False)
    monkeypatch.setattr(policy, "_tier_before_gate_off", None)
    return cfg


# ── parity: the legacy spelling and the /policy subcommand are the same handler ──────────────


def test_risk_parity_between_spellings(gate, ctx, capsys):
    from tools import registry

    declared = registry.risk_of("web_search")
    try:
        dispatch("/risk web_search destructive", ctx)
        legacy = capsys.readouterr().out
        assert registry.risk_of("web_search") == "destructive"

        dispatch("/policy risk web_search reset", ctx)
        capsys.readouterr()
        assert registry.risk_of("web_search") == declared

        dispatch("/policy risk web_search destructive", ctx)
        canonical = capsys.readouterr().out
        assert registry.risk_of("web_search") == "destructive"
        assert legacy == canonical  # one handler — identical observable effect AND text
        assert "failed:" not in legacy
    finally:
        registry.TOOL_RISK["web_search"] = declared
        policy.clear_risk_override("web_search")


def test_risk_save_persists_through_either_spelling(gate, ctx, capsys):
    from tools import registry

    declared = registry.risk_of("web_search")
    try:
        dispatch("/policy risk web_search side --save", ctx)
        assert policy.risk_overrides() == {"web_search": "side_effecting"}
        dispatch("/risk web_search reset", ctx)
        assert policy.risk_overrides() == {}
    finally:
        registry.TOOL_RISK["web_search"] = declared


def test_risk_save_flag_case_insensitive_any_position(gate, ctx, capsys):
    """The shared split_save_flag grammar: `--SAVE` counts, and the flag needn't trail the tier
    (the old parser scanned only args[2:], case-sensitively)."""
    from tools import registry

    declared = registry.risk_of("web_search")
    try:
        dispatch("/policy risk --SAVE web_search side", ctx)
        assert policy.risk_overrides() == {"web_search": "side_effecting"}
        assert "saved" in capsys.readouterr().out
        dispatch("/policy risk web_search reset", ctx)
        assert policy.risk_overrides() == {}
    finally:
        registry.TOOL_RISK["web_search"] = declared


def test_allow_metacharacter_prefix_gets_designed_refusal(gate, ctx, capsys):
    # add_shell_allow raises ValueError on a prefix the matcher could never honor — the command
    # renders its own refusal (with the why), never the dispatcher's generic '/policy failed'.
    dispatch("/policy allow echo hi > out.txt", ctx)
    out = capsys.readouterr().out
    assert "failed:" not in out
    assert "cannot allowlist" in out
    assert "metacharacter" in out
    assert policy.shell_allow() == []
    # Same refusal through the explicit add verb and the legacy spelling.
    dispatch("/policy allow add git status; rm -rf ~", ctx)
    assert "cannot allowlist" in capsys.readouterr().out
    dispatch("/allow git log | head", ctx)
    assert "cannot allowlist" in capsys.readouterr().out
    assert policy.shell_allow() == []


def test_allow_parity_between_spellings(gate, ctx, capsys):
    dispatch("/allow git status", ctx)
    legacy = capsys.readouterr().out
    assert policy.shell_allow() == ["git status"]

    dispatch("/policy allow remove git status", ctx)
    capsys.readouterr()
    assert policy.shell_allow() == []

    dispatch("/policy allow git status", ctx)
    canonical = capsys.readouterr().out
    assert policy.shell_allow() == ["git status"]
    assert legacy == canonical


def test_allow_remove_accepts_shared_verbs(gate, ctx, capsys):
    dispatch("/policy allow git status", ctx)
    dispatch("/policy allow rm 1", ctx)  # shared REMOVE_VERBS vocabulary, by index
    assert policy.shell_allow() == []
    dispatch("/allow ls -la", ctx)
    dispatch("/allow del ls -la", ctx)  # …and by exact text, through the legacy spelling
    assert policy.shell_allow() == []


def test_allow_add_verb_always_adds(gate, ctx, capsys):
    """`add` is the unambiguous escape hatch: it can allowlist a prefix whose first word is
    itself a removal verb (the only spelling that can)."""
    dispatch("/policy allow add del *.tmp", ctx)
    assert policy.shell_allow() == ["del *.tmp"]
    assert "allowed" in capsys.readouterr().out
    dispatch("/allow add git status", ctx)  # legacy spelling, ordinary prefix — same handler
    assert policy.shell_allow() == ["del *.tmp", "git status"]


def test_allow_remove_verb_with_unresolved_target_reports_never_guesses(gate, ctx, capsys):
    """'/policy allow del *.tmp' used to become a silent failed removal — with NO spelling able
    to allowlist such a prefix. Now an unresolved removal target reports and points at `add`."""
    dispatch("/policy allow git status", ctx)
    capsys.readouterr()
    dispatch("/policy allow del *.tmp", ctx)
    out = capsys.readouterr().out
    assert "no such allowlisted prefix" in out
    assert "/policy allow add del *.tmp" in out  # the disambiguating escape hatch
    assert policy.shell_allow() == ["git status"]  # nothing removed, nothing silently added


def test_allow_removal_verb_alone_points_at_add(gate, ctx, capsys):
    dispatch("/policy allow del", ctx)
    out = capsys.readouterr().out
    assert "usage" in out
    assert "/policy allow add del" in out  # how to allowlist the bare word itself
    assert policy.shell_allow() == []


@pytest.mark.parametrize("verb", ("list", "ls"))
def test_allow_lone_list_verb_lists_never_grants(gate, ctx, capsys, verb):
    """`/policy allow list` used to silently CREATE a gate exemption for the prefix `list` —
    a listing attempt becoming a security grant. A lone list verb is now the listing."""
    dispatch("/policy allow git status", ctx)
    capsys.readouterr()
    dispatch(f"/policy allow {verb}", ctx)
    out = capsys.readouterr().out
    assert "git status" in out  # the listing rendered
    assert policy.shell_allow() == ["git status"]  # nothing silently added

    dispatch(f"/allow {verb}", ctx)  # the legacy spelling, same handler
    assert policy.shell_allow() == ["git status"]


def test_allow_add_escape_hatch_for_lone_reserved_word(gate, ctx, capsys):
    """`add` allowlists a lone reserved word itself (`ls` is a real shell command)."""
    dispatch("/policy allow add ls", ctx)
    assert policy.shell_allow() == ["ls"]


def test_allow_list_verb_with_words_still_adds(gate, ctx, capsys):
    """A list verb FOLLOWED BY words stays an add — `ls -la` is a real command prefix, and
    listing never takes arguments, so the form disambiguates itself."""
    dispatch("/policy allow ls -la", ctx)
    assert policy.shell_allow() == ["ls -la"]


# ── /policy open · /autoapprove: bare = readout, mutation explicit ───────────────────────────


def test_bare_autoapprove_and_open_are_pure_readouts(gate, ctx, capsys):
    dispatch("/autoapprove", ctx)
    legacy = capsys.readouterr().out
    assert policy.tier() == "read_only"  # untouched — never a flip
    assert "prompting above" in legacy

    dispatch("/policy open", ctx)
    canonical = capsys.readouterr().out
    assert policy.tier() == "read_only"
    assert legacy == canonical


def test_open_explicit_round_trip(gate, ctx, capsys):
    policy.set_tier("side_effecting")
    dispatch("/policy open on", ctx)
    assert policy.gate_off()
    assert "AUTO-APPROVE ON" in capsys.readouterr().out
    dispatch("/policy open off", ctx)
    assert policy.tier() == "side_effecting"

    # The legacy spellings drive the exact same lever.
    dispatch("/yolo on", ctx)
    assert policy.gate_off()
    dispatch("/autoapprove off", ctx)
    assert policy.tier() == "side_effecting"


def test_open_unrecognized_arg_is_usage_not_flip(gate, ctx, capsys):
    dispatch("/policy open maybe", ctx)
    out = capsys.readouterr().out
    assert "usage" in out
    assert policy.tier() == "read_only"


def test_bare_open_reports_gate_off(gate, ctx, capsys):
    policy.set_gate_off(True)
    dispatch("/policy open", ctx)
    out = capsys.readouterr().out
    assert "gate OFF" in out
    assert policy.gate_off()  # a readout, even when open
    policy.set_gate_off(False)


# ── bare /policy posture: quarantine row + allowlist count ───────────────────────────────────


def test_bare_policy_posture_gains_quarantine_and_count(gate, ctx, capsys):
    policy.add_shell_allow("git status")
    dispatch("/policy", ctx)
    out = capsys.readouterr().out
    assert "quarantine" in out
    assert "1 prefix(es)" in out and "git status" in out


# ── /privacy hygiene: bare --save persists the CURRENT value ─────────────────────────────────


def test_airgap_save_without_value_persists_current(gate, ctx, capsys, monkeypatch):
    """`--save` with no on|off persists the CURRENT value (the shared convention) — it mutates
    nothing live, so the seal can never silently flip."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/privacy airgap --save", ctx)
    out = capsys.readouterr().out
    assert "no change" in out
    assert gate.get("runtime.airgap") is False  # NOT toggled
    assert persisted == ["runtime.airgap"]  # the current value, persisted


def test_airgap_on_with_save_sets_then_persists(gate, ctx, capsys, monkeypatch):
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/privacy airgap on --save", ctx)
    capsys.readouterr()
    assert gate.get("runtime.airgap") is True
    assert persisted == ["runtime.airgap"]


def test_airgap_bare_save_token_is_no_longer_a_save_flag(gate, ctx, capsys, monkeypatch):
    """Only --save/-s count (split_save_flag): the old bare-'save' token form is gone, so
    `/privacy airgap save` is an unrecognized toggle -> usage, no flip, no persist."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/privacy airgap save", ctx)
    out = capsys.readouterr().out
    assert "usage" in out
    assert gate.get("runtime.airgap") is False
    assert persisted == []


def test_redact_save_without_mode_persists_current(gate, ctx, capsys, monkeypatch):
    import config as config_mod

    runtime = gate._data["runtime"]
    monkeypatch.setitem(runtime, "redaction", "off")
    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/privacy redact --save", ctx)
    out = capsys.readouterr().out
    assert "no change" in out
    assert gate.get("runtime.redaction") == "off"  # NOT changed
    assert persisted == ["runtime.redaction"]  # the current mode, persisted


# ── /dryrun: bare = status, explicit on|off mutates ──────────────────────────────────────────


def test_dryrun_bare_is_status_explicit_mutates(gate, ctx, capsys):
    dispatch("/dryrun", ctx)
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert gate.get("runtime.dry_run") is False  # untouched

    dispatch("/dryrun on", ctx)
    assert gate.get("runtime.dry_run") is True
    assert "DRY-RUN ON" in capsys.readouterr().out  # the banner survives on actual change
    dispatch("/dryrun off", ctx)
    assert gate.get("runtime.dry_run") is False


def test_dryrun_unrecognized_arg_is_usage_not_flip(gate, ctx, capsys):
    dispatch("/dryrun maybe", ctx)
    out = capsys.readouterr().out
    assert "usage" in out
    assert gate.get("runtime.dry_run") is False  # untouched (parse_toggle_status -> "invalid")

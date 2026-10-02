"""
The /policy command namespace (commands/policy.py) — the one trust front door: the gate's levers
(`risk`, `allow`, `open`) and the network boundary (`egress`, `airgap`; the /privacy spelling is
a pointer that runs nothing). Retired spellings mutate NOTHING, bare /policy open is a pure
readout, `/policy allow add` is the unambiguous add verb
(removal verbs route to removal only when the target resolves), the shared --save grammar is
case-insensitive any-position, and `--save` with no value persists the CURRENT value (mutating
nothing live). Offline: no LLM, no network — registry imports with mcp.servers empty.
"""

import pytest

from trust import policy

from commands._framework import dispatch


@pytest.fixture
def gate(isolated_paths, monkeypatch):
    """Isolated permissions.json + session runtime knobs pinned to defaults and restored after."""
    from config import get_config

    cfg = get_config()
    runtime = cfg._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "auto_approve", "read_only")
    monkeypatch.setitem(runtime, "airgap", False)
    monkeypatch.setattr(policy, "_tier_before_gate_off", None)
    return cfg


# ── the cut legacy spellings: pointers that mutate NOTHING ───────────────────────────────────


def test_legacy_gate_spellings_change_nothing(gate, ctx, capsys):
    """/risk, /allow, /autoapprove, /yolo were CUT 2026-07-06 (their pointers dropped
    2026-09-30) — each is an unknown command and the policy object is untouched: a habit-typed
    `/yolo on` can never open the gate, and `/allow <prefix>` can never create an exemption."""
    from tools import registry

    declared = registry.risk_of("web_search")
    for line in ("/risk web_search destructive", "/allow git status", "/autoapprove on",
                 "/yolo on"):
        dispatch(line, ctx)
        assert "unknown command" in capsys.readouterr().out, line
    assert registry.risk_of("web_search") == declared  # no tier changed
    assert policy.risk_overrides() == {}               # nothing persisted
    assert policy.shell_allow() == []                  # no exemption created
    assert policy.tier() == "read_only"                # the gate did not open
    assert not policy.gate_off()


# ── /policy risk ──────────────────────────────────────────────────────────────────────────────


def test_risk_set_and_reset(gate, ctx, capsys):
    from tools import registry

    declared = registry.risk_of("web_search")
    try:
        dispatch("/policy risk web_search destructive", ctx)
        out = capsys.readouterr().out
        assert registry.risk_of("web_search") == "destructive"
        assert "failed:" not in out

        dispatch("/policy risk web_search reset", ctx)
        capsys.readouterr()
        assert registry.risk_of("web_search") == declared
    finally:
        registry.TOOL_RISK["web_search"] = declared
        policy.clear_risk_override("web_search")


def test_risk_save_persists_and_reset_forgets(gate, ctx, capsys):
    from tools import registry

    declared = registry.risk_of("web_search")
    try:
        dispatch("/policy risk web_search side --save", ctx)
        assert policy.risk_overrides() == {"web_search": "side_effecting"}
        dispatch("/policy risk web_search reset", ctx)
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
    # Same refusal through the explicit add verb and the pipe metacharacter.
    dispatch("/policy allow add git status; rm -rf ~", ctx)
    assert "cannot allowlist" in capsys.readouterr().out
    dispatch("/policy allow git log | head", ctx)
    assert "cannot allowlist" in capsys.readouterr().out
    assert policy.shell_allow() == []


def test_allow_add_and_remove_round_trip(gate, ctx, capsys):
    dispatch("/policy allow git status", ctx)
    out = capsys.readouterr().out
    assert policy.shell_allow() == ["git status"]
    assert "allowed" in out

    dispatch("/policy allow remove git status", ctx)
    capsys.readouterr()
    assert policy.shell_allow() == []


def test_allow_remove_accepts_shared_verbs(gate, ctx, capsys):
    dispatch("/policy allow git status", ctx)
    dispatch("/policy allow rm 1", ctx)  # shared REMOVE_VERBS vocabulary, by index
    assert policy.shell_allow() == []
    dispatch("/policy allow ls -la", ctx)
    dispatch("/policy allow del ls -la", ctx)  # …and by exact text
    assert policy.shell_allow() == []


def test_allow_add_verb_always_adds(gate, ctx, capsys):
    """`add` is the unambiguous escape hatch: it can allowlist a prefix whose first word is
    itself a removal verb (the only spelling that can)."""
    dispatch("/policy allow add del *.tmp", ctx)
    assert policy.shell_allow() == ["del *.tmp"]
    assert "allowed" in capsys.readouterr().out
    dispatch("/policy allow add git status", ctx)  # ordinary prefix through the same verb
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


def test_allow_add_escape_hatch_for_lone_reserved_word(gate, ctx, capsys):
    """`add` allowlists a lone reserved word itself (`ls` is a real shell command)."""
    dispatch("/policy allow add ls", ctx)
    assert policy.shell_allow() == ["ls"]


def test_allow_list_verb_with_words_still_adds(gate, ctx, capsys):
    """A list verb FOLLOWED BY words stays an add — `ls -la` is a real command prefix, and
    listing never takes arguments, so the form disambiguates itself."""
    dispatch("/policy allow ls -la", ctx)
    assert policy.shell_allow() == ["ls -la"]


# ── /policy open: bare = readout, mutation explicit ──────────────────────────────────────────


def test_bare_open_is_a_pure_readout(gate, ctx, capsys):
    dispatch("/policy open", ctx)
    out = capsys.readouterr().out
    assert policy.tier() == "read_only"  # untouched — never a flip
    assert "prompting above" in out


def test_open_explicit_round_trip(gate, ctx, capsys):
    policy.set_tier("side_effecting")
    dispatch("/policy open on", ctx)
    assert policy.gate_off()
    assert "AUTO-APPROVE ON" in capsys.readouterr().out
    dispatch("/policy open off", ctx)
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


# ── /privacy merged into /policy (2026-09-30): one trust front door ──────────────────────────


def test_bare_policy_is_the_one_posture_readout(gate, ctx, capsys, monkeypatch):
    """Bare /policy carries what the two bare readouts used to split: the gate AND the boundary
    — where the words are computed, what the web tools send, the air-gap, the quarantine, and
    where the data lives — each facet stated once."""
    from trust import egress

    monkeypatch.setattr(egress, "_inference", lambda: {
        "bindings": [{"role": "chat", "model": "qwen3.5:9b", "locality": "local"},
                     {"role": "embedder", "model": "qwen3-embedding:8b", "locality": "local"}],
        "all_local": True})
    dispatch("/policy", ctx)
    out = capsys.readouterr().out
    assert "defaults hold" in out                        # calm by default
    assert "asks before" in out and "side_effecting and destructive" in out
    assert "local — qwen3.5:9b" in out and "qwen3-embedding:8b" in out
    assert "DuckDuckGo" in out
    assert "your data" in out
    assert "telemetry" in out
    assert out.count("air-gap") == 1                    # deduped: one airgap row
    assert out.count("quarantine") == 1                 # deduped: one quarantine row
    assert "/policy egress" in out                      # the ledger is named from here


def test_bare_policy_is_loud_on_deviation(gate, ctx, capsys, monkeypatch):
    """A remote OLLAMA_HOST, an open gate, and the seal each speak — in the header (the startup
    posture line's own words) and in their row."""
    from trust import egress

    monkeypatch.setattr(egress, "_inference", lambda: {
        "bindings": [{"role": "chat", "model": "qwen3.5:9b", "locality": "remote"}],
        "all_local": False, "remote_ollama": "http://10.0.0.5:11434"})
    policy.set_gate_off(True)
    gate.set("runtime.airgap", True)
    dispatch("/policy", ctx)
    out = capsys.readouterr().out
    assert "defaults hold" not in out
    assert "GATE OFF" in out
    assert "inference off-machine: ollama @ http://10.0.0.5:11434" in out
    assert "BLOCKED by air-gap — ollama @ http://10.0.0.5:11434" in out
    assert "sealed by air-gap" in out  # the web tools row


def test_privacy_is_a_pointer_that_changes_nothing(gate, ctx, capsys, monkeypatch):
    """/privacy (any spelling, any args) prints the moved-pointer to /policy and runs NOTHING —
    `/privacy airgap on --save` must never flip or persist the seal, and `/privacy egress clear`
    must never wipe the ledger."""
    import config as config_mod
    from trust import egress

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    egress.record("web_search", "duckduckgo.com", "q", n_bytes=10)
    before = len(egress.events())
    assert before >= 1
    for line in ("/privacy", "/privacy egress", "/privacy egress clear",
                 "/privacy airgap on --save"):
        dispatch(line, ctx)
        out = capsys.readouterr().out
        assert "/privacy moved — use /policy" in out, line
        assert "/policy egress" in out and "/policy airgap" in out, line
    assert gate.get("runtime.airgap") is False
    assert persisted == []
    assert len(egress.events()) == before


def test_policy_egress_lists_the_ledger_and_clears(gate, ctx, capsys, monkeypatch):
    from trust import egress

    egress.clear()
    monkeypatch.setattr(egress, "_CLEARED_AT", 0)  # a FRESH session, not an operator clear
    egress.record("web_search", "duckduckgo.com", "query", n_bytes=321)
    egress.record("web_extract", "example.com", "page", status=egress.BLOCKED)
    dispatch("/policy egress", ctx)
    out = capsys.readouterr().out
    assert "1 egress event(s)" in out and "1 blocked" in out
    assert "duckduckgo.com" in out and "example.com" in out and "BLOCKED" in out
    dispatch("/policy ledger 1", ctx)  # the alias + the tail limit
    out = capsys.readouterr().out
    assert "last 1 of 2" in out and "example.com" in out and "duckduckgo.com" not in out
    dispatch("/policy egress clear", ctx)
    assert "cleared" in capsys.readouterr().out
    assert egress.events() == []
    dispatch("/policy egress", ctx)
    assert "cleared" in capsys.readouterr().out  # a cleared ledger never reads as "nothing left"


def test_policy_egress_never_claims_closed_over_an_untracked_run(gate, ctx, capsys, monkeypatch):
    from trust import egress

    egress.clear()
    monkeypatch.setattr(egress, "_CLEARED_AT", 0)
    egress.record("shell", "?", "git pull", status=egress.UNTRACKED)
    dispatch("/policy egress", ctx)
    out = capsys.readouterr().out
    assert "nothing has left" not in out and "stayed closed" not in out
    assert "1 untracked" in out and "UNTRACKED" in out and "git pull" in out
    assert "cannot see" in out  # the row's meaning is said, not left to guess


def test_policy_airgap_bare_is_status_only(gate, ctx, capsys, monkeypatch):
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy airgap", ctx)
    out = capsys.readouterr().out
    assert "air-gap" in out and "/policy airgap on" in out
    assert gate.get("runtime.airgap") is False
    assert persisted == []


def test_policy_airgap_on_without_save_is_session_only(gate, ctx, capsys, monkeypatch):
    """Trust settings never persist silently (config.TRUST_KEYS): a plain `on` flips the live
    seal and writes nothing."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy airgap on", ctx)
    out = capsys.readouterr().out
    assert "AIR-GAP ON" in out and "/policy airgap off" in out
    assert gate.get("runtime.airgap") is True
    assert persisted == []
    dispatch("/policy air-gap off", ctx)  # the alias
    assert gate.get("runtime.airgap") is False


# ── /policy airgap hygiene: bare --save persists the CURRENT value ───────────────────────────


def test_airgap_save_without_value_persists_current(gate, ctx, capsys, monkeypatch):
    """`--save` with no on|off persists the CURRENT value (the shared convention) — it mutates
    nothing live, so the seal can never silently flip."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy airgap --save", ctx)
    out = capsys.readouterr().out
    assert "no change" in out
    assert gate.get("runtime.airgap") is False  # NOT toggled
    assert persisted == ["runtime.airgap"]  # the current value, persisted


def test_airgap_on_with_save_sets_then_persists(gate, ctx, capsys, monkeypatch):
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy airgap on --save", ctx)
    capsys.readouterr()
    assert gate.get("runtime.airgap") is True
    assert persisted == ["runtime.airgap"]


def test_airgap_bare_save_token_is_no_longer_a_save_flag(gate, ctx, capsys, monkeypatch):
    """Only --save/-s count (split_save_flag): the old bare-'save' token form is gone, so
    `/policy airgap save` is an unrecognized toggle -> usage, no flip, no persist."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy airgap save", ctx)
    out = capsys.readouterr().out
    assert "usage" in out
    assert gate.get("runtime.airgap") is False
    assert persisted == []


def test_redact_subcommand_is_cut(gate, ctx, capsys, monkeypatch):
    """/privacy redact was CUT 2026-07-16 (and runtime.redaction behind it 2026-09-29): under
    the merged front door it is a plain unknown-subcommand error and mutates nothing."""
    import config as config_mod

    persisted = []
    monkeypatch.setattr(config_mod, "persist", lambda key: persisted.append(key))
    dispatch("/policy redact warn", ctx)
    out = capsys.readouterr().out
    assert "unknown /policy subcommand" in out
    assert persisted == []  # nothing written


def test_redact_legacy_spelling_is_plain_unknown(gate, ctx, capsys, monkeypatch):
    """The old top-level /redact pointer went with the cut — a cut feature leaves no pointer."""
    dispatch("/redact warn", ctx)
    out = capsys.readouterr().out
    assert "unknown command" in out


# ── /dryrun: CUT 2026-07-03 — the old spelling lands on a moved-pointer, never a flip ────────


def test_dryrun_is_not_a_command(gate, ctx, capsys):
    dispatch("/dryrun on", ctx)
    assert "unknown command" in capsys.readouterr().out
    assert gate.get("runtime.dry_run") is None  # the knob no longer exists, nothing was set


# ── grant lifetimes at the command surface (transplanted from the gating isolate) ────────────


def test_allow_command_persists_and_lists_scopes(gate, ctx, capsys):
    """`/policy allow <prefix>` IS the durable allowlist edit — persist scope regardless of the
    gate's default lifetime; the listing names each prefix's lifetime and removal reaches every
    scope by index over the effective list."""
    dispatch("/policy allow git status", ctx)
    capsys.readouterr()
    assert policy.persisted_shell_allow() == ["git status"]
    policy.add_shell_allow("git log")  # a task-scoped gate grant, for contrast
    dispatch("/policy allow", ctx)
    out = capsys.readouterr().out
    assert "git status" in out and "(persisted)" in out
    assert "git log" in out and "(this turn)" in out
    dispatch("/policy allow remove 1", ctx)  # index 1 = the task-scoped one (expiry order)
    capsys.readouterr()
    assert policy.shell_allow() == ["git status"]


def test_bare_policy_readout_names_the_grant_lifetime(gate, ctx, capsys):
    dispatch("/policy", ctx)
    out = capsys.readouterr().out
    assert "always-allow" in out and "this turn" in out and "grant_scope: task" in out


def test_airgap_offmachine_warning_names_the_remote_models(gate, ctx, capsys, monkeypatch):
    """Sealing the boundary while a model runs behind a remote OLLAMA_HOST says which models will
    fail and how to run local."""
    import config as config_mod
    from commands import policy as policy_cmd

    monkeypatch.setattr(config_mod, "persist", lambda key: key)
    monkeypatch.setattr(policy_cmd, "_offmachine_models",
                        lambda: [("chat", "ollama @ http://10.0.0.5:11434", "qwen3.5:9b")])
    dispatch("/policy airgap on", ctx)
    out = capsys.readouterr().out

    assert "will now FAIL" in out and "chat qwen3.5:9b (ollama @ http://10.0.0.5:11434)" in out
    assert "OLLAMA_HOST" in out


def test_risk_never_lowers_a_no_blanket_grant_tool(gate, ctx, capsys):
    """`/policy risk run_shell read_only --save` would un-gate every shell command with no
    allowlist — the one thing CLAUDE.md says never happens (run_shell is always destructive).
    Refused at the front door, nothing persisted; `reset` still works."""
    from tools import registry

    for name in ("run_shell", "run_shortcut", "send_message"):
        declared = registry.risk_of(name)
        dispatch(f"/policy risk {name} read_only --save", ctx)
        out = capsys.readouterr().out
        assert registry.risk_of(name) == declared
        assert policy.risk_overrides() == {}
        assert "always" in out and "->" not in out
    dispatch("/policy risk run_shell reset", ctx)
    assert registry.risk_of("run_shell") == "destructive"


def test_policy_open_off_on_a_closed_gate_changes_nothing(gate, ctx, capsys):
    """`/policy open off` typed to CONFIRM the gate is closed must not drop a configured
    side_effecting threshold to read_only (and must not claim it "restored" anything)."""
    prev = policy.tier()
    try:
        policy.set_tier("side_effecting")
        dispatch("/policy open off", ctx)
        out = capsys.readouterr().out
        assert policy.tier() == "side_effecting"
        assert "restored" not in out and "side_effecting" in out
    finally:
        policy.set_tier(prev)
        policy._tier_before_gate_off = None

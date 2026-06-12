"""
Gate-UX pure logic (tui/ui/approval.py + plan.py): the first-gate preamble (receipt.take_hint —
once per install, failing safe to once per session), the full-width-argument branch selection +
per-value head/tail clamp (textutil.head_tail — its unit tests live here too, no textutil suite
exists), the byte-faithful hard wrap (`_wrap_exact` — no whitespace mutation in the very
arguments the human approves), the scoped run_shell always-allow (full-command proposal + grant
validated through the one policy matcher — never a second mechanism), and the plan-review
editor's fail-closed Ctrl-C/EOF handling. Presentation (rich frames, prompt strings) is not
asserted — only the decisions underneath.
"""

import types

from trust import policy
from trust import receipt
from textutil import head_tail
from tui.ui import approval
from tui.ui import plan as plan_ui
from tui.ui._base import _RAIL_GLYPH


# ── first-gate teaching preamble: once per install via receipt.take_hint ─────────────────────


def test_preamble_once_per_install(isolated_paths, monkeypatch):
    monkeypatch.setattr(receipt, "_HINTS_SHOWN", set())
    assert approval._preamble_due() is True
    # Marked for the session…
    assert approval._preamble_due() is False
    # …and for the install: a "new session" (hint set reset) still finds the sentinel.
    monkeypatch.setattr(receipt, "_HINTS_SHOWN", set())
    assert approval._preamble_due() is False

    from config import get_config

    # The sentinel is receipt.take_hint's — one mechanism, not a parallel .gate_seen file.
    assert (get_config().path("database") / ".hint_gate_seen").exists()
    assert not (get_config().path("database") / ".gate_seen").exists()


def test_preamble_fails_safe_when_sentinel_unwritable(isolated_paths, monkeypatch):
    # Make the database path a FILE so the sentinel write must fail — the preamble still shows,
    # but at most once per session (take_hint's in-memory guard), and again next "session".
    from config import get_config

    db_dir = get_config().path("database")
    db_dir.parent.mkdir(parents=True, exist_ok=True)
    db_dir.write_text("not a directory", encoding="utf-8")

    monkeypatch.setattr(receipt, "_HINTS_SHOWN", set())
    assert approval._preamble_due() is True
    assert approval._preamble_due() is False  # same session: never twice
    monkeypatch.setattr(receipt, "_HINTS_SHOWN", set())
    assert approval._preamble_due() is True  # install-level mark could not persist


# ── full-width argument rendering: branch selection + per-value clamp ─────────────────────────


def test_full_width_args_branch():
    # No bespoke renderer + gated tier → full surface. All mcp_* tools land here (they fail
    # closed to destructive precisely because they're untrusted).
    assert approval._full_width_args("mcp_github_create_issue", "destructive")
    assert approval._full_width_args("some_new_tool", "side_effecting")
    # The four bespoke-rendered tools keep their dedicated views.
    for name in ("write_file", "edit_file", "run_shell", "http_request"):
        assert not approval._full_width_args(name, "destructive")
    # read_only calls (gated only via a quarantine/taint escalation) keep the compact repr.
    assert not approval._full_width_args("web_search", "read_only")
    assert not approval._full_width_args("mcp_x_read", "read_only")


def test_clamp_value_head_and_tail():
    short = "x" * approval._MAX_ARG_VALUE
    assert approval._clamp_value(short) == short

    long = "H" * 1500 + "M" * 1500 + "T" * 1500
    out = approval._clamp_value(long)  # cap 2000 → 2500 dropped
    assert out.startswith("H")  # head kept
    assert out.endswith("T")  # tail kept — where a long payload hides the part that matters
    assert "truncated 2500 characters" in out
    # Delegation, not a third hand-rolled copy: byte-identical to the textutil primitive.
    assert out == head_tail(long, approval._MAX_ARG_VALUE)


# ── textutil.head_tail: THE head+tail elision primitive (no textutil suite — tested here) ─────


def test_head_tail_passthrough_is_same_object():
    s = "y" * 100
    assert head_tail(s, 100) is s  # under-cap text passes through untouched (identity holds)


def test_head_tail_default_marker_split():
    long = "H" * 1500 + "M" * 1500 + "T" * 1500
    out = head_tail(long, 2000)  # head 2/3 = 1333, tail 1/3 = 667, dropped 2500
    assert out.startswith("H" * 1333)
    assert out.endswith("T" * 667)
    assert "\n… [truncated 2500 characters] …\n" in out


def test_head_tail_custom_marker():
    long = "a" * 50
    out = head_tail(long, 30, marker="<cut {dropped}>")
    assert out == "a" * 20 + "<cut 20>" + "a" * 10


def test_head_tail_reproduces_observation_clamp_exactly():
    # nodes/tools._clamp_observation now rides head_tail; its historical marker text is
    # part of what the model (and the clamp tests) read, so it must reproduce byte-exactly.
    from nodes.tools import _clamp_observation, _MAX_OBSERVATION

    s = "a" * (_MAX_OBSERVATION + 5000)
    head = _MAX_OBSERVATION * 2 // 3
    tail = _MAX_OBSERVATION - head
    expected = (
        s[:head]
        + "\n\n... [truncated 5000 characters of tool output] ...\n\n"
        + s[-tail:]
    )
    assert _clamp_observation(s) == expected


# ── byte-faithful hard wrap: the safety surface is never whitespace-rewritten ────────────────


def test_wrap_exact_byte_faithful_modulo_line_breaks():
    # Tabs, space runs, and trailing spaces survive exactly — rejoining the chunks reproduces
    # the input. textwrap.wrap would expand the tab and collapse the runs: the bug.
    line = "cmd\targ1   arg2    end  "
    for width in (4, 7, 200):
        assert "".join(approval._wrap_exact(line, width)) == line


def test_wrap_exact_preserves_indentation():
    # Continuation indentation is part of the approved bytes (think YAML/Python in an argument).
    line = "        indented = True"
    assert approval._wrap_exact(line, 100) == [line]
    assert "".join(approval._wrap_exact(line, 6)) == line


def test_wrap_exact_whitespace_only_and_empty_lines_render_as_themselves():
    assert approval._wrap_exact("   ", 80) == ["   "]  # not dropped to nothing
    assert approval._wrap_exact("\t", 80) == ["\t"]
    assert approval._wrap_exact("", 80) == [""]  # an empty line is still a row


def test_wrap_exact_chunk_widths():
    assert approval._wrap_exact("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


# ── scoped run_shell always-allow: proposal + grant via the one policy matcher ───────────────


def test_propose_shell_prefix():
    # The FULL command — the narrowest grant covering exactly what was reviewed. The bare
    # leading token would un-gate `git push --force` from one confirmation, the broad grant
    # /allow's own help warns against (a shorter prefix only by the user typing it).
    assert approval._propose_shell_prefix("git status --short") == "git status --short"
    assert approval._propose_shell_prefix("git  status") == "git status"  # whitespace-normalized
    assert approval._propose_shell_prefix("") is None
    assert approval._propose_shell_prefix("   ") is None
    # No prefix can ever exempt a chained/redirected command — policy refuses metacharacters
    # wholesale, so proposing one would teach a false rule.
    assert approval._propose_shell_prefix("git status; rm -rf ~") is None
    assert approval._propose_shell_prefix("git log | head") is None


def test_grant_shell_prefix_valid(isolated_paths):
    granted, _msg = approval._grant_shell_prefix("git", "git status")
    assert granted
    assert policy.shell_allowed("git status --short") == "git"


def test_grant_shell_prefix_rejects_non_prefix(isolated_paths):
    granted, _msg = approval._grant_shell_prefix("curl", "git status")
    assert not granted
    assert policy.shell_allow() == []  # rolled back, never stored


def test_grant_shell_prefix_rejects_metacharacters(isolated_paths):
    # The rejection flows through policy.shell_allowed itself — the one matcher.
    granted, _msg = approval._grant_shell_prefix("git;", "git status")
    assert not granted
    assert policy.shell_allow() == []
    # A chained COMMAND can never be exempted either, even by its honest leading token.
    granted, _msg = approval._grant_shell_prefix("git", "git status; rm -rf ~")
    assert not granted
    assert policy.shell_allow() == []


def test_grant_shell_prefix_token_boundary(isolated_paths):
    granted, _msg = approval._grant_shell_prefix("git status", "git statusx")
    assert not granted
    assert policy.shell_allow() == []


def test_grant_shell_prefix_already_covered(isolated_paths):
    policy.add_shell_allow("git")
    granted, _msg = approval._grant_shell_prefix("git status", "git status --short")
    assert granted  # the command IS exempt going forward…
    assert policy.shell_allow() == ["git"]  # …but no redundant entry stacked


def test_grant_shell_prefix_empty(isolated_paths):
    granted, _msg = approval._grant_shell_prefix("   ", "git status")
    assert not granted
    assert policy.shell_allow() == []


def test_grant_shell_prefix_keeps_preexisting_entry_on_failed_grant(isolated_paths):
    policy.add_shell_allow("curl")
    granted, _msg = approval._grant_shell_prefix("curl", "git status")
    assert not granted
    assert policy.shell_allow() == ["curl"]  # not ours — never rolled back


def test_always_allow_scopes_run_shell(isolated_paths, monkeypatch):
    fake_registry = types.SimpleNamespace(TOOL_RISK={})
    # `_always_allow` resolves the registry lazily via `from tools import registry`, which binds
    # the package attribute — patch that attribute, not a bare sys.modules name.
    monkeypatch.setattr("tools.registry", fake_registry, raising=False)
    calls = [
        {"id": "1", "name": "run_shell", "risk": "destructive",
         "args": {"command": "git status"}},
        {"id": "2", "name": "mcp_x_do", "risk": "destructive", "args": {}},
    ]
    answers = iter(["y"])
    approval._always_allow(calls, lambda _prompt: next(answers))
    # The non-shell tool drops to read_only for the session; run_shell does NOT.
    assert fake_registry.TOOL_RISK == {"mcp_x_do": "read_only"}
    # Instead the FULL command landed in the /allow store, via the one policy path.
    assert policy.shell_allow() == ["git status"]
    assert policy.shell_allowed("git status") == "git status"


def test_always_allow_default_grant_does_not_cover_siblings(isolated_paths, monkeypatch):
    # One `y` at the gate must never widen beyond the reviewed command: the default proposal is
    # the full command, so `git push --force` (and even shorter siblings) still face the gate.
    fake_registry = types.SimpleNamespace(TOOL_RISK={})
    # `_always_allow` resolves the registry lazily via `from tools import registry`, which binds
    # the package attribute — patch that attribute, not a bare sys.modules name.
    monkeypatch.setattr("tools.registry", fake_registry, raising=False)
    calls = [{"id": "1", "name": "run_shell", "risk": "destructive",
              "args": {"command": "git status --short"}}]
    answers = iter(["y"])
    approval._always_allow(calls, lambda _prompt: next(answers))
    assert policy.shell_allow() == ["git status --short"]
    assert policy.shell_allowed("git status --short") == "git status --short"
    assert policy.shell_allowed("git push --force") is None
    assert policy.shell_allowed("git status") is None  # a prefix never matches a SHORTER command


def test_always_allow_typed_shorter_prefix(isolated_paths, monkeypatch):
    # Broadening below the full command stays available — but only by TYPING the prefix.
    fake_registry = types.SimpleNamespace(TOOL_RISK={})
    # `_always_allow` resolves the registry lazily via `from tools import registry`, which binds
    # the package attribute — patch that attribute, not a bare sys.modules name.
    monkeypatch.setattr("tools.registry", fake_registry, raising=False)
    calls = [{"id": "1", "name": "run_shell", "risk": "destructive",
              "args": {"command": "git status --short"}}]
    answers = iter(["git status"])
    approval._always_allow(calls, lambda _prompt: next(answers))
    assert policy.shell_allow() == ["git status"]
    assert policy.shell_allowed("git status --porcelain") == "git status"
    assert policy.shell_allowed("git push") is None  # still narrower than the bare token
    assert "run_shell" not in fake_registry.TOOL_RISK


def test_always_allow_enter_declines_shell_grant(isolated_paths, monkeypatch):
    fake_registry = types.SimpleNamespace(TOOL_RISK={})
    # `_always_allow` resolves the registry lazily via `from tools import registry`, which binds
    # the package attribute — patch that attribute, not a bare sys.modules name.
    monkeypatch.setattr("tools.registry", fake_registry, raising=False)
    calls = [{"id": "1", "name": "run_shell", "risk": "destructive",
              "args": {"command": "git status"}}]
    answers = iter([""])  # bare Enter = no, like every gate read
    approval._always_allow(calls, lambda _prompt: next(answers))
    assert policy.shell_allow() == []
    assert "run_shell" not in fake_registry.TOOL_RISK


# ── decision semantics: the fail-closed default is unchanged, only now signaled ──────────────


def test_resolve_decision_default_rejects():
    assert approval._resolve_decision("", [], lambda _p: "") is False
    assert approval._resolve_decision("zzz", [], lambda _p: "") is False
    assert approval._resolve_decision("y", [], lambda _p: "") is True


def test_select_calls_enter_defaults_to_no():
    calls = [{"id": "a", "name": "t1"}, {"id": "b", "name": "t2"}]
    answers = iter(["", "y"])
    out = approval._select_calls(calls, lambda _p: next(answers))
    assert out == {"approved_ids": ["b"]}


# ── plan review: bare plan rows inside the frame + fail-closed Ctrl-C/EOF ────────────────────


def test_plan_line_bare_drops_the_rail():
    step = {"step_id": 1, "label": "find sources", "status": "pending",
            "intended_tool": "web_search"}
    bare = plan_ui._plan_line_bare(step, show_tool=True)
    railed = plan_ui._plan_line(step, show_tool=True)
    bare_s = bare.plain if hasattr(bare, "plain") else str(bare)
    railed_s = railed.plain if hasattr(railed, "plain") else str(railed)
    assert _RAIL_GLYPH not in bare_s
    assert _RAIL_GLYPH in railed_s
    assert bare_s in railed_s  # the railed form is exactly rail + bare


def _quiet_review(monkeypatch):
    monkeypatch.setattr(plan_ui, "_live_stop", lambda: None)
    monkeypatch.setattr(plan_ui, "_live_start", lambda: None)


def test_review_plan_ctrl_c_aborts(monkeypatch):
    _quiet_review(monkeypatch)

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(plan_ui, "_review_input", interrupt)
    out = plan_ui.review_plan({"plan": [], "reason": "", "active_step": None})
    assert out["action"] == "abort"


def test_review_plan_eof_aborts(monkeypatch):
    _quiet_review(monkeypatch)

    def eof():
        raise EOFError

    monkeypatch.setattr(plan_ui, "_review_input", eof)
    out = plan_ui.review_plan({"plan": [], "reason": "", "active_step": None})
    assert out["action"] == "abort"


def test_review_plan_enter_continues(monkeypatch):
    _quiet_review(monkeypatch)
    monkeypatch.setattr(plan_ui, "_review_input", lambda: "")
    out = plan_ui.review_plan({"plan": [], "reason": "", "active_step": None})
    assert out["action"] == "continue"

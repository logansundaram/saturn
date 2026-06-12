"""
The /help redesign + the dispatcher help grammar (commands/_framework + commands/help).

Covers: the static grouping table exactly matching the live registry (a future command can't
silently vanish from /help), the dead scaffold legend staying dead, the _RENAMED fallback
(`/help why` prints the same moved-pointer as `/why`), the cut /commands command landing on a
pointer, and the standalone --help/-h token showing help ONLY at the first or final argument
position (mid-position is data — `/memory add prefer -h over --help in docs` must execute).
"""

import commands  # noqa: F401 — triggers registration of every built-in module (two-phase load)
from commands._framework import (
    COMMANDS,
    CommandContext,
    SlashCommand,
    dispatch,
)
from commands.system import _GATE_VIEWS, _GROUPS
from commands.user_commands import registered_names


def _ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


# --- the grouping table ------------------------------------------------------------------------

def test_grouping_table_exactly_covers_live_registry():
    grouped = [n for _group, names in _GROUPS for n in names]
    assert len(grouped) == len(set(grouped)), "a command appears in two /help groups"
    # Built-ins are whatever is registered that the user-template loader didn't put there.
    builtins = set(COMMANDS) - registered_names()
    assert set(grouped) == builtins, (
        f"/help grouping table drifted from the registry — "
        f"missing: {sorted(builtins - set(grouped))}, stale: {sorted(set(grouped) - builtins)}"
    )


def test_groups_are_alphabetical_and_bounded():
    assert len(_GROUPS) <= 6
    for _group, names in _GROUPS:
        assert list(names) == sorted(names)


def test_gate_views_live_in_a_group():
    grouped = {n for _group, names in _GROUPS for n in names}
    for v in _GATE_VIEWS:
        assert v in grouped  # coverage counts them even though they render as one compact line


# --- the rendered listing ----------------------------------------------------------------------

def test_help_renders_groups_map_and_no_dead_legend(capsys):
    dispatch("/help", _ctx())
    out = capsys.readouterr().out
    assert "* = scaffolded" not in out  # the dead legend is gone
    for group, _names in _GROUPS:
        assert group in out
    # the three-line trust-stack map
    assert "posture" in out and "activity" in out and "proof" in out
    # the legacy gate spellings render as one compact line, not three full rows (a full row
    # leads with the rail glyph + name; the compact line carries them mid-line after the label)
    assert "views of /policy" in out
    for view in _GATE_VIEWS:
        assert f"│ /{view}" not in out


# --- /help <name> fallbacks ----------------------------------------------------------------------

def test_help_renamed_name_prints_same_pointer_as_direct(capsys):
    dispatch("/help why", _ctx())
    via_help = capsys.readouterr().out
    dispatch("/why", _ctx())
    direct = capsys.readouterr().out
    assert "/trace why" in via_help and "moved" in via_help
    assert via_help == direct


def test_help_unknown_name_still_errors(capsys):
    dispatch("/help not-a-command-at-all", _ctx())
    assert "unknown command" in capsys.readouterr().out


def test_cut_commands_command_gets_pointer(capsys):
    for spelling in ("/commands", "/cmds"):
        dispatch(spelling, _ctx())
        out = capsys.readouterr().out
        assert "moved" in out and "/help" in out
        assert "reload automatically" in out


# --- the dispatcher help grammar -----------------------------------------------------------------

def test_help_flag_first_or_last_shows_help_without_executing(capsys):
    calls = []
    COMMANDS["probetest"] = SlashCommand(
        name="probetest",
        summary="in-test probe",
        handler=lambda ctx, args: calls.append(args),
        usage="/probetest <args>",
        details="probe details body",
    )
    try:
        # --help/-h leading (position 0) or as the FINAL argument shows help — the muscle-memory
        # forms `/probetest --help`, `/probetest export --help`, `/probetest -H` (case-folded).
        for line in ("/probetest --help", "/probetest -H",
                     "/probetest export --help", "/probetest a b -h"):
            dispatch(line, _ctx())
            out = capsys.readouterr().out
            assert "probe details body" in out, line
        assert calls == []  # never executed

        # A MID-position token is data, not a flag — '/memory add prefer -h over --help in CLI
        # docs' must store the fact, never show help. (First-or-last semantics.)
        dispatch("/probetest add prefer -h over --help in docs", _ctx())
        assert calls == [["add", "prefer", "-h", "over", "--help", "in", "docs"]]
        assert "probe details body" not in capsys.readouterr().out

        # An argument merely CONTAINING the substring is data even at the edge positions.
        dispatch("/probetest notes--help.md", _ctx())
        assert calls[-1] == ["notes--help.md"]
        assert "probe details body" not in capsys.readouterr().out
    finally:
        COMMANDS.pop("probetest", None)

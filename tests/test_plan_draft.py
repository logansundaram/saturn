"""
/draft — the user-authored plan (2026-07-16; promoted from `/plan draft` to its own command the
same day). The user composes a plan BY HAND in the same plan_ops editor the mid-turn review
uses, the draft waits on ctx.pending_plan, and the next request executes it: the REPL seeds it
into the fresh turn's state and plan_node honors a pre-seeded plan verbatim (drafting skipped —
the human's plan outranks the engine's, the same authority rule plan review's vetoes encode).
Tool spellings normalize through the planner path's one authority (norm_tool + the no-tool
markers); an unresolvable spelling is KEPT RAW so execute fails closed on it instead of silently
answering the step from the model's priors.
"""

import pytest

from commands._framework import CommandContext
from commands.plan import _draft, _normalize_draft, _plan


@pytest.fixture
def ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


def _step(sid, label, tool=None, status="pending", result=None):
    return {"step_id": sid, "label": label, "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


def _out(capsys) -> str:
    return capsys.readouterr().out


# --- plan_node: the seed seam ------------------------------------------------------------------

def test_plan_node_honors_seeded_plan(monkeypatch):
    """A non-empty plan at plan_node means the REPL seeded a user draft this turn (_fresh_turn
    resets `plan` to [] on every boundary) — the planner must NOT run, and the plan must come
    back as a delta so the rail renders it like an engine draft."""
    from nodes import plan as plan_mod

    def _boom(*a, **k):
        raise AssertionError("the planner must not be called for a seeded plan")

    monkeypatch.setattr(plan_mod, "structured", _boom)
    seeded = [_step(1, "read the notes file", "read_file")]
    out = plan_mod.plan_node({"plan": seeded, "current_query": "q", "context": ""})
    assert out == {"plan": seeded}


def test_plan_node_still_drafts_when_plan_empty(monkeypatch):
    """The seam never skips a normal turn: an empty plan goes through the planner (here: a
    planner that returns nothing parseable, landing on the parse-error incident fallback)."""
    from core.structured import _PlanOut
    from nodes import plan as plan_mod
    from nodes.plan import PLAN_PARSE_ERROR

    roles = []

    def fake_structured(role, msgs, model, fmt, shape, default=None):
        roles.append(role)
        return _PlanOut()

    monkeypatch.setattr(plan_mod, "structured", fake_structured)
    out = plan_mod.plan_node({"plan": [], "current_query": "q", "context": ""})
    assert roles == ["planner"]
    assert out["plan"][0]["result"] == PLAN_PARSE_ERROR  # incident, not a silent reasoning step


# --- /draft: the command -------------------------------------------------------------------------

def test_draft_saves_and_normalizes_synonyms(ctx, monkeypatch, capsys):
    """A saved draft lands on ctx.pending_plan with tool spellings normalized onto the registry
    (calc → calculate), and the editor opens with the draft-mode wording, not the mid-turn
    review's ("execution paused" would be a lie between turns)."""
    from tui import ui

    payloads = []

    def fake_review(value):
        payloads.append(value)
        return {"action": "continue", "plan": [_step(1, "compute the total", "calc")]}

    monkeypatch.setattr(ui, "review_plan", fake_review)
    _draft(ctx, [])

    assert ctx.pending_plan is not None
    assert ctx.pending_plan[0]["intended_tool"] == "calculate"
    assert payloads[0]["title"] == "plan draft"
    assert payloads[0]["verbs"] == ("draft saved", "draft unchanged")
    assert "draft saved" in _out(capsys)


def test_draft_unknown_tool_kept_raw(ctx, monkeypatch, capsys):
    """An unresolvable tool spelling is preserved RAW (to_steps' rule): execute fails closed on
    it as an error incident — never silently degraded to a reasoning step — and the save notes
    the problem so the user can fix it before running."""
    from tui import ui

    monkeypatch.setattr(
        ui, "review_plan",
        lambda value: {"action": "continue", "plan": [_step(1, "do the thing", "frobnicate")]},
    )
    _draft(ctx, [])
    assert ctx.pending_plan[0]["intended_tool"] == "frobnicate"
    assert "not a registered tool" in _out(capsys)


def test_draft_no_tool_marker_becomes_reasoning_step(ctx, monkeypatch):
    """A spelled-out no-tool marker ("reasoning") normalizes to a genuine tool-less step."""
    from tui import ui

    monkeypatch.setattr(
        ui, "review_plan",
        lambda value: {"action": "continue", "plan": [_step(1, "summarize findings", "reasoning")]},
    )
    _draft(ctx, [])
    assert ctx.pending_plan[0]["intended_tool"] is None


def test_draft_abort_keeps_prior_draft(ctx, monkeypatch):
    """Abort (or Ctrl-C in the editor) never destroys the pending draft — the edit is discarded,
    not the draft. The editor also opens ON the pending draft so re-editing continues from it."""
    from tui import ui

    prior = [_step(1, "keep me", "read_file")]
    ctx.pending_plan = prior
    seen = []

    def fake_review(value):
        seen.append(value.get("plan"))
        return {"action": "abort", "plan": []}

    monkeypatch.setattr(ui, "review_plan", fake_review)
    _draft(ctx, [])
    assert ctx.pending_plan == prior
    assert seen[0] == prior


def test_draft_empty_save_clears(ctx, monkeypatch, capsys):
    """Saving an empty plan is the editor-native way to end up with no draft."""
    from tui import ui

    ctx.pending_plan = [_step(1, "old step")]
    monkeypatch.setattr(ui, "review_plan", lambda value: {"action": "continue", "plan": []})
    _draft(ctx, [])
    assert ctx.pending_plan is None
    assert "empty draft" in _out(capsys)


def test_draft_clear_discards(ctx, capsys):
    """`clear` (and the shared REMOVE_VERBS) discard the pending draft; with nothing pending
    they say so instead of pretending."""
    ctx.pending_plan = [_step(1, "x")]
    _draft(ctx, ["clear"])
    assert ctx.pending_plan is None
    assert "draft discarded" in _out(capsys)

    _draft(ctx, ["rm"])
    assert "no pending draft" in _out(capsys)


def test_draft_unknown_argument_errors(ctx, capsys):
    _draft(ctx, ["wat"])
    assert "unknown /draft argument" in _out(capsys)
    assert ctx.pending_plan is None


def test_plan_draft_is_a_moved_pointer(ctx, capsys):
    """`/plan draft` (the original spelling, promoted the same day it shipped) prints the moved
    pointer and never opens the editor or touches the pending draft."""
    ctx.pending_plan = [_step(1, "keep me")]
    _plan(ctx, ["draft"])
    out = _out(capsys)
    assert "moved" in out and "/draft" in out
    assert ctx.pending_plan == [_step(1, "keep me")]


def test_bare_plan_shows_pending_draft(ctx, capsys):
    ctx.state = {"plan": []}
    ctx.pending_plan = [_step(1, "my own step", "read_file")]
    _plan(ctx, [])
    assert "drafted plan (pending" in _out(capsys)


# --- _normalize_draft: the pure half -----------------------------------------------------------

def test_normalize_draft_mixed(capsys):
    plan = [
        _step(1, "search", "rag_search"),      # synonym → search_knowledge_base
        _step(2, "think", None),               # no tool stays no tool
        _step(3, "mystery", "made_up_tool"),   # unresolvable → kept raw + noted
    ]
    out, notes = _normalize_draft(plan)
    assert out[0]["intended_tool"] == "search_knowledge_base"
    assert out[1]["intended_tool"] is None
    assert out[2]["intended_tool"] == "made_up_tool"
    assert any("made_up_tool" in n for n in notes)

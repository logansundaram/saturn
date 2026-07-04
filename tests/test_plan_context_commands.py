"""
The /plan and /context command surfaces after the 2026-07-03 clean sweep: removed subcommands
(recipes, lockstep, `/context compact`) are plain unknown-argument errors — no legacy pointer
stubs — and nothing they name mutates. Bare `review` reports status (explicit on|off mutates;
the shared parse_toggle_status grammar), `--save` persists runtime.num_ctx via config.persist
(the shared split_save_flag grammar: case-insensitive, any position; with no value it persists
the CURRENT one), and plan_node always drafts (no seed seam).
"""

import pytest

from commands._framework import CommandContext
from commands.runtime import _context
from commands.plan import _plan


@pytest.fixture
def ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


@pytest.fixture
def no_model_rebuild(monkeypatch):
    """/context rebuilds the model cache on a resize; the no-op keeps the test side-effect-free."""
    from core import llms

    monkeypatch.setattr(llms, "reset_models", lambda: None)


@pytest.fixture
def recording_persist(monkeypatch):
    """Capture config.persist calls instead of writing the real config.yaml."""
    import config

    saved: list[str] = []
    monkeypatch.setattr(config, "persist", lambda key: saved.append(key) or config._CONFIG_PATH)
    return saved


def _out(capsys) -> str:
    return capsys.readouterr().out


# --- removed subcommands: clean-sweep — an unknown verb errors, no legacy pointers --------------

def test_removed_plan_subcommands_error_as_unknown(ctx, capsys, recording_persist):
    """The legacy pointer stubs (recipes 2026-06-11, lockstep 2026-07-03) were swept with the
    new-version cleanup: a removed verb is just an unknown subcommand, and nothing mutates."""
    from config import get_config

    for sub in ("save", "recipes", "run", "lockstep"):
        _plan(ctx, [sub, "whatever"])
        out = _out(capsys)
        assert "unknown /plan subcommand" in out
        assert "review, pause" in out  # the live verbs are named
    assert ctx.requeue is None  # nothing is queued to run
    assert recording_persist == []  # nothing persists
    assert "lockstep" not in get_config()._data.get("runtime", {})  # never (re)creates the key
    assert not hasattr(get_config(), "lockstep")  # the property is gone with the feature


def test_plan_node_always_drafts(monkeypatch):
    from core.structured import _PlanOut
    from nodes import plan as plan_mod

    assert not hasattr(plan_mod, "seed_next_plan")  # the recipe seam is gone

    # The hardened structured call degrading to its empty default (no model reachable) must
    # still yield a plan — the generic fallback step.
    monkeypatch.setattr(plan_mod, "structured", lambda *a, **k: _PlanOut())
    delta = plan_mod.plan_node({"context": "", "current_query": "q"})
    assert delta["plan"][0]["label"] == "Resolve the user's request"
    assert delta["plan"][0]["result"] is None  # the data-bus pointer marks it un-run


# --- /plan review: bare = status, explicit on|off mutates ---------------------------------------

def test_plan_review_bare_reports_without_flipping(ctx, capsys):
    _plan(ctx, ["review"])
    out = _out(capsys)
    assert "off" in out and "on|off to change" in out
    assert ctx.review_plan is False  # status never mutates

    _plan(ctx, ["review", "on"])
    assert ctx.review_plan is True
    _plan(ctx, ["review"])
    assert ctx.review_plan is True  # still a pure status read
    _plan(ctx, ["review", "off"])
    assert ctx.review_plan is False


# --- /context: compact forwarding gone (swept), --save persists num_ctx --------------------------

def test_context_compact_is_just_an_unknown_arg(ctx, capsys, no_model_rebuild):
    """The old `/context compact` pointer stub was swept with the new-version cleanup: the
    argument is not a size, so plain usage prints and nothing compacts or mutates."""
    _context(ctx, ["compact"])
    out = _out(capsys)
    assert "not a size" in out and "usage" in out


def test_context_set_size_session_only(ctx, capsys, monkeypatch, no_model_rebuild,
                                       recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", None)
    _context(ctx, ["16384"])
    assert cfg.num_ctx_override == 16384
    assert recording_persist == []
    assert "--save" in _out(capsys)  # the session-only note points at the persist flag


def test_context_set_size_save_persists(ctx, capsys, monkeypatch, no_model_rebuild,
                                        recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", None)
    _context(ctx, ["16384", "--save"])
    assert cfg.num_ctx_override == 16384
    assert recording_persist == ["runtime.num_ctx"]


def test_context_auto_save_persists_the_cleared_override(ctx, capsys, monkeypatch,
                                                         no_model_rebuild, recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", 8192)
    _context(ctx, ["auto", "--save"])
    assert cfg.num_ctx_override is None
    assert recording_persist == ["runtime.num_ctx"]


def test_context_bare_save_persists_current_window(ctx, capsys, monkeypatch, no_model_rebuild,
                                                   recording_persist):
    """`/context --save` with no size persists the CURRENT window setting (the shared
    convention) instead of printing usage — it mutates nothing live."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", 8192)
    _context(ctx, ["--save"])
    assert cfg.num_ctx_override == 8192  # unchanged
    assert recording_persist == ["runtime.num_ctx"]

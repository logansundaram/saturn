"""
The /plan and /context command surfaces after the June 2026 recipes cut: removed recipe
subcommands print a migration note, bare `review`/`lockstep` report status (explicit on|off
mutates; the shared parse_toggle_status grammar), `--save` persists runtime.lockstep /
runtime.num_ctx via config.persist (the shared split_save_flag grammar: case-insensitive, any
position; with no value it persists the CURRENT one), the `/context compact` forwarding is
gone, and plan_node always drafts (no seed seam).
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


# --- recipes cut -------------------------------------------------------------------------------

def test_removed_recipe_subcommands_print_migration_note(ctx, capsys):
    for sub in ("save", "recipes", "run"):
        _plan(ctx, [sub, "weekly-brief"])
        out = _out(capsys)
        assert "plan recipes were removed" in out
        assert "database/commands/<name>.md" in out
    assert ctx.requeue is None  # nothing is queued to run


def test_plan_node_always_drafts(monkeypatch):
    from nodes import plan as plan_mod

    assert not hasattr(plan_mod, "seed_next_plan")  # the recipe seam is gone

    class _NoModel:
        def invoke(self, prompt):
            raise RuntimeError("tests never reach a model")

    monkeypatch.setattr(plan_mod, "get_plan_model", lambda: _NoModel())
    delta = plan_mod.plan_node({"context": "", "current_query": "q"})
    # drafted (here: the no-model generic fallback) — there is no seed to consume
    assert delta["plan"][0]["label"] == "Resolve the user's request"


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


# --- /plan lockstep: bare = status, on|off mutates, --save persists ------------------------------

def test_plan_lockstep_bare_reports_without_flipping(ctx, capsys, monkeypatch):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep"])
    out = _out(capsys)
    assert "on" in out and "on|off to change" in out
    assert cfg.lockstep is True  # unchanged

    _plan(ctx, ["lockstep", "off"])
    assert cfg.lockstep is False


def test_plan_lockstep_save_persists(ctx, capsys, monkeypatch, recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep", "off", "--save"])
    assert cfg.lockstep is False
    assert recording_persist == ["runtime.lockstep"]


def test_plan_lockstep_no_save_does_not_persist(ctx, capsys, monkeypatch, recording_persist):
    from config import get_config

    monkeypatch.setitem(get_config()._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep", "off"])
    assert recording_persist == []


def test_plan_lockstep_bare_save_persists_current_value(ctx, capsys, monkeypatch,
                                                        recording_persist):
    """`--save` with no on|off persists the CURRENT value (the shared convention) — never a
    flip, never a refusal."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep", "--save"])
    assert cfg.lockstep is True  # unchanged
    assert recording_persist == ["runtime.lockstep"]


def test_plan_lockstep_save_flag_case_insensitive(ctx, capsys, monkeypatch, recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep", "off", "--SAVE"])  # split_save_flag: case-insensitive, any position
    assert cfg.lockstep is False
    assert recording_persist == ["runtime.lockstep"]


def test_plan_lockstep_unrecognized_arg_is_usage_not_flip(ctx, capsys, monkeypatch,
                                                          recording_persist):
    from config import get_config

    monkeypatch.setitem(get_config()._data["runtime"], "lockstep", True)
    _plan(ctx, ["lockstep", "maybe"])
    assert "usage" in _out(capsys)
    assert get_config().lockstep is True  # untouched (parse_toggle_status -> "invalid")
    assert recording_persist == []


# --- /context: compact forwarding removed, --save persists num_ctx -------------------------------

def test_context_compact_forwarding_removed(ctx, capsys, no_model_rebuild):
    _context(ctx, ["compact"])
    out = _out(capsys)
    assert "removed" in out and "/compact" in out  # a pointer, not a forwarded compaction


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

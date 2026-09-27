"""
The /plan and /config context command surfaces after the 2026-07-03 clean sweep + the 2026-07-07
command fold. Removed subcommands (recipes, lockstep, `/context compact`) are plain
unknown-argument errors — no legacy pointer stubs — and nothing they name mutates. The old
standalone /context folded into /config as `/config context` (dispatched to `_config_context`);
bare `review` reports status (explicit on|off mutates; the shared parse_toggle_status grammar),
and a size set PERSISTS runtime.num_ctx via config.persist BY DEFAULT (the shared
split_persist_flags grammar: settings persist by default, --session opts a single edit out; a bare
`--save` with no value still persists the CURRENT one). The RECIPE seed seam stays gone;
plan_node drafts unless the REPL seeded a user-drafted plan this turn (/draft, 2026-07-16 —
see test_plan_draft.py).
"""

import pytest

from commands._framework import CommandContext
from commands.config import _config_context as _context


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

def test_context_compact_is_just_an_unknown_arg(ctx, capsys, no_model_rebuild):
    """The old `/context compact` pointer stub was swept with the new-version cleanup: the
    argument is not a size, so plain usage prints and nothing compacts or mutates."""
    _context(ctx, ["compact"])
    out = _out(capsys)
    assert "not a size" in out and "usage" in out


def test_context_set_size_persists_by_default(ctx, capsys, monkeypatch, no_model_rebuild,
                                              recording_persist):
    """A size set now writes config.yaml BY DEFAULT — settings should survive a restart."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", None)
    _context(ctx, ["16384"])
    assert cfg.num_ctx_override == 16384
    assert recording_persist == ["runtime.num_ctx"]


def test_context_set_size_session_flag_stays_session_only(ctx, capsys, monkeypatch,
                                                          no_model_rebuild, recording_persist):
    """--session opts a single edit out of the persist-by-default."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["runtime"], "num_ctx", None)
    _context(ctx, ["16384", "--session"])
    assert cfg.num_ctx_override == 16384
    assert recording_persist == []
    out = _out(capsys)
    assert "session only" in out and "--session" in out


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

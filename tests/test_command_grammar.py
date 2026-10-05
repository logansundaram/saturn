"""
The shared command grammar (June 2026 audit): every removal verb in commands/_utils.REMOVE_VERBS
works identically in /docs, /memory, /resume, and /config key (which also keeps unset/clear) —
muscle memory transfers, so the audit's inversion ('/memory remove 3' failing while '/docs forget
x' worked) is gone. Plus /models: every binding form PERSISTS the SAME dotted key(s) the session
edit sets via config.persist BY DEFAULT (settings should survive a restart), while --session opts
a single edit out. The second audit pass added LIST_VERBS (`list`/`ls`, the `git stash
list`/`docker ls` spelling) accepted identically by every enumerating command, /mcp erroring
on unknown subcommands, and /config
riding the shared split_persist_flags grammar (bare 'save' is data — refused with a pointer, never
silently stored). Offline: the RAG drop is
stubbed, memory/sessions ride isolated_paths, .env is a tmp file, and config.persist is recorded
instead of writing the real config.yaml.
"""

import pytest

from commands._utils import LIST_VERBS, REMOVE_VERBS
from commands.config import _config
from commands.knowledge import _docs
from commands.knowledge import _memory
from commands.knowledge import _undo
from commands.runtime import _mcp, _models
from commands.conversation import _resume


@pytest.fixture
def models_env(monkeypatch):
    """Keep /models side-effect-free: no model-cache rebuild, and the embedder→re-embed seam is
    recorded (returned list) rather than touching the RAG store."""
    from core import llms
    from commands import runtime as models_mod

    monkeypatch.setattr(llms, "reset_models", lambda: None)
    resyncs: list[bool] = []
    monkeypatch.setattr(models_mod, "_resync_rag_after_model_change",
                        lambda: resyncs.append(True))
    return resyncs


def _out(capsys) -> str:
    return capsys.readouterr().out


# --- the shared removal-verb vocabulary -----------------------------------------------------

@pytest.mark.parametrize("verb", REMOVE_VERBS)
def test_docs_accepts_every_removal_verb(ctx, capsys, monkeypatch, verb):
    import stores.rag as rag

    dropped: list[str] = []
    monkeypatch.setattr(rag, "forget_document", lambda name: dropped.append(name) or True)
    _docs(ctx, [verb, "spec.pdf"])
    assert dropped == ["spec.pdf"]
    assert "removed spec.pdf" in _out(capsys)


@pytest.mark.parametrize("verb", REMOVE_VERBS)
def test_memory_accepts_every_removal_verb(ctx, capsys, isolated_paths, verb):
    from stores.memory_registry import add_memory, entries

    add_memory("the sky is blue")
    assert len(entries()) == 1
    _memory(ctx, [verb, "1"])
    assert entries() == []
    assert "removed:" in _out(capsys)


def test_memory_remove_by_index_the_audit_inversion(ctx, capsys, isolated_paths):
    """'/memory remove 3' was the audit's cross-inversion example: /docs accepted 'forget' but
    /memory rejected 'remove'. The shared vocabulary makes both directions work."""
    from stores.memory_registry import add_memory, entries

    for word in ("one", "two", "three"):
        add_memory(f"fact {word}")
    _memory(ctx, ["remove", "3"])
    facts = entries()
    assert len(facts) == 2
    assert not any("fact three" in f["text"] for f in facts)
    assert "removed:" in _out(capsys)


def test_memory_names_its_removal_verb_remove(ctx, capsys, isolated_paths):
    """Dogfooding 2026-10-05: `/memory add` beside `/memory forget` read as two vocabularies.
    The pair is add / remove, as in /docs; `forget` and the other removal verbs still work."""
    from commands._framework import COMMANDS

    for verb in ("remove", "forget"):
        _memory(ctx, [verb])
        assert "usage: /memory remove <n>" in _out(capsys)
    _memory(ctx, ["bogus"])
    assert "| remove <n> |" in _out(capsys)
    cmd = COMMANDS["memory"]
    assert "remove <n>" in cmd.usage and "forget <n>" not in cmd.usage
    assert "/memory remove <n>" in cmd.details and "/memory forget <n>" not in cmd.details
    assert "forget" in cmd.details      # still named as an accepted spelling


@pytest.mark.parametrize("verb", REMOVE_VERBS)
def test_resume_removal_verbs_intercepted_by_the_cut(ctx, capsys, isolated_paths, verb):
    """Session delete was CUT 2026-07-16 — every removal verb still ROUTES (so `/resume rm x`
    can't misparse as loading a session named 'rm x') but lands on the cut note and deletes
    nothing."""
    from commands._session import _session_file

    _session_file("scratch").write_text('{"version": 1, "messages": []}', encoding="utf-8")
    _resume(ctx, [verb, "scratch"])
    assert _session_file("scratch").exists()  # nothing deleted
    assert "was cut" in _out(capsys)


def test_config_key_is_cut(ctx, capsys, tmp_path, monkeypatch):
    """/config key was CUT 2026-07-16 — every spelling prints the .env pointer and mutates
    nothing (env vars are edited in .env directly; env_keys keeps only the read path MCP's
    ${VAR} expansion uses)."""
    import env_keys

    monkeypatch.setattr(env_keys, "_ENV_PATH", tmp_path / ".env")
    (tmp_path / ".env").write_text("MY_TEST_VAR=value-1\n", encoding="utf-8")

    for spelling in (["key"], ["key", "set", "MY_TEST_VAR", "v2"], ["key", "unset", "MY_TEST_VAR"]):
        _config(ctx, spelling)
        out = _out(capsys)
        assert "was cut" in out and ".env" in out
    assert env_keys.get("MY_TEST_VAR") == "value-1"  # nothing mutated
    assert not hasattr(env_keys, "set_value")  # the write path left with the command


# --- /models persists the same dotted keys the session edit sets (by default) ----------------

def _pin_tier(monkeypatch, spec: dict):
    """Give the active tier an explicit shape for this test, independent of the live
    config.yaml."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data["tiers"], cfg.active_tier, dict(spec, embedder="e"))
    return cfg


def test_models_use_save_persists_the_dotted_key(ctx, capsys, monkeypatch, models_env,
                                                 recording_persist):
    cfg = _pin_tier(monkeypatch, {"model": "qwen3.5:4b"})
    key = f"tiers.{cfg.active_tier}.model"

    _models(ctx, ["use", "qwen3.5:9b", "--save"])
    assert cfg.get(key) == "qwen3.5:9b"
    assert cfg.chat_model == "qwen3.5:9b"
    assert recording_persist == [key]
    assert "(session only)" not in _out(capsys)


def test_models_save_flag_case_insensitive_any_position(ctx, capsys, monkeypatch, models_env,
                                                        recording_persist):
    """--save / -s is still accepted (persisting is now the default): `-S` counts, anywhere."""
    cfg = _pin_tier(monkeypatch, {"model": "qwen3.5:4b"})
    key = f"tiers.{cfg.active_tier}.model"

    _models(ctx, ["use", "-S", "qwen3.5:9b"])
    assert cfg.get(key) == "qwen3.5:9b"
    assert recording_persist == [key]


def test_models_use_persists_by_default(ctx, capsys, monkeypatch, models_env,
                                        recording_persist):
    """A bare bind writes config.yaml BY DEFAULT — a model switch should stick."""
    cfg = _pin_tier(monkeypatch, {"model": "qwen3.5:4b"})

    _models(ctx, ["use", "qwen3.5:9b"])
    assert recording_persist == [f"tiers.{cfg.active_tier}.model"]
    assert "(session only)" not in _out(capsys)


def test_models_use_session_flag_stays_session_only(ctx, capsys, monkeypatch, models_env,
                                                    recording_persist):
    """--session opts a single bind out of the persist-by-default."""
    cfg = _pin_tier(monkeypatch, {"model": "qwen3.5:4b"})

    _models(ctx, ["use", "qwen3.5:9b", "--session"])
    assert cfg.chat_model == "qwen3.5:9b"
    assert recording_persist == []
    out = _out(capsys)
    assert "(session only)" in out and "--session" in out  # the note points at the flag


@pytest.mark.parametrize("spelling", ["all", "tool_caller", "utility"])
def test_models_retired_role_spellings_point_at_use(ctx, capsys, monkeypatch, models_env,
                                                    recording_persist, spelling):
    cfg = _pin_tier(monkeypatch, {"model": "qwen3.5:4b"})

    _models(ctx, [spelling, "qwen3.5:9b"])
    assert "/models use" in _out(capsys)
    assert cfg.chat_model == "qwen3.5:4b"  # nothing bound
    assert recording_persist == []


def test_models_embedder_save_persists_and_still_resyncs(ctx, capsys, monkeypatch, models_env,
                                                         recording_persist):
    from config import get_config

    cfg = get_config()
    tiers = cfg._data["tiers"]
    for tier in tiers.values():
        monkeypatch.setitem(tier, "embedder", tier["embedder"])

    _models(ctx, ["embedder", "test-embed", "--save"])
    # The embedder is a machine choice: the typed bind lands on EVERY tier, like the page's pick.
    assert all(cfg.get(f"tiers.{key}.embedder") == "test-embed" for key in tiers)
    assert recording_persist == [f"tiers.{key}.embedder" for key in tiers]
    assert models_env  # --save must not bypass the embedder→re-embed flow


# --- the shared listing-verb vocabulary (`git stash list` / `docker ls` style) ----------------

@pytest.mark.parametrize("verb", LIST_VERBS)
def test_docs_accepts_every_list_verb(ctx, monkeypatch, verb):
    import commands.knowledge as knowledge

    listed: list[bool] = []
    monkeypatch.setattr(knowledge, "_list_docs", lambda: listed.append(True))
    _docs(ctx, [verb])
    assert listed == [True]


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_memory_accepts_every_list_verb(ctx, capsys, isolated_paths, verb):
    from stores.memory_registry import add_memory, entries

    add_memory("the sky is blue")
    _memory(ctx, [verb])
    assert "the sky is blue" in _out(capsys)
    assert len(entries()) == 1  # a listing never mutates the store


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_resume_accepts_every_list_verb(ctx, capsys, isolated_paths, verb):
    from commands._session import _session_file

    _session_file("scratch").write_text('{"version": 1, "messages": []}', encoding="utf-8")
    _resume(ctx, [verb])
    assert "scratch" in _out(capsys)
    assert _session_file("scratch").exists()  # listed, not loaded or deleted


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_undo_accepts_every_list_verb(ctx, capsys, isolated_paths, verb):
    _undo(ctx, [verb])
    assert "no snapshots stored" in _out(capsys)  # the list view, not a restore attempt


@pytest.mark.parametrize("arg", ("lst", "lis", "2", "show"))
def test_undo_unknown_argument_never_reverts(ctx, capsys, monkeypatch, arg):
    """/undo is destructive with no redo: a typo'd listing attempt must ERROR, never fall
    through to the revert (the /mcp typo'd-'relod' rule — here the silent default would
    overwrite workspace files instead of printing a readout)."""
    from stores import snapshots

    reverted: list[bool] = []
    monkeypatch.setattr(snapshots, "undo_last",
                        lambda: reverted.append(True) or ("turn", []))
    _undo(ctx, [arg])
    assert "unknown argument" in _out(capsys)
    assert reverted == []  # the revert never ran


def test_bare_undo_still_reverts(ctx, capsys, monkeypatch):
    """Only a BARE /undo performs the restore — pin the happy path alongside the guard."""
    from stores import snapshots

    reverted: list[bool] = []
    monkeypatch.setattr(snapshots, "undo_last",
                        lambda: reverted.append(True) or ("turn", ["restored a.txt"]))
    _undo(ctx, [])
    assert reverted == [True]
    assert "restored a.txt" in _out(capsys)


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_models_list_verb_is_noninteractive(ctx, monkeypatch, models_env, verb):
    """`/models list` renders the page WITHOUT the row prompt (`ollama list` style) — bare
    /models keeps the interactive flow."""
    import commands.runtime as runtime_mod

    prompted: list[bool] = []
    monkeypatch.setattr(runtime_mod, "_models_page",
                        lambda cfg, *, prompt, **k: prompted.append(prompt))

    _models(ctx, [verb])
    assert prompted == [False]  # rendered, without the prompt
    _models(ctx, [])
    assert prompted == [False, True]


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_mcp_accepts_every_list_verb(ctx, capsys, verb):
    _mcp(ctx, [verb])
    assert "unknown subcommand" not in _out(capsys)


def test_mcp_unknown_subcommand_errors_instead_of_silent_status(ctx, capsys):
    _mcp(ctx, ["relod"])  # the typo'd reload must not silently render status as if it reloaded
    out = _out(capsys)
    assert "unknown subcommand" in out and "/mcp [list | reload]" in out


# --- /config rides the shared persist grammar (split_persist_flags) ---------------------------

@pytest.fixture
def runtime_key(monkeypatch):
    """Pin runtime.max_iterations so each test's session edit is restored after."""
    from config import get_config

    cfg = get_config()
    runtime = cfg._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "max_iterations", runtime.get("max_iterations", 10))
    return cfg


def test_config_save_flag_any_position_case_insensitive(ctx, capsys, runtime_key,
                                                        recording_persist):
    _config(ctx, ["--SAVE", "runtime.max_iterations", "12"])
    assert runtime_key.get("runtime.max_iterations") == 12
    assert recording_persist == ["runtime.max_iterations"]


def test_config_save_without_value_persists_current(ctx, capsys, runtime_key, recording_persist):
    """`--save` with no value persists the CURRENT value (the shared convention — the same act
    as /config persist <key>); it mutates nothing live."""
    before = runtime_key.get("runtime.max_iterations")
    _config(ctx, ["runtime.max_iterations", "--save"])
    assert runtime_key.get("runtime.max_iterations") == before
    assert recording_persist == ["runtime.max_iterations"]


def test_config_bare_save_word_is_refused_not_stored(ctx, capsys, runtime_key, recording_persist):
    """The old trailing bare-'save' flag form is gone (split_save_flag: only --save/-s count).
    Storing 'save' silently as value text would corrupt the setting — refuse and point."""
    before = runtime_key.get("runtime.max_iterations")
    _config(ctx, ["runtime.max_iterations", "12", "save"])
    out = _out(capsys)
    assert "did you mean --save" in out
    assert runtime_key.get("runtime.max_iterations") == before  # nothing set
    assert recording_persist == []


def test_config_set_persists_by_default(ctx, capsys, runtime_key, recording_persist):
    """A plain set now writes config.yaml BY DEFAULT — a setting should survive a restart."""
    _config(ctx, ["runtime.max_iterations", "12"])
    assert runtime_key.get("runtime.max_iterations") == 12
    assert recording_persist == ["runtime.max_iterations"]
    assert "session only" not in _out(capsys)


def test_config_set_session_flag_stays_session_only(ctx, capsys, runtime_key, recording_persist):
    """--session opts a single edit out of the persist-by-default."""
    _config(ctx, ["runtime.max_iterations", "12", "--session"])
    assert runtime_key.get("runtime.max_iterations") == 12
    assert recording_persist == []
    out = _out(capsys)
    assert "session only" in out and "--session" in out


# --- /config guards: section keys refuse, typo'd keys warn, missing keys read honestly --------

@pytest.fixture
def sandboxed_config(monkeypatch):
    """Run /config against a deep copy of the live config data, so keys the test creates (or
    sections it tries to clobber) never leak into other tests sharing the module singleton."""
    import copy
    from config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", copy.deepcopy(cfg._data))
    return cfg


def test_config_refuses_to_set_a_section(ctx, capsys, sandboxed_config, recording_persist):
    """`/config web foo` would scalar-replace the whole mapping in memory (web.* reads silently
    degrade to defaults) and `--save` would rewrite the bare `web:` header into unparseable
    YAML — refused at the door, with the child keys listed."""
    cfg = sandboxed_config
    before = dict(cfg.get("web"))
    _config(ctx, ["web", "foo"])
    out = _out(capsys)
    assert "is a section" in out and "web.max_results" in out
    assert cfg.get("web") == before  # nothing replaced
    assert recording_persist == []


def test_config_refuses_to_set_a_section_even_with_save(ctx, capsys, sandboxed_config,
                                                        recording_persist):
    cfg = sandboxed_config
    _config(ctx, ["runtime", "foo", "--save"])
    assert "is a section" in _out(capsys)
    assert isinstance(cfg.get("runtime"), dict)  # still a mapping
    assert recording_persist == []  # and nothing reached config.persist


def test_config_refuses_to_set_a_list(ctx, capsys, sandboxed_config, recording_persist):
    cfg = sandboxed_config
    cfg._data["scratch_list"] = ["a"]
    _config(ctx, ["scratch_list", "foo"])
    assert "is a list" in _out(capsys)
    assert cfg.get("scratch_list") == ["a"]
    assert recording_persist == []


def test_config_set_near_miss_key_warns_and_leaves_real_key(ctx, capsys, sandboxed_config):
    """A typo'd safety knob must not print a success-shaped line while the real setting stays
    untouched — warn, suggest the real key (the /policy risk did-you-mean wording)."""
    cfg = sandboxed_config
    before = cfg.get("runtime.auto_approve")
    _config(ctx, ["runtime.autoapprove", "destructive"])
    out = _out(capsys)
    assert "was not an existing config key" in out
    assert "did you mean runtime.auto_approve?" in out
    assert cfg.get("runtime.auto_approve") == before  # the real knob untouched
    assert "session only" not in out  # the warning REPLACES the success line


def test_config_set_saves_a_setting_the_file_predates(ctx, capsys, sandboxed_config,
                                                      recording_persist):
    """A config.yaml seeded before a setting existed does not hold it, but it is a real setting
    (the template has it): it sets and saves like any other, with no not-a-key warning."""
    cfg = sandboxed_config
    cfg._data["runtime"].pop("think", None)
    _config(ctx, ["runtime.think", "deep"])
    out = _out(capsys)
    assert cfg.get("runtime.think") == "deep"
    assert recording_persist == ["runtime.think"]
    assert "was not an existing config key" not in out and "session" not in out


def test_config_set_new_key_still_takes_effect(ctx, capsys, sandboxed_config):
    """Default-tolerant knobs must keep working on a
    config.yaml predating them: an absent key warns but still sets."""
    cfg = sandboxed_config
    _config(ctx, ["runtime.brand_new_knob", "true"])
    assert cfg.get("runtime.brand_new_knob") is True  # set (and coerced) despite the warning
    assert "was not an existing config key" in _out(capsys)


def test_config_read_missing_key_says_not_set(ctx, capsys, sandboxed_config):
    """An absent key reads as 'is not set' with a suggestion — never the success-shaped
    `= None`, which stays the rendering for a key explicitly present with a null value."""
    _config(ctx, ["runtime.max_iteratons"])
    out = _out(capsys)
    assert "is not set" in out
    assert "= None" not in out
    assert "did you mean runtime.max_iterations?" in out


def test_config_read_present_null_key_still_renders_none(ctx, capsys, sandboxed_config):
    cfg = sandboxed_config
    cfg._data.setdefault("runtime", {})["nullable_knob"] = None
    _config(ctx, ["runtime.nullable_knob"])
    assert "runtime.nullable_knob = None" in _out(capsys)


# --- /config reload matches case-insensitively like every sibling subcommand ------------------

@pytest.mark.parametrize("spelling", ["reload", "Reload", "RELOAD"])
def test_config_reload_case_insensitive(ctx, capsys, monkeypatch, spelling):
    """`/config Reload` must reload, not fall through to the dotted-key reader (it used to print
    the baffling `Reload = None`)."""
    import config as config_module
    import commands.config as config_cmd
    from core import llms

    calls: list[bool] = []
    monkeypatch.setattr(config_module, "reload",
                        lambda: calls.append(True) or config_module.get_config())
    monkeypatch.setattr(llms, "reset_models", lambda: None)
    monkeypatch.setattr(config_cmd, "_resync_rag_after_model_change", lambda: None)

    _config(ctx, [spelling])
    assert calls == [True]
    assert "reloaded" in _out(capsys)


# --- the trimmed command surface -------------------------------------------------------------

@pytest.mark.parametrize("sub", ["setup", "doctor", "check", "context", "persist"])
def test_retired_config_subcommands_point_somewhere_and_change_nothing(ctx, capsys, sub):
    from config import get_config

    before = dict(get_config()._data)
    _config(ctx, [sub, "4096"])
    out = _out(capsys)
    assert f"/config {sub} is gone" in out
    assert get_config()._data == before


def test_config_refuses_a_num_ctx_below_the_floor(ctx, capsys, monkeypatch, recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data, "runtime", dict(cfg._data.get("runtime", {})))
    _config(ctx, ["runtime.num_ctx", "100"])
    assert "too small" in _out(capsys)
    assert recording_persist == []


def test_docs_rebuild_is_the_forced_sync_and_sync_points_at_it(ctx, capsys, monkeypatch):
    import commands.knowledge as knowledge

    calls = []
    monkeypatch.setattr(knowledge, "_sync", lambda *, force: calls.append(force))
    _docs(ctx, ["rebuild"])
    assert calls == [True]
    _docs(ctx, ["sync", "--force"])
    assert calls == [True]  # sync no longer runs anything
    assert "/docs rebuild" in _out(capsys)


def test_clear_takes_no_arguments_and_has_no_aliases(ctx, capsys):
    from commands import dispatch
    from commands._framework import _ALIASES
    from commands.conversation import _clear

    ctx.state = {"messages": ["kept"]}
    _clear(ctx, ["--screen"])
    assert "usage: /clear" in _out(capsys)
    assert ctx.state == {"messages": ["kept"]}  # an argument never wipes the conversation
    for alias in ("cls", "reset", "new"):
        assert alias not in _ALIASES
        dispatch(f"/{alias}", ctx)
        assert "unknown command" in _out(capsys)
    assert ctx.state == {"messages": ["kept"]}


def test_memory_pending_is_gone(ctx, capsys, isolated_paths):
    _memory(ctx, ["pending"])
    assert "unknown" in _out(capsys).lower()


def test_config_summary_shows_the_launch_folder_not_the_fallback_path(ctx, capsys, tmp_path):
    from core import workspace

    launched, added = tmp_path / "proj", tmp_path / "other"
    launched.mkdir()
    added.mkdir()
    try:
        workspace.set_root(launched)
        workspace.add(added)
        _config(ctx, [])
    finally:
        workspace.reset()
    out = _out(capsys)
    assert f"working folder          : {workspace.display(launched.resolve())}" in out
    assert workspace.display(added.resolve()) in out and "/add-dir" in out
    assert "database/workspace" not in out

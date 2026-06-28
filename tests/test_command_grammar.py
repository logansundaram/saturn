"""
The shared command grammar (June 2026 audit): every removal verb in commands/_utils.REMOVE_VERBS
works identically in /docs, /memory, /resume, and /config key (which also keeps unset/clear) —
muscle memory transfers, so the audit's inversion ('/memory remove 3' failing while '/docs forget
x' worked) is gone. Plus /models --save: every binding form persists the SAME dotted key(s) the
session edit sets via config.persist, while no---save stays session-only. The second audit pass
added LIST_VERBS (`list`/`ls`, the `git stash list`/`docker ls` spelling) accepted identically by
every enumerating command, /models --provider (the named-flag form of the bare positional), /mcp
erroring on unknown subcommands, and /config riding the shared split_save_flag grammar (bare
'save' is data — refused with a pointer, never silently stored). Offline: the RAG drop is
stubbed, memory/sessions ride isolated_paths, .env is a tmp file, and config.persist is recorded
instead of writing the real config.yaml.
"""

import pytest

from commands._framework import CommandContext
from commands._utils import _ROLES, LIST_VERBS, REMOVE_VERBS
from commands.config import _config
from commands.knowledge import _docs
from commands.knowledge import _memory
from commands.knowledge import _undo
from commands.runtime import _mcp, _models
from commands.conversation import _resume


@pytest.fixture
def ctx():
    return CommandContext(state={}, make_initial_state=dict, db_path="")


@pytest.fixture
def recording_persist(monkeypatch):
    """Capture config.persist calls instead of writing the real config.yaml."""
    import config

    saved: list[str] = []
    monkeypatch.setattr(config, "persist", lambda key: saved.append(key) or config._CONFIG_PATH)
    return saved


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
    from stores.memory_registry import add_memory, list_memory

    add_memory("the sky is blue")
    assert len(list_memory()) == 1
    _memory(ctx, [verb, "1"])
    assert list_memory() == []
    assert "forgot:" in _out(capsys)


def test_memory_remove_by_index_the_audit_inversion(ctx, capsys, isolated_paths):
    """'/memory remove 3' was the audit's cross-inversion example: /docs accepted 'forget' but
    /memory rejected 'remove'. The shared vocabulary makes both directions work."""
    from stores.memory_registry import add_memory, list_memory

    for word in ("one", "two", "three"):
        add_memory(f"fact {word}")
    _memory(ctx, ["remove", "3"])
    facts = list_memory()
    assert len(facts) == 2
    assert not any("fact three" in f for f in facts)
    assert "forgot:" in _out(capsys)


@pytest.mark.parametrize("verb", REMOVE_VERBS)
def test_resume_accepts_every_removal_verb(ctx, capsys, isolated_paths, verb):
    from commands._session import _session_file

    _session_file("scratch").write_text('{"version": 1, "messages": []}', encoding="utf-8")
    _resume(ctx, [verb, "scratch"])
    assert not _session_file("scratch").exists()
    assert "deleted session" in _out(capsys)


@pytest.mark.parametrize("verb", REMOVE_VERBS + ("unset", "clear"))
def test_config_key_accepts_unset_clear_and_every_removal_verb(ctx, capsys, tmp_path,
                                                               monkeypatch, verb):
    import env_keys

    monkeypatch.setattr(env_keys, "_ENV_PATH", tmp_path / ".env")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    # setenv registers the prior (absent) state so teardown drops whatever the test set.
    monkeypatch.setenv("MY_TEST_VAR", "seed")
    env_keys.set_value("MY_TEST_VAR", "value-1")  # ALL-CAPS = unmanaged: no on_change hook
    assert env_keys.is_set("MY_TEST_VAR")

    _config(ctx, ["key", verb, "MY_TEST_VAR"])
    assert not env_keys.is_set("MY_TEST_VAR")
    assert "removed from .env" in _out(capsys)


# --- /models --save persists the same dotted keys the session edit sets ----------------------

def test_models_role_save_persists_the_dotted_key(ctx, capsys, monkeypatch, models_env,
                                                  recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])
    key = f"tiers.{cfg.active_tier}.roles.planner"

    _models(ctx, ["planner", "test-model", "--save"])
    assert cfg.get(key) == "test-model"
    assert recording_persist == [key]
    assert "(session only)" not in _out(capsys)


def test_models_save_flag_case_insensitive_any_position(ctx, capsys, monkeypatch, models_env,
                                                        recording_persist):
    """The shared split_save_flag grammar: `-S`/`--SAVE` count, anywhere in the args."""
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])
    key = f"tiers.{cfg.active_tier}.roles.planner"

    _models(ctx, ["planner", "-S", "test-model"])
    assert cfg.get(key) == "test-model"
    assert recording_persist == [key]


def test_models_role_without_save_stays_session_only(ctx, capsys, monkeypatch, models_env,
                                                     recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])

    _models(ctx, ["planner", "test-model"])
    assert cfg.get(f"tiers.{cfg.active_tier}.roles.planner") == "test-model"
    assert recording_persist == []
    out = _out(capsys)
    assert "(session only)" in out and "--save" in out  # the note points at the persist flag


def test_models_all_save_persists_every_role_key(ctx, capsys, monkeypatch, models_env,
                                                 recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    for role in _ROLES:
        monkeypatch.setitem(roles, role, roles[role])

    _models(ctx, ["all", "test-model", "--save"])
    assert recording_persist == [f"tiers.{cfg.active_tier}.roles.{r}" for r in _ROLES]
    for role in _ROLES:
        assert cfg.get(f"tiers.{cfg.active_tier}.roles.{role}") == "test-model"


def test_models_embedder_save_persists_and_still_resyncs(ctx, capsys, monkeypatch, models_env,
                                                         recording_persist):
    from config import get_config

    cfg = get_config()
    tier = cfg._data["tiers"][cfg.active_tier]
    monkeypatch.setitem(tier, "embedder", tier["embedder"])

    _models(ctx, ["embedder", "test-embed", "--save"])
    assert cfg.get(f"tiers.{cfg.active_tier}.embedder") == "test-embed"
    assert recording_persist == [f"tiers.{cfg.active_tier}.embedder"]
    assert models_env  # --save must not bypass the embedder→re-embed flow


def test_models_tier_save_persists_active_tier(ctx, capsys, monkeypatch, models_env,
                                               recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data, "active_tier", cfg._data["active_tier"])
    other = next(t for t in cfg._data["tiers"] if t != cfg.active_tier)

    _models(ctx, ["tier", other, "--save"])
    assert cfg.active_tier == other
    assert recording_persist == ["active_tier"]


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
    from stores.memory_registry import add_memory, list_memory

    add_memory("the sky is blue")
    _memory(ctx, [verb])
    assert "the sky is blue" in _out(capsys)
    assert len(list_memory()) == 1  # a listing never mutates the store


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
    """`/models list` renders the table WITHOUT dropping into the picker (`ollama list` style) —
    bare /models keeps the interactive flow."""
    from core import llms
    import commands.runtime as runtime_mod
    from tui import ui

    monkeypatch.setattr(llms, "list_local_models", lambda: [])
    shown: list[bool] = []
    monkeypatch.setattr(ui, "show_models", lambda *a, **k: shown.append(bool(k.get("numbered"))))
    picked: list[bool] = []
    monkeypatch.setattr(runtime_mod, "_models_picker", lambda *a, **k: picked.append(True))

    _models(ctx, [verb])
    assert shown == [False]  # rendered, without picker numbering
    assert picked == []      # and never prompted


@pytest.mark.parametrize("verb", LIST_VERBS)
def test_mcp_accepts_every_list_verb(ctx, capsys, verb):
    _mcp(ctx, [verb])
    assert "unknown subcommand" not in _out(capsys)


def test_mcp_unknown_subcommand_errors_instead_of_silent_status(ctx, capsys):
    _mcp(ctx, ["relod"])  # the typo'd reload must not silently render status as if it reloaded
    out = _out(capsys)
    assert "unknown subcommand" in out and "/mcp [list | reload]" in out


# --- /models --provider: the named-flag spelling of the bare positional ----------------------

def test_models_provider_flag_binds_provider_form(ctx, capsys, monkeypatch, models_env,
                                                  recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])

    _models(ctx, ["planner", "claude-x", "--provider", "anthropic"])
    assert cfg.get(f"tiers.{cfg.active_tier}.roles.planner") == {
        "provider": "anthropic", "model": "claude-x",
    }


def test_models_provider_flag_and_positional_must_agree(ctx, capsys, monkeypatch, models_env,
                                                        recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])
    before = cfg.get(f"tiers.{cfg.active_tier}.roles.planner")

    _models(ctx, ["planner", "claude-x", "openai", "--provider", "anthropic"])
    assert "disagree" in _out(capsys)
    assert cfg.get(f"tiers.{cfg.active_tier}.roles.planner") == before  # nothing bound


def test_models_dangling_provider_flag_refuses(ctx, capsys, monkeypatch, models_env,
                                               recording_persist):
    from config import get_config

    cfg = get_config()
    roles = cfg._data["tiers"][cfg.active_tier]["roles"]
    monkeypatch.setitem(roles, "planner", roles["planner"])
    before = cfg.get(f"tiers.{cfg.active_tier}.roles.planner")

    _models(ctx, ["planner", "claude-x", "--provider"])
    assert "needs a value" in _out(capsys)
    assert cfg.get(f"tiers.{cfg.active_tier}.roles.planner") == before


def test_models_provider_flag_refused_on_tier_and_embedder(ctx, capsys, monkeypatch, models_env,
                                                           recording_persist):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg._data, "active_tier", cfg._data["active_tier"])

    _models(ctx, ["tier", cfg.active_tier, "--provider", "anthropic"])
    assert "--provider applies to role bindings" in _out(capsys)
    assert recording_persist == []


# --- /config rides the shared --save grammar (split_save_flag) -------------------------------

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


def test_config_set_without_save_stays_session_only(ctx, capsys, runtime_key, recording_persist):
    _config(ctx, ["runtime.max_iterations", "12"])
    assert runtime_key.get("runtime.max_iterations") == 12
    assert recording_persist == []
    assert "session only" in _out(capsys)


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
    _config(ctx, ["web", "tavily"])
    out = _out(capsys)
    assert "is a section" in out and "web.provider" in out
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


def test_config_set_new_key_still_takes_effect(ctx, capsys, sandboxed_config):
    """Default-tolerant knobs (shell.background, paths.user_commands, …) must keep working on a
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

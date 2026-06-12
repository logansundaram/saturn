"""
The shared command grammar (June 2026 audit): every removal verb in commands/_utils.REMOVE_VERBS
works identically in /docs, /memory, /resume, and /config key (which also keeps unset/clear) —
muscle memory transfers, so the audit's inversion ('/memory remove 3' failing while '/docs forget
x' worked) is gone. Plus /models --save: every binding form persists the SAME dotted key(s) the
session edit sets via config.persist, while no---save stays session-only. Offline: the RAG drop
is stubbed, memory/sessions ride isolated_paths, .env is a tmp file, and config.persist is
recorded instead of writing the real config.yaml.
"""

import pytest

from commands._framework import CommandContext
from commands._utils import _ROLES, REMOVE_VERBS
from commands.config import _config
from commands.docs import _docs
from commands.memory import _memory
from commands.models import _models
from commands.resume import _resume


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
    from commands import models as models_mod

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

"""
First-launch onboarding polish (2026-06-11), pure pieces only: the doctor's optional-key
rendering and tier-honesty closing line, the inline `ollama pull` offer DECISION (the
interactive prompt + subprocess pulls themselves are deliberately untested), the one-line RAG
ingest warning selection, and /init's absolute-path success message. Offline — reachability is
injected, never probed.
"""

import agent
from commands import config as config_cmd
from config import Config


# --- doctor: api-key machinery --------------------------------------------------------------

def test_doctor_key_machinery_left_with_the_config_key_cut():
    """The doctor's per-key rendering (_key_line/_required_keys/_OPTIONAL_KEY_NOTES) was CUT
    2026-07-16 with /config key — nothing needs a key (inference is local, the web tools are
    keyless), so the doctor prints one honest line instead. A resurrected helper here
    means the cut regressed."""
    for gone in ("_key_line", "_required_keys", "_OPTIONAL_KEY_NOTES"):
        assert not hasattr(config_cmd, gone), gone


# --- doctor: tier-honesty closing line ------------------------------------------------------
# The line fires when the active tier is a ladder class at or below the install default AND
# more than one tier exists; a tier named outside the ladder never fires it.

_CAPS = {
    "qwen3.5:2b": {"context_window": 8192},
    "qwen3.5:9b": {"context_window": 32768},
}


def _tier(model, **role_overrides):
    from config import MODEL_ROLES

    roles = {r: model for r in MODEL_ROLES}
    roles.update(role_overrides)
    return {"roles": roles}


def _cfg(active, tiers):
    return Config({"active_tier": active, "tiers": tiers, "capabilities": _CAPS})


def test_tier_honesty_line_names_the_model_and_the_upgrade_pointer():
    from core import model_family as mf

    line = config_cmd._tier_honesty_line(_cfg(mf.DEFAULT_CLASS, _ladder_tiers()))
    assert line is not None
    assert "small model tier" in line
    assert mf.tag_for(mf.DEFAULT_CLASS) in line   # the active tier's tool_caller model, derived live
    assert "/models" in line                      # the upgrade pointer


def _ladder_tiers():
    from core import model_family as mf

    return {key: _tier(tag) for key, tag in mf.SIZE_LADDER}


def test_tier_honesty_fires_on_the_fresh_install_default_tier():
    from core import model_family as mf

    line = config_cmd._tier_honesty_line(_cfg(mf.DEFAULT_CLASS, _ladder_tiers()))
    assert line is not None
    assert mf.tag_for(mf.DEFAULT_CLASS) in line


def test_tier_honesty_fires_on_every_class_up_to_the_default():
    for key in config_cmd._small_classes():
        assert config_cmd._tier_honesty_line(_cfg(key, _ladder_tiers())) is not None, key


def test_tier_honesty_silent_on_the_classes_above_the_default():
    from core import model_family as mf

    small = set(config_cmd._small_classes())
    bigger = [k for k in mf.classes() if k not in small]
    assert bigger                                   # there is something to upgrade to
    for key in bigger:
        assert config_cmd._tier_honesty_line(_cfg(key, _ladder_tiers())) is None, key


def test_tier_honesty_silent_on_a_tier_named_outside_the_ladder():
    tiers = {"laptop": _tier("qwen3.5:2b"), "workstation": _tier("qwen3.5:9b")}
    assert config_cmd._tier_honesty_line(_cfg("laptop", tiers)) is None
    assert config_cmd._tier_honesty_line(_cfg("workstation", tiers)) is None


def test_tier_honesty_silent_with_a_single_preset():
    assert config_cmd._tier_honesty_line(_cfg("only", {"only": _tier("qwen3.5:2b")})) is None


# --- doctor: the inline-pull offer decision -------------------------------------------------

def test_should_offer_pull_truth_table():
    offer = config_cmd._should_offer_pull
    assert offer(["gemma4:e4b"], True, True)
    assert not offer([], True, True)              # nothing missing
    assert not offer(["gemma4:e4b"], False, True)  # daemon down — nothing to pull into
    assert not offer(["gemma4:e4b"], True, False)  # off-TTY / headless: never prompt


# --- the one-line RAG ingest warning ---------------------------------------------------------

class _Boom(Exception):
    pass


def test_ingest_warning_ollama_down_defers_to_the_model_check():
    msg = agent._ingest_warning(_Boom("connect error\nmultiline repr"), reachable=False)
    assert msg == (
        "knowledge-base ingest skipped (Ollama not reachable — the model check below explains)"
    )


def test_ingest_warning_headless_drops_the_below_claim():
    # Headless (-p) prints no health check after the warning, so the deferral clause would
    # overclaim there.
    msg = agent._ingest_warning(_Boom("x"), reachable=False, interactive=False)
    assert msg == "knowledge-base ingest skipped (Ollama not reachable)"
    assert "below" not in msg


def test_ingest_warning_other_failures_clip_to_one_line():
    exc = _Boom("first line\n  second   line\n" + "x" * 1000)
    msg = agent._ingest_warning(exc, reachable=True)
    assert msg.startswith(
        "knowledge-base ingest failed, continuing without RAG: first line second line"
    )
    assert "\n" not in msg
    assert len(msg) < 400  # clipped, never the full repr


def test_ingest_warning_empty_exception_names_the_class():
    msg = agent._ingest_warning(_Boom(), reachable=True)
    assert "_Boom" in msg


# --- /init: success message orients the user -------------------------------------------------

def test_init_success_prints_absolute_workspace_path(isolated_paths, capsys):
    from commands.knowledge import _init
    from config import get_config

    _init(None, [])  # empty isolated workspace -> template branch, no LLM call
    out = capsys.readouterr().out
    target = get_config().path("workspace") / "SATURDAY.md"
    assert target.exists()
    assert str(target) in out  # the ABSOLUTE path, not a bare basename
    assert "sandboxed workspace, not your current directory" in out


def test_init_existing_file_refusal_also_prints_the_path(isolated_paths, capsys):
    from commands.knowledge import _init
    from config import get_config

    workspace = get_config().path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "SATURDAY.md"
    target.write_text("# mine\n", encoding="utf-8")
    _init(None, [])
    out = capsys.readouterr().out
    assert "already exists" in out
    assert str(target) in out
    assert target.read_text(encoding="utf-8") == "# mine\n"  # refused without --force


# --- standing instructions: ~/.saturn/SATURN.md + the workspace file -------------------------

def test_global_saturn_md_loads_under_the_workspace_file(isolated_paths, monkeypatch, tmp_path):
    from nodes import ground

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SATURN_HOME", raising=False)
    monkeypatch.setattr(ground, "memory_context_split", lambda q: ("", "", []))
    assert ground.global_instructions_path() == tmp_path / ".saturn" / "SATURN.md"
    assert "Standing instructions" not in ground.stable_grounding()  # no file, no section

    (tmp_path / ".saturn").mkdir()
    (tmp_path / ".saturn" / "SATURN.md").write_text("always metric", encoding="utf-8")
    workspace = isolated_paths / "database" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "SATURDAY.md").write_text("old name", encoding="utf-8")
    (workspace / "SATURN.md").write_text("be terse", encoding="utf-8")
    stable = ground.stable_grounding()
    assert "Standing instructions (~/.saturn/SATURN.md" in stable and "always metric" in stable
    assert "Workspace instructions (SATURN.md" in stable and "be terse" in stable
    assert "old name" not in stable  # SATURN.md wins over SATURDAY.md when both exist
    assert stable.index("always metric") < stable.index("be terse")  # global first, workspace refines

    (workspace / "SATURN.md").unlink()
    assert "Workspace instructions (SATURDAY.md" in ground.stable_grounding()  # the old name still reads


def test_saturn_home_overrides_the_global_instructions_dir(monkeypatch, tmp_path):
    from nodes import ground

    monkeypatch.setenv("SATURN_HOME", str(tmp_path / "elsewhere"))
    assert ground.global_instructions_path() == tmp_path / "elsewhere" / "SATURN.md"


# --- /memory names the file it keeps ---------------------------------------------------------

def test_memory_listing_names_the_file(isolated_paths, capsys):
    from commands.knowledge import _memory
    from config import get_config

    _memory(None, [])  # empty store
    assert str(get_config().path("memory")) in capsys.readouterr().out
    from stores import memory_registry as mr
    mr.add_memory("likes tea")
    _memory(None, [])
    out = capsys.readouterr().out
    assert str(get_config().path("memory")) in out and "likes tea" in out

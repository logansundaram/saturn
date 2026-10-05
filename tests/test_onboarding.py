"""
First-launch onboarding, pure pieces only: the doctor's absence (folded into /models), the
one-line RAG ingest warning selection, and /init's absolute-path success message. Offline — reachability is
injected, never probed.
"""

import agent
from commands import config as config_cmd


# --- /config setup (the doctor) folded into /models + the startup health check (2026-09-30) ---

def test_the_doctor_is_gone():
    """A resurrected helper here means the fold regressed: /models offers the pulls, the
    startup health check warns, /mcp shows MCP status."""
    for gone in ("_config_doctor", "_tier_honesty_line", "_should_offer_pull", "_offer_pull",
                 "_key_line", "_required_keys", "_OPTIONAL_KEY_NOTES"):
        assert not hasattr(config_cmd, gone), gone


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
    target = get_config().path("workspace") / "SATURN.md"
    assert target.exists()
    assert str(target) in out  # the ABSOLUTE path, not a bare basename


def test_init_existing_file_refusal_also_prints_the_path(isolated_paths, capsys):
    from commands.knowledge import _init
    from config import get_config

    workspace = get_config().path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "SATURN.md"
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
    assert "old name" not in stable  # SATURDAY.md is never read (cut 2026-09-30)
    assert stable.index("always metric") < stable.index("be terse")  # global first, workspace refines

    (workspace / "SATURN.md").unlink()
    assert "Workspace instructions" not in ground.stable_grounding()  # the old name alone loads nothing


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

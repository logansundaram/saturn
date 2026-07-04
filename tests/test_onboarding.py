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


# --- doctor: optional-key rendering --------------------------------------------------------

def test_key_line_set_is_ok_with_no_fix_arrow():
    line = config_cmd._key_line("TAVILY_API_KEY", True, frozenset())
    assert line.startswith("ok")
    assert "->" not in line


def test_key_line_missing_required_gets_the_fix_arrow():
    # No key is currently required (cloud shelved — _required_keys is always empty), but the
    # rendering seam stays: a name in the required set gets the fix arrow.
    line = config_cmd._key_line(
        "SOME_REQUIRED_KEY", False, frozenset({"SOME_REQUIRED_KEY"})
    )
    assert "MISSING" in line
    assert "-> /config key set SOME_REQUIRED_KEY" in line


def test_key_line_optional_tavily_names_the_keyless_fallback():
    line = config_cmd._key_line("TAVILY_API_KEY", False, frozenset())
    assert line.startswith("optional")
    assert "keyless fallback active" in line
    assert "->" not in line  # fix-arrow vocabulary reserved for genuinely broken items


def test_key_line_optional_unlisted_key_is_not_a_gap():
    line = config_cmd._key_line("SOME_OTHER_KEY", False, frozenset())
    assert line.startswith("optional")
    assert "not needed by the active tier" in line
    assert "->" not in line


def test_required_keys_empty_under_the_cloud_shelve():
    # Cloud model support is shelved (2026-07-03): even a legacy config still carrying a cloud
    # binding requires no key — the binding itself can't run (check_models reports it; the key
    # would unlock nothing).
    cfg = _cfg("hybrid", {
        "local": _tier("tiny"),
        "hybrid": _tier("tiny", planner={"provider": "anthropic", "model": "cloud"}),
    })
    assert config_cmd._required_keys(cfg) == set()
    assert config_cmd._required_keys(_cfg("local", {"local": _tier("tiny")})) == set()


def test_check_models_reports_a_shelved_cloud_binding(monkeypatch):
    """A pre-shelve config with a cloud-bound role must surface at startup as an actionable
    problem, not as a mid-turn failure."""
    import config as config_mod
    from core import llms

    cfg = _cfg("hybrid", {
        "hybrid": _tier("tiny", planner={"provider": "anthropic", "model": "claude-x"}),
    })
    cfg._data["tiers"]["hybrid"]["embedder"] = "tiny-embed"
    monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
    monkeypatch.setattr(llms, "list_local_models", lambda: [])
    monkeypatch.setattr(llms, "ollama_reachable", lambda: True)
    problems = llms.check_models()
    assert any("cloud model support is shelved" in p and "planner" in p for p in problems)
    assert not any("ANTHROPIC" in p for p in problems)  # no key demand for a shelved binding


# --- doctor: tier-honesty closing line ------------------------------------------------------
# Convention under test: config.yaml's `tiers:` mapping is declared smallest -> largest (YAML
# mapping order is preserved), so the FIRST declared tier is the smallest. The line fires when
# the active tier is first-declared AND more than one tier exists — declaration order, never a
# size heuristic.

_CAPS = {
    "tiny": {"context_window": 8192},
    "mid": {"context_window": 32768},
    "cloud": {"context_window": 200000},
}


def _tier(model, **role_overrides):
    from config import MODEL_ROLES

    roles = {r: model for r in MODEL_ROLES}
    roles.update(role_overrides)
    return {"provider": "ollama", "roles": roles}


def _cfg(active, tiers):
    return Config({"active_tier": active, "tiers": tiers, "capabilities": _CAPS})


def test_tier_honesty_fires_on_the_first_declared_preset():
    cfg = _cfg("laptop", {"laptop": _tier("tiny"), "workstation": _tier("mid")})
    line = config_cmd._tier_honesty_line(cfg)
    assert line is not None
    assert "smallest model tier" in line
    assert "tiny" in line          # the active tier's tool_caller model, derived live
    assert "/models" in line       # the upgrade pointer


def test_tier_honesty_silent_on_a_later_declared_tier():
    cfg = _cfg("workstation", {"laptop": _tier("tiny"), "workstation": _tier("mid")})
    assert config_cmd._tier_honesty_line(cfg) is None


def test_tier_honesty_silent_with_a_single_preset():
    assert config_cmd._tier_honesty_line(_cfg("only", {"only": _tier("tiny")})) is None


def test_tier_honesty_is_declaration_order_not_window_sums():
    # The old heuristic summed declared context windows — orthogonal to model size: a small
    # model with a huge window outsummed a big model with a modest one, firing the line on the
    # wrong tier. Declaration order decides now: the FIRST tier fires even when its windows
    # outsum the second's, and the second never fires even when its windows are smaller.
    tiers = {"small-but-big-window": _tier("cloud"), "big-but-small-window": _tier("tiny")}
    assert config_cmd._tier_honesty_line(_cfg("small-but-big-window", tiers)) is not None
    assert config_cmd._tier_honesty_line(_cfg("big-but-small-window", tiers)) is None


def test_tier_honesty_silent_on_a_hybrid_declared_after_the_local_preset():
    # The hybrid preset is declared after the all-local one (bigger by convention), so only the
    # first-declared local preset triggers the line.
    tiers = {
        "laptop": _tier("tiny"),
        "hybrid": _tier(
            "tiny",
            planner={"provider": "anthropic", "model": "cloud"},
            synthesizer={"provider": "anthropic", "model": "cloud"},
        ),
    }
    assert config_cmd._tier_honesty_line(_cfg("hybrid", tiers)) is None
    assert config_cmd._tier_honesty_line(_cfg("laptop", tiers)) is not None


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

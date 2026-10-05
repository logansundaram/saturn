"""
/think (commands/think.py) and the one-turn `/think <request>` seam (app/session.think_for_line)
— spec docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md §5.
"""

from types import SimpleNamespace

import pytest

import commands
from core import think


def _cfg(monkeypatch, think_policy=None, **runtime):
    """Set runtime.* for one test. `think_policy` is not a setting: it points `auto` at a
    policy for the test (core.think.set_policy is what the benchmark's baseline run uses)."""
    from config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "runtime": {**cfg._data.get("runtime", {}), **runtime}})
    if think_policy is not None:
        assert think_policy in think.POLICIES
        monkeypatch.setattr(think, "_POLICY", think_policy)
    return cfg


def _run(line, state=None):
    ctx = SimpleNamespace(state=state or {}, should_quit=False)
    commands.dispatch(line, ctx)
    return ctx


# ── the level-word rule ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("line, want", [
    ("/think why does my build fail", "why does my build fail"),
    ("/think deep dive into the logs", "deep dive into the logs"),   # more than a level word
    ("  /THINK  Is 391 prime?  ", "Is 391 prime?"),
    ("/think fast enough for a 4b?", "fast enough for a 4b?"),
    ("/think", None),
    ("/think deep", None), ("/think fast --session", None), ("/think on", None),
    ("/think --session auto", None),
    ("/think --help", None), ("/think what is this -h", None),
    ("/thinking of you", None), ("think about it", None), ("/skills", None), ("", None),
])
def test_think_for_line(line, want):
    from app.session import think_for_line
    assert think_for_line(line) == want


def test_think_is_a_built_in_so_no_skill_can_shadow_it():
    assert commands.resolves("think")


# ── the command ──────────────────────────────────────────────────────────────────────────────


def test_bare_think_is_the_readout(monkeypatch, capsys):
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "qwen3.5:4b")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    _cfg(monkeypatch, think="auto", think_budget=1024)
    d = think.decide("auto", "information")
    state = {"think": [
        think.entry(n=1, kind="first", decision=think.decide("auto", "first")),
        think.entry(n=2, kind="information", decision=d, outcome="thought", thought={"seconds": 1.84}),
    ]}
    _run("/think", state)
    out = capsys.readouterr().out
    assert "thinking: auto — thinks before it acts" in out and "what auto does" in out
    assert "policy" not in out  # one rule behind auto; nothing to choose between
    assert "qwen3.5:4b" in out and "1024 tokens" in out
    lines = {ln.split("  ")[2].strip(): ln for ln in out.splitlines() if ln.startswith("    ") and "pass" not in ln}
    for kind in ("first move", "new information", "wrap-up"):
        assert "thinks only if it is about to call a tool" in lines[kind]
    assert lines["after an error"].rstrip().endswith("thinks")
    assert lines["you steered"].rstrip().endswith("thinks")
    assert lines["budget spent"].rstrip().endswith("no thought")
    assert "pass 1: first move · no thought" in out and "pass 2: new information · thought 1.8s" in out


def test_the_readout_says_when_the_model_cannot_think(monkeypatch, capsys):
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "plain:8b")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", {"plain:8b"})
    _cfg(monkeypatch, think="deep")
    _run("/think")
    out = capsys.readouterr().out
    assert "plain:8b rejects the think flag" in out
    assert "thinks\n" not in out.replace("no thought", "")  # nothing thinks on this model
    assert "last turn" not in out


def test_the_readout_flags_a_setting_that_is_not_a_level(monkeypatch, capsys):
    _cfg(monkeypatch, think="always")
    _run("/think")
    out = capsys.readouterr().out
    assert "'always'" in out and "thinking: auto" in out


@pytest.mark.parametrize("word, level", [("deep", "deep"), ("FAST", "fast"), ("on", "deep"),
                                         ("off", "fast"), ("adaptive", "auto")])
def test_setting_the_level_for_the_session(monkeypatch, capsys, word, level):
    cfg = _cfg(monkeypatch, think="auto")
    _run(f"/think {word} --session")
    assert cfg.get("runtime.think") == level and think.level() == level
    assert f"thinking: {level}" in capsys.readouterr().out


CONFIG = """\
# the user's file
active_tier: 4b
runtime:
  max_iterations: 16   # the cap
  think: auto          # fast | auto | deep
tiers:
  4b:
    model: m
"""


def _config_file(monkeypatch, tmp_path, text):
    """Point config.persist at a throwaway config.yaml (never the real one)."""
    import config
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return path


def test_setting_the_level_persists_by_default(monkeypatch, tmp_path, capsys):
    import yaml
    cfg = _cfg(monkeypatch, think="auto")
    path = _config_file(monkeypatch, tmp_path, CONFIG)
    _run("/think deep")
    after = path.read_text()
    assert "  think: deep          # fast | auto | deep" in after   # one line edited, comment kept
    assert after.replace("think: deep", "think: auto") == CONFIG
    assert think.normalise(yaml.safe_load(after)["runtime"]["think"]) == ("deep", True)
    assert "saved to config.yaml" in capsys.readouterr().out
    _run("/think fast --session")
    assert path.read_text() == after and cfg.get("runtime.think") == "fast"


def test_a_config_without_the_think_line_gains_it(monkeypatch, tmp_path, capsys):
    """A config.yaml seeded before the key existed has no line to edit (dogfooding 2026-10-05:
    the level could not be saved at all). The line is added under `runtime:`, nothing else in
    the file changes, and the next save edits that same line."""
    import yaml
    cfg = _cfg(monkeypatch, think="auto")
    old = CONFIG.replace("  think: auto          # fast | auto | deep\n", "")
    path = _config_file(monkeypatch, tmp_path, old)
    _run("/think deep")
    out = capsys.readouterr().out
    after = path.read_text()
    assert after.replace("  think: deep\n", "") == old
    assert yaml.safe_load(after)["runtime"] == {"max_iterations": 16, "think": "deep"}
    assert cfg.get("runtime.think") == "deep" and "saved to config.yaml" in out
    _run("/think fast")
    assert path.read_text() == after.replace("think: deep", "think: fast")


def test_a_config_without_the_runtime_section_is_left_exactly_as_it_was(monkeypatch, tmp_path,
                                                                       capsys):
    """With no `runtime:` section there is nowhere to add the line: the level applies for the
    session, the file is untouched — never emptied — and the user is told what to add."""
    cfg = _cfg(monkeypatch, think="auto")
    old = "# the user's file\nactive_tier: 4b\ntiers:\n  4b:\n    model: m\n"
    path = _config_file(monkeypatch, tmp_path, old)
    _run("/think deep")
    out = capsys.readouterr().out
    assert path.read_text() == old
    assert cfg.get("runtime.think") == "deep" and "Add `think: deep`" in out


def test_a_persisted_level_round_trips_through_yaml():
    """`on` and `off` are YAML booleans: the level is written as its own word, which reads
    back as the same level."""
    import yaml

    from config import _dump_scalar
    for level in think.LEVELS:
        raw = yaml.safe_load("think: " + _dump_scalar(level))["think"]
        assert think.normalise(raw) == (level, True)


def test_a_request_that_reaches_the_command_is_pointed_at_the_prompt(monkeypatch, capsys):
    cfg = _cfg(monkeypatch, think="auto")
    _run("/think why is the sky blue")
    assert "type it at the prompt" in capsys.readouterr().out
    assert cfg.get("runtime.think") == "auto"


def test_think_help(capsys):
    _run("/think --help")
    out = capsys.readouterr().out
    assert "/think <request>" in out and "deep dive" in out


# ── the seams ────────────────────────────────────────────────────────────────────────────────


def test_the_startup_lines_name_a_bad_setting(monkeypatch):
    _cfg(monkeypatch, think="yes please")
    (line,) = think.problems()
    assert "runtime.think" in line and "running as auto" in line


def test_headless_runs_a_think_request_at_deep(isolated_paths, monkeypatch, capsys):
    """`saturn -p "/think <request>"`: the prefix comes off, the turn carries think_level=deep,
    and the model sees the plain request."""
    import io

    from langchain.messages import AIMessage

    from app import headless

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr(headless, "startup_load", lambda interactive=False: (object(), None))
    monkeypatch.setattr(headless, "DB_PATH", str(isolated_paths / "trace.sqlite"))
    seen = {}

    def fake_turn(graph, state, config, **kw):
        seen["level"] = state.get("think_level")
        seen["query"] = state["current_query"]
        seen["last"] = state["messages"][-1].content
        return {**state, "messages": state["messages"] + [AIMessage(content="391 = 17 × 23")]}

    monkeypatch.setattr(headless, "run_turn", fake_turn)
    args = SimpleNamespace(prompt="/think Is 391 prime?", query=None, json=False, export=None, yolo=False)
    headless.run_headless(args)
    captured = capsys.readouterr()
    assert seen == {"level": "deep", "query": "Is 391 prime?", "last": "Is 391 prime?"}
    assert "391 = 17 × 23" in captured.out and "thinking: deep for this run" in captured.err

    seen.clear()
    headless.run_headless(SimpleNamespace(prompt="Is 391 prime?", query=None, json=False, export=None, yolo=False))
    assert seen["level"] == "" and seen["query"] == "Is 391 prime?"

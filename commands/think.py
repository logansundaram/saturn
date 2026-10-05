"""
/think — how much Saturn reasons before it answers (core/think.py; spec
docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md). One front door: the readout
(the level, what `auto` does per kind of step, the last turn pass by pass), the level setter,
and — through app/session.think_for_line, before dispatch — `/think <request>`, one turn at
`deep`.
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import split_persist_flags

_KEY = "runtime.think"

# What each level means, in the readout's words.
_LEVEL_LINES = {
    "fast": "never thinks",
    "auto": "thinks before it acts, never before a plain answer",
    "deep": "thinks on every pass but the capped last one",
}


def _readout(ctx) -> None:
    from core import llms, think

    level = think.level()
    for line in think.problems():
        _print(f"  ⚠ {line}")
    _print(f"  thinking: {level} — {_LEVEL_LINES[level]}")
    tag = llms.model_tag() or "the model"
    if think.supported():
        _print(f"  model: {tag} · a thought may spend {think.budget()} tokens · Esc stops one")
    else:
        _print(f"  model: {tag} rejects the think flag — every pass runs without thinking")
    _print(f"  what {level} does")
    for kind in ("first", "information", "recovery", "steered", "wrap-up", "capped"):
        d = think.decide(level, kind, supported=think.supported())
        said = ("thinks only if it is about to call a tool" if d.draft
                else "thinks" if d.think else "no thought")
        _print(f"    {think.KIND_WORDS[kind]:<16} {said}")
    passes = [e for e in ((getattr(ctx, "state", None) or {}).get("think") or []) if isinstance(e, dict)]
    if passes:
        _print("  last turn")
        for e in passes:
            _print(f"    pass {e.get('pass')}: {think.describe(e)}")
    _print("  /think fast|auto|deep sets the level · /think <request> runs one request at deep")


@command(
    "think",
    "How much Saturn reasons before it answers: fast, auto or deep.",
    usage="/think | /think fast|auto|deep [--session] | /think <request>",
    details="""
Thinking is the model reasoning before it answers. It costs seconds and, on the passes where a
decision is being made, it is what gets the decision right. Saturn decides per pass, from the
kind of step the pass is — never from the words of your request.

  /think                    the level, what each kind of pass does, and the last turn pass by pass
  /think fast               never think
  /think auto               think before acting: a pass about to call a tool is thought
                            through first; a plain answer never waits on a thought
  /think deep               think on every pass
  /think deep --session     for this session only (the default writes config.yaml)
  /think <request>          run this one request at deep, whatever the level is

A level word on its own sets the level; anything else is a request (`/think deep dive into the
logs` asks about the logs). While a thought is in flight the status bar says so and Esc stops
it; a thought is also cut at runtime.think_budget tokens. A thought that is stopped, cut or
comes back empty is dropped and the pass answers without it.
""",
)
def _think(ctx, args):
    from config import get_config
    from core import think

    rest, session_only, _save = split_persist_flags(list(args))
    if not rest:
        _readout(ctx)
        return
    if len(rest) != 1 or not think.is_level_word(rest[0]):
        # A request reaches this handler only when something dispatched the line directly (the
        # REPL and headless seams take `/think <request>` before dispatch).
        _print("  /think <request> runs one request at deep — type it at the prompt. "
               "To set the level: /think fast|auto|deep")
        return
    level = think.normalise(rest[0])[0]
    cfg = get_config()
    cfg.set(_KEY, level)
    if session_only:
        _print(f"  thinking: {level} — {_LEVEL_LINES[level]} (this session only)")
        return
    from config import persist

    _print(f"  thinking: {level} — {_LEVEL_LINES[level]}")
    try:
        _print(f"  saved to {persist(_KEY).name}")
    except KeyError:
        # persist adds the line to a config.yaml seeded before the key existed; what is left
        # is a file with no `runtime:` section to add it to.
        _print(f"  set for this session — config.yaml has no `runtime:` section to save it in. "
               f"Add `think: {level}` under `runtime:` to keep it.")
    except Exception as exc:
        _print(f"  set for this session, but saving failed: {exc}")

"""Saturn's entry point — `python agent.py` from the repo root, or the `saturn` console script
(`agent:main` in pyproject.toml).

This file is deliberately thin: it forces UTF-8 console output, parses the command line, and
routes into the `app/` package, where the application actually lives:

    app/cli.py       the argparse surface + piped-stdin capture
    app/graph.py     build_agent(): the nodes/ package wired into the ReAct loop
    app/turn.py      run_turn(): drive one turn (stream updates + tokens, resolve interrupts)
    app/session.py   per-turn state shape, fresh-turn reset, history compaction
    app/startup.py   shared startup: knowledge-base sync + attachment warnings
    app/headless.py  the -p/-q path (one query -> stdout; --json / --export / -q receipt)
    app/repl.py      the interactive loop

The re-export block below is the import surface benchmark.py and the tests use (`from agent
import build_agent, run_turn, ...`). New code should import from the app/ modules directly.
"""

import sys
from pathlib import Path

# Force UTF-8 console output: rail glyphs, tool results and model output routinely carry
# characters a non-UTF-8 locale's stdout would refuse.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app import __version__  # noqa: E402  (single source: app/__init__.py)

# Compatibility re-exports — the names benchmark.py and tests/ import from `agent`.
from app.cli import _build_parser, _parse_cli, _read_piped_stdin  # noqa: E402,F401
from app.graph import DB_PATH, build_agent  # noqa: E402,F401
from app.session import (  # noqa: E402,F401
    _compact_history,
    _fresh_turn,
    _initial_state,
    _maybe_autocompact,
)
from app.startup import _ingest_warning, _warn_flagged_attachments, startup_load  # noqa: E402,F401
from app.turn import run_turn, _make_on_update, _trace_warning  # noqa: E402,F401


def main():
    """CLI entry point: parse the command line, then route — --replay / headless -p / the
    interactive TUI loop. The flag reference lives in `saturn --help` (see app/cli.py)."""
    args = _parse_cli()

    # --- replay path: render an exported run record and exit (no graph, no models) -----
    if args.replay:
        from commands.trace import render_export

        sys.exit(0 if render_export(args.replay) else 1)

    # The launch folder is the workspace (core/workspace): the file tools, the shell
    # and the folder's SATURN.md follow it in both modes. Set once, before either runs.
    from core import workspace

    try:
        launched = Path.cwd()
    except OSError:  # the folder was deleted under the shell
        launched = Path.home()
    workspace.set_root(launched)

    # --yolo: the CLI view of the gate policy — open the gate up front (threshold ->
    # destructive) so gated calls never interrupt; same mechanism as /policy open. Honored in
    # BOTH modes: interactively the status bar derives ⚠ GATE OFF straight from the live
    # threshold, so no extra UI wiring is needed.
    if args.yolo:
        from trust import policy

        policy.set_gate_off(True)

    # --- headless path: one query, print answer, exit (-p, or the -q one-shot render) --
    if args.prompt is not None or args.query is not None:
        from app.headless import run_headless

        run_headless(args)
        return

    # --- interactive path -------------------------------------------------------------
    from app.repl import run_repl

    run_repl()


if __name__ == "__main__":
    main()

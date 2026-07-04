"""The application shell — everything between the `saturn` command line and the graph nodes.

Split out of the old 1,000-line agent.py (2026-07-03) so each concern reads on its own:

    cli.py       the argparse surface (strict flags, exit 2 on typos) + piped-stdin capture
    graph.py     build_agent(): wire the nodes/ package into the compiled plan/execute engine
    turn.py      run_turn(): drive one turn — stream updates + answer tokens, resolve interrupts
    session.py   cross-turn state: the per-turn state shape, fresh-turn reset, history compaction
    startup.py   shared startup work: knowledge-base sync + attachment admission warnings
    headless.py  the -p path: one query -> stdout, honoring the --json/--export contracts
    repl.py      the interactive loop: prompt, type-ahead, slash commands, autosave

The root agent.py remains the entry point (`saturn = agent:main` in pyproject.toml): it parses
the CLI, routes into headless.py or repl.py, and re-exports the names benchmark.py and the
tests import (`from agent import build_agent, run_turn, ...` keeps working).
"""

__version__ = "0.1.0"  # keep in sync with `version` in pyproject.toml

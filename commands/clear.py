from commands._framework import command, _print


@command(
    "clear",
    "Start a fresh conversation: reset state + clear the screen.",
    aliases=("cls", "reset", "new"),
    details="""
The "new conversation" button. Drops the in-memory conversation — the message history and every
per-turn field (plan, iteration, accumulators) — AND clears the visible terminal, then reprints
the session header. One command for a clean slate.

What is NOT touched: config, model/tier bindings, the RAG corpus, the durable memory store
(remember/recall), and the on-disk trace. The trace survives, so /trace and /calls still show
past runs after a clear.

Pass --screen (-s) to ONLY repaint the terminal, leaving the conversation intact.

Aliases /reset and /new are the same fresh-start; /cls too.

Examples:
  /clear            new conversation + clean screen
  /clear --screen   just repaint the terminal, keep the conversation
""",
)
def _clear(ctx, args):
    import subprocess
    import sys

    screen_only = bool(args) and args[0].lower() in ("--screen", "-s", "screen")

    if not screen_only:
        ctx.state = ctx.make_initial_state()

    subprocess.run("cls" if sys.platform == "win32" else "clear", shell=True, check=False)

    if screen_only:
        return

    _reprint_banner(ctx)
    _print("  new conversation — fresh state, no message history.")


def _reprint_banner(ctx) -> None:
    """Repaint the startup session header after a clear, so the fresh slate looks like a new launch.
    Best-effort: a failure here must never undo the reset that already happened."""
    try:
        from config import get_config
        from llms import model_id
        from registry import tool as _tools
        from stores.rag import iter_documents
        from tui import ui

        cfg = get_config()
        n_docs = sum(1 for _ in iter_documents())
        ui.banner(f"{cfg.active_tier}:{model_id('tool_caller')}", len(_tools), n_docs, ctx.db_path)
    except Exception:
        pass

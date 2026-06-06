from commands._framework import command, _print


@command(
    "reset",
    "Reset the conversation (clears messages + per-turn state).",
    aliases=("new",),
    details="""
Rebuilds a clean AgentState: drops the message history and every per-turn field (plan,
iteration, accumulators). Starts a fresh conversation without restarting the process.

Config, model/tier bindings, the RAG corpus, and persistent memory are all unaffected —
only the in-process conversation is cleared.

Example:
  /reset
""",
)
def _reset(ctx, args):
    ctx.state = ctx.make_initial_state()
    _print("  conversation reset — fresh state, no message history.")

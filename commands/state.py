from commands._framework import command, _print


@command(
    "state",
    "Dump a summary of the current agent state.",
    usage="/state [--full]",
    details="""
Prints a one-line-per-field summary of the live AgentState: message count, current query,
loop iteration, plan step count, the tools called this turn, and how many documents were
retrieved.

Pass --full to also dump the raw state dict (verbose — useful for debugging, noisy otherwise).

Examples:
  /state
  /state --full
""",
)
def _state(ctx, args):
    s = ctx.state
    _print("  agent state:")
    _print(f"    messages      : {len(s.get('messages', []))}")
    _print(f"    current_query : {s.get('current_query', '')!r}")
    _print(f"    iteration     : {s.get('iteration', 0)}")
    _print(f"    plan steps    : {len(s.get('plan', []))}")
    _print(f"    tools_called  : {s.get('tools_called', [])}")
    _print(f"    docs_retrieved: {len(s.get('documents_retrieved', []))}")
    if "--full" in args:
        _print("    ---- raw ----")
        _print(f"    {s}")

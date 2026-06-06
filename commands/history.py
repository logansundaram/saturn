from commands._framework import command, _print


@command(
    "history",
    "Print the conversation messages for this session.",
    aliases=("hist",),
    usage="/history [n]",
    details="""
Prints the conversation messages held in memory this session — one line each: index, role
(human / ai / tool / system), and content with whitespace collapsed to a single line. AI turns
that only call tools (empty content) surface the tool names instead.

Pass n to show just the most recent n messages. This is the in-process scratchpad, not the
durable trace — see /trace for the on-disk run record, /reset to clear it.

Examples:
  /history
  /history 10
""",
)
def _history(ctx, args):
    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  (no messages yet)")
        return

    shown = messages
    if args:
        try:
            n = int(args[0])
            shown = messages[-n:] if n > 0 else messages
        except ValueError:
            _print(f"  ignoring non-numeric count: {args[0]!r}")

    _print(f"  conversation history ({len(shown)} of {len(messages)} messages):")
    for i, msg in enumerate(shown, start=len(messages) - len(shown) + 1):
        role = getattr(msg, "type", type(msg).__name__)
        content = msg.content
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        content = " ".join(str(content).split())
        if len(content) > 100:
            content = content[:99] + "…"
        line = f"    {i:>3}  {role:<6} {content}"
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            names = ", ".join(tc.get("name", "?") for tc in tool_calls)
            line += f"[tool_calls: {names}]" if not content else f"  → {names}"
        _print(line)

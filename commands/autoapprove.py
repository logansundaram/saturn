from commands._framework import command, _print
from commands._utils import _parse_toggle


@command(
    "autoapprove",
    "Toggle the approval gate (auto-approve side-effecting tools).",
    aliases=("yolo",),
    usage="/autoapprove on|off",
    details="""
Disables (or re-enables) the human-in-the-loop approval gate for the session. When ON, every
tool call — including side-effecting and destructive ones — runs WITHOUT prompting, and a loud
banner is printed on enable.

⚠  This removes the main safety check. Use it only when you trust the task and the tools.
Prefer /risk to relax a single tool while keeping the gate on. With no argument, flips the
current state.

Examples:
  /autoapprove on
  /autoapprove off
  /yolo            alias — same thing
""",
)
def _autoapprove(ctx, args):
    new = _parse_toggle(args, ctx.auto_approve)
    if new is None:
        _print(f"  usage: /autoapprove on|off   (currently {'on' if ctx.auto_approve else 'off'})")
        return
    ctx.auto_approve = new
    if new:
        _print("  ┏━ ⚠  AUTO-APPROVE ON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        _print("  ┃  the approval gate is DISABLED. every tool call —")
        _print("  ┃  including side-effecting and destructive ones —")
        _print("  ┃  will run WITHOUT asking. /autoapprove off to restore.")
        _print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        _print("  auto-approve off — the approval gate is back on.")

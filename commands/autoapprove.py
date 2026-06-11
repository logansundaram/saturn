from commands._framework import command, _print
from commands._utils import _parse_toggle


@command(
    "autoapprove",
    "Open the approval gate — sets the auto-approve threshold to `destructive`.",
    aliases=("yolo",),
    usage="/autoapprove [on|off]",
    details="""
Opens (or closes) the human-in-the-loop approval gate for the session. This is not a separate
switch: it is a view of the ONE gate policy (policy.py) — `on` raises the `runtime.auto_approve`
threshold to `destructive` so every tier auto-approves; `off` restores the threshold that was in
effect before. The status bar shows ⚠ GATE OFF for as long as the threshold sits there, however
it got there (this command, Shift+Tab cycling, or config.yaml).

⚠  This removes the main safety check. Use it only when you trust the task and the tools.
Prefer /risk to relax a single tool, or /allow to exempt specific shell commands, while keeping
the gate on. With no argument, flips the current state. Session-only — it never persists.

Examples:
  /autoapprove on
  /autoapprove off
  /yolo            alias — same thing (mirrors the headless --yolo flag)
""",
)
def _autoapprove(ctx, args):
    import policy

    new = _parse_toggle(args, policy.gate_off())
    if new is None:
        _print(f"  usage: /autoapprove on|off   (currently {'on' if policy.gate_off() else 'off'})")
        return
    policy.set_gate_off(new)
    if new:
        _print("  ┏━ ⚠  AUTO-APPROVE ON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        _print("  ┃  the approval gate is OPEN: auto-approve threshold =")
        _print("  ┃  destructive, so every tool call — including side-")
        _print("  ┃  effecting and destructive ones — runs WITHOUT asking.")
        _print("  ┃  /autoapprove off to restore the previous threshold.")
        _print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        _print(f"  auto-approve off — gate threshold restored to `{policy.tier()}`.")

"""
CLI rendering for the live plan and the approval prompt.

The plan is rendered as a quiet, gutter-aligned list that nests under the node stream
(`show_node` prints `|  <node>`), so it reads as ambient progress rather than a separate
panel — it blends with the rest of the terminal instead of standing out. The approval prompt
keeps a thin boxed grid on purpose: it is a blocking safety gate and *should* draw the eye.

The agent emits plan/state updates; this module is one *subscriber* that renders them (the
"side window" of SATURDAY_MVP_PLAN.md §6). Swapping it for a Textual sidebar or an Electron
window later requires no change to the graph. Degrades to plain ASCII if `rich` is absent.
"""

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _console = Console()
    _RICH = True
except Exception:  # pragma: no cover - fallback path
    _console = None
    _RICH = False

# Calm blueprint palette.
_ACCENT = "cyan"
_LINE = "dim cyan"

# status -> rich style for the plan line. The marker carries the state; only the active step
# gets a touch of accent so the current step is findable. Everything else recedes (dim).
_STATUS = {
    "pending": "dim",
    "active": _ACCENT,
    "done": "dim",
    "skipped": "dim strike",
}
_MARKER = {"pending": "[ ]", "active": "[~]", "done": "[x]", "skipped": "[-]"}


def show_plan(plan) -> None:
    """Quiet, gutter-aligned plan list that nests under the node stream — no box, no rules,
    no title. Reads as ambient progress, not a separate panel."""
    if not plan:
        return

    for s in plan:
        marker = _MARKER.get(s["status"], "[ ]")
        tool = s.get("intended_tool")
        if _RICH:
            # Build with Text.append (no markup parsing) so brackets in markers/labels are
            # treated literally, not as Rich style tags.
            line = Text()
            line.append("|   ", style=_LINE)
            line.append(
                f"{marker} {s['step_id']:>2}  {s['label']}",
                style=_STATUS.get(s["status"], "dim"),
            )
            if tool:
                line.append(f"  ::{tool}", style="dim")
            _console.print(line)
        else:
            tool_txt = f"  ::{tool}" if tool else ""
            print(f"|    {marker} {s['step_id']:>2}  {s['label']}{tool_txt}")


def show_node(node: str) -> None:
    if _RICH:
        _console.print(f"[{_LINE}]|  {node}[/{_LINE}]")
    else:
        print(f"|  {node}")


def ask_approval(value: dict) -> bool:
    """Render the gated tool calls as a thin grid and ask the user. True approves."""
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    if _RICH:
        table = Table(
            box=box.SQUARE,
            border_style=_LINE,
            header_style=f"bold {_ACCENT}",
            title="APPROVAL REQUIRED",
            title_style=f"bold {_ACCENT}",
            title_justify="left",
            show_lines=True,
            expand=False,
            pad_edge=False,
        )
        table.add_column("RISK", style=_ACCENT, no_wrap=True)
        table.add_column("TOOL", no_wrap=True)
        table.add_column("ARGUMENTS")
        for tc in tool_calls:
            table.add_row(str(tc.get("risk")), str(tc.get("name")), str(tc.get("args")))
        _console.print(table)
    else:
        print("\n+-- APPROVAL REQUIRED " + "-" * 27)
        for tc in tool_calls:
            print(f"  [{tc.get('risk')}] {tc.get('name')}({tc.get('args')})")
        print("+" + "-" * 48)

    resp = input("approve? [y/N] > ").strip().lower()
    return resp in ("y", "yes")

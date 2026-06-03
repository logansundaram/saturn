"""
CLI rendering for the live plan and the approval prompt — "blueprint terminal" aesthetic:
thin single-line grids, uppercase labels, a calm cyan accent, no emoji. Engineered, not edgy.

The agent emits plan/state updates; this module is one *subscriber* that renders them (the
"side window" of SATURDAY_MVP_PLAN.md §6). Swapping it for a Textual sidebar or an Electron
window later requires no change to the graph. Degrades to plain ASCII if `rich` is absent.
"""

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    _console = Console()
    _RICH = True
except Exception:  # pragma: no cover - fallback path
    _console = None
    _RICH = False

# Calm blueprint palette.
_ACCENT = "cyan"
_LINE = "dim cyan"

# status -> (label, rich style) for the grid, and (marker) for the ASCII fallback.
_STATUS = {
    "pending": ("PENDING", "dim"),
    "active": ("ACTIVE", f"bold {_ACCENT}"),
    "done": ("DONE", _ACCENT),
    "skipped": ("SKIPPED", "dim strike"),
}
_MARKER = {"pending": "[ ]", "active": "[~]", "done": "[x]", "skipped": "[-]"}


def render_plan(plan) -> str:
    """Plain-text (ASCII) plan, used as the fallback and by callers that want a string."""
    if not plan:
        return "(no plan)"
    lines = []
    for s in plan:
        tool = f"  ::{s['intended_tool']}" if s.get("intended_tool") else ""
        lines.append(f"{_MARKER.get(s['status'], '[ ]')} {s['step_id']:>2}  {s['label']}{tool}")
    return "\n".join(lines)


def show_plan(plan) -> None:
    if not _RICH:
        print("\n+-- PLAN " + "-" * 40)
        print(render_plan(plan))
        print("+" + "-" * 48 + "\n")
        return

    table = Table(
        box=box.SQUARE,
        border_style=_LINE,
        header_style=f"bold {_ACCENT}",
        title="PLAN",
        title_style=f"bold {_ACCENT}",
        title_justify="left",
        show_lines=True,
        expand=False,
        pad_edge=False,
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("STATUS", no_wrap=True)
    table.add_column("STEP")
    table.add_column("TOOL", style=_LINE, no_wrap=True)
    for s in plan:
        label, style = _STATUS.get(s["status"], ("PENDING", "dim"))
        table.add_row(
            str(s["step_id"]),
            f"[{style}]{label}[/{style}]",
            s["label"],
            s.get("intended_tool") or "-",
        )
    _console.print(table)


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

"""
Shared list/table rendering for the readout commands (/tools, /docs, /memory, /models, …).

One vocabulary instead of per-command hand-rolled output: `section()` opens a block with the
`╶── title ───` rule the existing readouts use, and `table()` prints column-aligned rows in the
trace-rail style. Alignment, color, and truncation behave identically across every listing in
the app — the difference between "a pile of prints" and a serious terminal tool.

Cells are plain values or `(text, style)` tuples for a per-cell override (e.g. a risk tier
colored by value). The LAST column flexes into the remaining terminal width and truncates with
`…`; the others fit their widest cell. Degrades to plain aligned text without rich.
"""

from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _RAIL_GLYPH, _RISK,
    _term_width, _truncate,
)

# Friendly style aliases so callers don't import palette internals.
_STYLE_ALIAS = {"dim": _DIM, "accent": _ACCENT, None: "default", "": "default"}


def risk_style(risk: str) -> str:
    """The semantic color for a risk tier (green/yellow/red) — shared with the approval gate so
    a tier reads the same in /tools as it does at the gate."""
    return _RISK.get(risk, "bold red")


def section(title: str, subtitle: str = "") -> None:
    """The `╶── title ───…` rule that opens every readout block, with an optional dim subtitle
    line (counts, the active binding, a hint)."""
    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append(title, style=f"bold {_ACCENT}")
        rule.append(" " + "─" * max(8, 46 - len(title)), style=_DIM)
        _console.print(rule)
        if subtitle:
            sub = Text("  ")
            sub.append(subtitle, style=_DIM)
            _console.print(sub)
    else:
        print(f"  ╶── {title} " + "─" * max(8, 46 - len(title)))
        if subtitle:
            print(f"  {subtitle}")


def _cell(value, default_style: str) -> "tuple[str, str]":
    """Normalize a cell to (text, resolved_style)."""
    if isinstance(value, tuple):
        text, style = value
    else:
        text, style = value, default_style
    text = "" if text is None else str(text)
    return text, _STYLE_ALIAS.get(style, style)


def table(rows, styles=None) -> None:
    """Print column-aligned rows in the rail style.

    `rows` — a list of cell sequences (ragged rows are padded). A cell is a str-able value or a
    `(text, style)` tuple. `styles` — optional per-column default styles ("dim" / "accent" / any
    rich style / None). The last column flexes to the terminal width; the rest fit their widest
    cell. Prints nothing for an empty list — pair with `note()` for the empty case."""
    if not rows:
        return
    styles = list(styles or [])
    ncols = max(len(r) for r in rows)
    styles += [None] * (ncols - len(styles))
    norm = [
        [_cell(r[i] if i < len(r) else "", styles[i]) for i in range(ncols)]
        for r in rows
    ]
    widths = [max(len(row[i][0]) for row in norm) for i in range(ncols)]
    # The last column flexes into what's left of the terminal (rail + 2-space gaps accounted).
    gap = 2
    fixed = 4 + sum(widths[:-1]) + gap * (ncols - 1)
    flex_w = max(8, _term_width() - fixed)

    for row in norm:
        if _RICH:
            line = Text()
            line.append(f"  {_RAIL_GLYPH} ", style=_DIM)
            for i, (text, style) in enumerate(row):
                if i == ncols - 1:
                    line.append(_truncate(text, flex_w), style=style)
                else:
                    line.append(f"{text:<{widths[i]}}" + " " * gap, style=style)
            _console.print(line)
        else:
            parts = []
            for i, (text, _style) in enumerate(row):
                if i == ncols - 1:
                    parts.append(_truncate(text, flex_w))
                else:
                    parts.append(f"{text:<{widths[i]}}")
            print(f"  {_RAIL_GLYPH} " + (" " * gap).join(parts))

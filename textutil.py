"""
Shared text-shaping primitives — the one home for the ellipsis-truncation idiom.

Before this module the `s[: n - 1] + "…"` pattern was hand-rolled in a dozen places (trace
previews, plan labels, steer notes, arg reprs, recap lines — deferred-review #5). Every layer may
import it: it is a leaf with no project imports, so there is no circular-import risk from nodes,
tools, stores, commands, or the TUI.
"""

from __future__ import annotations


def truncate(s: str, n: int) -> str:
    """`s` capped at `n` chars total; a cut is marked with a trailing ellipsis."""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def clip(s, n: int) -> str:
    """One-line preview: collapse all whitespace runs to single spaces, then truncate to `n`."""
    return truncate(" ".join(str(s or "").split()), n)


def fmt_args(args: dict, cap: int) -> str:
    """Render a tool-call kwargs dict as `k='v', k2=3, …` with each value's repr capped, so one
    fat payload (a write_file body) can't bloat a trace line or approval prompt."""
    return ", ".join(f"{k}={truncate(repr(v), cap)}" for k, v in (args or {}).items())

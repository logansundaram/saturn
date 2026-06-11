"""
On-demand readouts (`/context`, `/models`) and the one-off log lines (notes, warnings,
steering acknowledgements, queued-line echoes). All render in the trace-rail style and reuse the
shared meter vocabulary (`_mini_bar`/`_meter_color`), so a gauge reads identically here, in the
status bar, and in the live trace. None of these touch per-turn state.
"""

from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _RAIL_GLYPH,
    _emit, _meter_color, _mini_bar, _rail, _truncate,
)
from .listing import section


# ── system metrics display ───────────────────────────────────────────────────────
def show_system_metrics(metrics) -> None:
    """Display a compact system-resource readout in the trace-rail style. Shares the one meter
    glyph + threshold vocabulary (`_mini_bar` / `_meter_color`) with the status bar and /context,
    so a hot gauge reads identically everywhere; percentages are whole numbers (no false precision)."""

    def _row(label: str, pct: float, detail: str = "") -> None:
        bar = _mini_bar(pct, 20)
        col = _meter_color(pct)
        if _RICH:
            line = _rail()
            line.append(f"{label:<6}", style=_DIM)
            line.append(f"  {bar}", style=col)
            line.append(f"  {pct:>3.0f}%", style=col)
            if detail:
                line.append(f"   {detail}", style=_DIM)
            _console.print(line)
        else:
            print(f"  {_RAIL_GLYPH} {label:<6}  {bar}  {pct:>3.0f}%{'   ' + detail if detail else ''}")

    section("system")

    _row("cpu", metrics.cpu_usage_percent)
    ram_pct = metrics.ram_used_gb / metrics.total_ram_gb * 100
    _row("ram", ram_pct, f"{metrics.ram_used_gb:.1f} / {metrics.total_ram_gb:.1f} GB")
    if metrics.gpu_usage_percent is not None:
        _row("gpu", metrics.gpu_usage_percent)
    if metrics.vram_used_gb is not None and metrics.total_vram_gb is not None:
        vram_pct = metrics.vram_used_gb / metrics.total_vram_gb * 100
        _row("vram", vram_pct, f"{metrics.vram_used_gb:.1f} / {metrics.total_vram_gb:.1f} GB")


# ── context-window readout (the /context command) ──────────────────────────────────
def show_context(window: int, used: int, source: str, per_role: dict[str, int]) -> None:
    """Detailed context-window readout for /context: the active window + where it comes from, a
    wide fill bar for the last measured usage, and the per-role windows. Same trace-rail
    vocabulary as show_system_metrics; the compact form of this fill gauge also rides the live
    status bar during a turn."""
    pct = (used / window * 100) if window else 0.0
    col = _meter_color(pct)
    bar = _mini_bar(pct, width=28)

    section("context")

    if _RICH:
        win = Text("  ")
        win.append("window ", style=_DIM)
        win.append(f"{window:,}", style="default")
        win.append(" tokens", style=_DIM)
        win.append(f"   ({source})", style=_DIM)
        _console.print(win)

        usage = _rail()
        usage.append("usage ", style=_DIM)
        usage.append(f" {bar}", style=col)
        usage.append(f"  {pct:>4.0f}%", style=col)
        usage.append(f"   {used:,} / {window:,}", style=_DIM)
        _console.print(usage)
    else:
        print(f"  window {window:,} tokens   ({source})")
        print(f"  {_RAIL_GLYPH} usage  {bar}  {pct:>4.0f}%   {used:,} / {window:,}")

    if per_role:
        roles_txt = "  ·  ".join(f"{r} {w:,}" for r, w in per_role.items())
        _emit(f"  roles: {roles_txt}")
    _emit("  set with /context <size> (or /context auto for per-model capability)")


# ── model picker / listing ───────────────────────────────────────────────────────
def show_models(models, bindings: dict, active_tier: str, embedder: str,
                *, numbered: bool = False) -> None:
    """Render the locally-installed (Ollama) models plus the live role bindings, in the
    trace-rail style. `models` is a list of `llms.LocalModel`; `bindings` maps role -> model id;
    `embedder` is the active embedder tag. With `numbered=True` each installed row gets a 1-based
    index (the selector the interactive picker reads). A `◂ <roles>` tail marks what each model
    currently drives, so the bindings are visible inline."""
    # role(s) / embedder each installed tag currently serves -> shown as a tail marker.
    serves: dict[str, list[str]] = {}
    for role, mid in (bindings or {}).items():
        serves.setdefault(mid, []).append(role)
    if embedder:
        serves.setdefault(embedder, []).append("embedder")

    all_roles = set(bindings or {})

    def _tail_for(name: str) -> str:
        """Compact 'what this tag drives' marker. Collapses every-role bindings to 'all roles'
        so a model serving the whole loop doesn't spill five role names across the line."""
        entries = serves.get(name, [])
        roles = [e for e in entries if e != "embedder"]
        parts = []
        if roles:
            parts.append("all roles" if all_roles and set(roles) == all_roles
                         else " ".join(roles))
        if "embedder" in entries:
            parts.append("embedder")
        return "  ".join(parts)

    section("models", f"tier {active_tier}  ·  embedder {embedder or '—'}")

    if not models:
        _emit("  (no local models — is the Ollama daemon running? `ollama list`)")
    else:
        for i, m in enumerate(models, start=1):
            meta = " ".join(p for p in (m.parameter_size, m.quantization) if p) or "·"
            tail = _tail_for(m.name)
            idx = f"{i:>2}  " if numbered else ""
            if _RICH:
                line = _rail()
                if numbered:
                    line.append(f"{i:>2}  ", style=_ACCENT)
                line.append(f"{m.name:<26}", style="default")
                line.append(f"{m.size_h:>7}  ", style=_DIM)
                line.append(f"{meta:<14}", style=_DIM)
                if m.is_embedding:
                    line.append("[embed] ", style="yellow")
                if tail:
                    line.append("◂ " + tail, style="green")
                _console.print(line)
            else:
                emb = "[embed] " if m.is_embedding else ""
                bound = ("◂ " + tail) if tail else ""
                print(f"  {_RAIL_GLYPH} {idx}{m.name:<26}{m.size_h:>7}  {meta:<14}{emb}{bound}")

    # Role bindings summary — the full role list, even for roles whose model isn't pulled locally
    # (e.g. a cloud-hybrid anthropic binding won't appear in the installed list above).
    if bindings:
        _emit("  bindings:")
        for role, mid in bindings.items():
            _emit(f"    {role:<12} {mid}")


# ── log lines (startup notices, warnings) ────────────────────────────────────────
def note(msg: str) -> None:
    """A quiet informational line (dim) — e.g. the `@file` attachment notice. Distinct from
    `warn` (yellow), which flags a problem; a note is just neutral context."""
    if _RICH:
        t = Text()
        t.append("  · ", style=_DIM)
        t.append(msg, style=_DIM)
        _console.print(t)
    else:
        print(f"  · {msg}")


def warn(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ! ", style="yellow")
        t.append(msg, style="yellow")
        _console.print(t)
    else:
        print(f"  ! {msg}")


def steer_note(text: str) -> None:
    """Acknowledge a mid-turn steering correction the moment it's captured (Esc with typed text).
    The correction is injected into the running turn at the next step boundary (see plan_gate); this
    is the immediate feedback that it landed, printed above the live status bar."""
    msg = _truncate(text, 80)
    if _RICH:
        t = Text()
        t.append("  ↪ ", style=f"bold {_ACCENT}")
        t.append("steering — applies at the next step: ", style=_ACCENT)
        t.append(msg, style=_DIM)
        _console.print(t)
    else:
        print(f"  ↪ steering — applies at the next step: {msg}")


def echo_queued(line: str) -> None:
    """Echo a type-ahead line as the REPL pulls it off the queue to run, so a query/command the
    user typed while a previous turn was working shows up in the transcript just like a line typed
    live at the `»` prompt (with a quiet `queued` tag to mark where it came from)."""
    if _RICH:
        t = Text()
        t.append("» ", style=f"bold {_ACCENT}")
        t.append(line, style="default")
        t.append("   (queued)", style=_DIM)
        _console.print(t)
    else:
        print(f"» {line}   (queued)")

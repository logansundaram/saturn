"""
On-demand readouts (`/config context`, `/models`) and the one-off log lines (notes, warnings,
steering/pause acknowledgements, queued-line echoes). All render in the trace-rail style and reuse the
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
    glyph + threshold vocabulary (`_mini_bar` / `_meter_color`) with the status bar and /config context,
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


# ── context-window readout (the /config context command) ──────────────────────────────────
def show_context(window: int, used: int, source: str, per_role: dict[str, int]) -> None:
    """Detailed context-window readout for /config context: the active window + where it comes from, a
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
        roles_txt = " · ".join(f"{r} {w:,}" for r, w in per_role.items())
        _emit(f"  roles: {roles_txt}")
    _emit("  set with /config context <size> (or /config context auto for per-model capability)")


# ── model picker / listing ───────────────────────────────────────────────────────
# Fixed column widths for the model table. The NAME column is enforced in both directions (pad
# AND truncate) — `:<26` alone only pads, so one long tag (a `hf.co/…:Q4_K_M` id) pushed the size,
# detail and binding columns out of alignment for every row that followed it.
#
# The detail column is deliberately pad-only: it is the LAST fixed column, so overflowing it
# shifts nothing but the trailing `[embed]` / `◂ roles` annotations, while truncating it would
# drop the calibration state — the fact `/models tier` exists to show. Losing data to tidy a
# trailing annotation is the wrong trade.
_NAME_W = 26
_DETAIL_W = 26


def show_models(models, bindings: dict, active_tier: str, embedder: str,
                *, numbered: bool = False, meta: "dict | None" = None,
                hidden: int = 0) -> None:
    """Render the locally-installed (Ollama) models plus the live role bindings, in the
    trace-rail style. `models` is a list of `llms.LocalModel`; `bindings` maps role -> model id;
    `embedder` is the active embedder tag. With `numbered=True` each installed row gets a 1-based
    index (the selector the interactive picker reads). A `◂ <roles>` tail marks what each model
    currently drives, so the bindings are visible inline. `meta` optionally maps a model id to
    `{"ctx": int, "max_ctx": int, "calibrated": bool}`, appending its runtime/max context window
    and confidence-calibration state to the detail column — the same metrics `/models tier`
    shows, so a tag reads identically in both listings. Windows render compactly (`32k/256k`):
    the raw token counts pushed the bindings tail off the line. `hidden` is how many installed
    models the CALLER filtered out of `models` — with it nonzero, an empty table means "nothing
    offerable", not "daemon down", and the empty-state hint says so instead of misdiagnosing."""
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

    def _ctx_h(n) -> str:
        """A context window in the shortest honest form: 32768 -> 32k, 262144 -> 256k."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            return "?"
        return f"{n // 1024}k" if n >= 1024 else str(n)

    def _meta_bits(name: str) -> list:
        info = meta.get(name) if meta else None
        if not info:
            return []
        bits = []
        if info.get("ctx") or info.get("max_ctx"):
            bits.append(f"{_ctx_h(info.get('ctx'))}/{_ctx_h(info.get('max_ctx'))}")
        if "calibrated" in info:
            bits.append("calibrated" if info["calibrated"] else "uncalibrated")
        return bits

    section("models", f"tier {active_tier} · embedder {embedder or '—'}")

    if not models:
        # Two distinct empty states: the daemon has nothing (or is down), vs. the daemon HAS
        # models but the caller filtered every one out (`hidden`). The daemon-down hint on the
        # second would misdiagnose — the caller's hidden-count note follows, so contradicting it
        # here ("no local models" over a nonzero hidden count) is worse than saying nothing.
        if hidden > 0:
            _emit("  (none of the installed models can be offered here — bind one by name with")
            _emit("   `/models all <tag>` / `/models embedder <tag>`, or switch tiers: /models tier)")
        else:
            _emit("  (no local models — is the Ollama daemon running? `ollama list`)")
    else:
        for i, m in enumerate(models, start=1):
            bits = [p for p in (m.parameter_size, m.quantization) if p]
            bits += _meta_bits(m.name)
            detail = " ".join(bits) or "·"
            tail = _tail_for(m.name)
            idx = f"{i:>2}  " if numbered else ""
            name = _truncate(m.name, _NAME_W)  # see _NAME_W: pad-only left later columns adrift
            if _RICH:
                line = _rail()
                if numbered:
                    line.append(f"{i:>2}  ", style=_ACCENT)
                line.append(f"{name:<{_NAME_W}}", style="default")
                line.append(f"{m.size_h:>7}  ", style=_DIM)
                line.append(f"{detail:<{_DETAIL_W}}", style=_DIM)
                if m.is_embedding:
                    line.append("[embed] ", style="yellow")
                if tail:
                    line.append("◂ " + tail, style="green")
                _console.print(line)
            else:
                emb = "[embed] " if m.is_embedding else ""
                bound = ("◂ " + tail) if tail else ""
                print(f"  {_RAIL_GLYPH} {idx}{name:<{_NAME_W}}{m.size_h:>7}  "
                      f"{detail:<{_DETAIL_W}}{emb}{bound}")

    # Role bindings summary — the full role list, even for roles whose model isn't pulled locally
    # (a bound tag that hasn't been `ollama pull`ed won't appear in the installed list above).
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


def pause_note() -> None:
    """Acknowledge an empty-line Esc the moment it's captured. The plan-review pause lands at the
    next step boundary (see plan_gate), which on a local model can be a long wait — this is the
    immediate feedback that the keypress registered, printed above the live status bar exactly
    like steer_note's steering acknowledgement."""
    if _RICH:
        t = Text()
        t.append("  ⏸ ", style=f"bold {_ACCENT}")
        t.append("pausing for plan review at the next step…", style=_ACCENT)
        _console.print(t)
    else:
        print("  ⏸ pausing for plan review at the next step…")


def freeze_note() -> None:
    """Acknowledge an Esc that froze the streaming answer (interrupt-and-correct) the moment
    it's captured — the stream stops at the next token and the freeze editor opens, but on a
    slow local model that beat can lag the keypress; this is the immediate feedback, printed
    above the live answer region exactly like steer_note/pause_note."""
    if _RICH:
        t = Text()
        t.append("  ✂ ", style=f"bold {_ACCENT}")
        t.append("freezing the answer — the editor opens when the stream stops…", style=_ACCENT)
        _console.print(t)
    else:
        print("  ✂ freezing the answer — the editor opens when the stream stops…")


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

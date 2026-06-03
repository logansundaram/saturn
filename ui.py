"""
CLI rendering for the agent console.

Design target: a serious local-agent console — git status / htop / pytest / a trace viewer,
not a chatbot. Dense, fast, keyboard-first, low-noise, inspectable. The aesthetic comes from
*structure*, not decoration:

  - A dim vertical rail (`│`) carries the execution trace. Consecutive node lines form one
    continuous gutter, so a turn reads as a single inspectable block (the htop/tree feel).
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk tier. Structure
    is dim. Nothing is colored just to look nice — if it has color, it means something.
  - The plan prints **once** as the intended route, then emits a single line per status change
    as steps advance — a log/trace, not a re-rendered panel. This is the transparency surface
    and the main noise source, so it's diffed.
  - The approval gate deliberately breaks out of the rail with a heavy rule. It's a blocking
    safety decision and *should* draw the eye; everything else recedes.
  - A single-line **status bar** is pinned at the bottom of the screen for the duration of a
    turn (`rich.live.Live`): `model · iter · elapsed · tools · ▸node`. The trace lines above it
    keep scrolling normally (rich routes `console.print` and captured `stdout` above the live
    region). It's `transient`, so it vanishes when the turn ends — the scrolling trace is the
    permanent record, the bar is just a live "where are we now" readout. Because `input()` can't
    run inside an active `Live`, the bar is torn down around the `»` prompt, the approval gate,
    and the final response, then restarted as the loop continues.

The agent emits node/plan/state updates; this module is one subscriber that renders them
(SATURDAY_MVP_PLAN.md §6). Swapping it for a Textual/Electron surface needs no graph change.
Degrades to plain ASCII-ish output if `rich` is absent (still UTF-8: stdout is reconfigured in
agent.py, so box-drawing glyphs are safe even on the no-color path).
"""

import time

try:
    from rich.console import Console
    from rich.text import Text
    from rich.live import Live

    _console = Console(highlight=False)
    _RICH = True
except Exception:  # pragma: no cover - fallback path
    _console = None
    _RICH = False

# prompt_toolkit drives the `»` input line so a typed `/command` is highlighted live, character
# by character — valid commands glow cyan, typos go red. Independent of rich: if it's missing we
# fall back to rich's (or plain) input(), just without the live highlight.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.lexers import Lexer as _PTKLexer
    from prompt_toolkit.styles import Style as _PTKStyle

    _PTK = True
except Exception:  # pragma: no cover - fallback path
    _PTK = False


# ── palette ──────────────────────────────────────────────────────────────────
# One accent, semantic status colors, everything else dim. Change here to retheme.
_ACCENT = "cyan"
_RAIL = "grey39"  # the trace gutter; quiet but visible
_DIM = "grey46"

# status -> (glyph, style). Markers carry the state; color reinforces it.
_PLAN = {
    "pending": ("·", _DIM),
    "active": ("▸", f"bold {_ACCENT}"),
    "done": ("✓", "green"),
    "skipped": ("⨯", "grey30 strike"),
}
# risk tier -> style for the approval gate. Read-only never reaches the gate, but kept for parity.
_RISK = {
    "read_only": "green",
    "side_effecting": "yellow",
    "destructive": "bold red",
}

_RAIL_GLYPH = "│"
_NODE_W = 12  # node-name column width, keeps timings aligned
_LABEL_W = 46  # plan-label truncation (keeps step lines on one row at 80 cols)

# ── per-turn state (timing + plan diff). Reset via reset_turn() each turn. ─────
_t_last = None
_plan_seen: dict = {}

# ── live status bar (bottom-pinned) ───────────────────────────────────────────
# `_status` is the live readout the bar renders; `_turn_start` anchors the elapsed
# clock; `_live` holds the active rich.live.Live (None when torn down for input).
# `_model` is captured once in banner() so the bar needs no model passed per turn.
_turn_start = None
_status = {"node": "", "iteration": 0, "tools": 0}
_model = "unknown"
_live = None


class _StatusBar:
    """Renderable for the pinned bar. `__rich__` is re-evaluated on every Live refresh, so the
    elapsed clock ticks even when no node update has fired."""

    def __rich__(self) -> "Text":
        elapsed = time.perf_counter() - _turn_start if _turn_start else 0.0
        n = _status["tools"]
        bar = Text()
        bar.append("  ╶ ", style=_DIM)
        bar.append("saturday", style=f"bold {_ACCENT}")
        for label in (_model, f"iter {_status['iteration']}", _fmt_dur(elapsed).strip(),
                      f"{n} tool{'' if n == 1 else 's'}"):
            bar.append("  ·  ", style=_DIM)
            bar.append(label, style="default")
        if _status["node"]:
            bar.append("  ·  ", style=_DIM)
            bar.append(f"▸ {_status['node']}", style=f"bold {_ACCENT}")
        return bar


def _live_start() -> None:
    """Pin a fresh status bar at the bottom. No-op without rich or if one is already running.
    `transient=True` erases the bar on stop (the scrolling trace stays); rich's default
    stdout/stderr redirect keeps node `print()`s flowing above the live region."""
    global _live
    if not _RICH or _live is not None:
        return
    _live = Live(_StatusBar(), console=_console, transient=True,
                 auto_refresh=True, refresh_per_second=4)
    _live.start()


def _live_stop() -> None:
    """Tear the bar down (before any input()) so it never fights a blocking prompt."""
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def _live_refresh() -> None:
    if _live is not None:
        _live.refresh()


def reset_turn() -> None:
    """Call once at the start of each user turn: resets node timing + plan-diff state and
    starts the bottom-pinned status bar for the turn."""
    global _t_last, _plan_seen, _turn_start, _status
    _t_last = time.perf_counter()
    _turn_start = _t_last
    _plan_seen = {}
    _status = {"node": "", "iteration": 0, "tools": 0}
    _live_start()


# ── small helpers ──────────────────────────────────────────────────────────────
def _emit(text) -> None:
    if _RICH:
        _console.print(text)
    else:
        print(text if isinstance(text, str) else str(text))


def _rail(style: str = _RAIL) -> "Text":
    t = Text()
    t.append(f"  {_RAIL_GLYPH} ", style=style)
    return t


def _fmt_dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:>4.0f}ms"
    if seconds < 60:
        return f"{seconds:>5.2f}s"
    return f"{seconds / 60:>5.1f}m"


def _fmt_args(args: dict, cap: int = 48) -> str:
    parts = []
    for k, v in (args or {}).items():
        r = repr(v)
        if len(r) > cap:
            r = r[: cap - 1] + "…"
        parts.append(f"{k}={r}")
    return ", ".join(parts)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ── startup banner ─────────────────────────────────────────────────────────────
def banner(model: str, n_tools: int, n_docs: int, db_path: str) -> None:
    """Compact two-line header. Reads like a tool's startup line, not a splash screen."""
    global _model
    _model = model  # captured here so the live status bar needs no model passed per turn
    if _RICH:
        head = Text()
        head.append("saturday", style=f"bold {_ACCENT}")
        head.append(".ai", style=_DIM)
        for label in (model, f"{n_tools} tools", f"{n_docs} docs"):
            head.append("  ·  ", style=_DIM)
            head.append(label, style="default")
        _console.print(head)
        sub = Text()
        sub.append(f"sqlite:{db_path}", style=_DIM)
        sub.append("   ", style=_DIM)
        sub.append("/help", style=_ACCENT)
        sub.append(" for commands", style=_DIM)
        _console.print(sub)
    else:
        print(f"saturday.ai  ·  {model}  ·  {n_tools} tools  ·  {n_docs} docs")
        print(f"sqlite:{db_path}   /help for commands")


# ── input prompt ───────────────────────────────────────────────────────────────
# Live highlight for the `»` line: a `/token` is colored by how it matches the command set, so
# a typo never blends in with a real command. Valid -> cyan, a prefix of some command (mid-type)
# -> yellow, anything else -> red. Args after the token stay dim. Built only when prompt_toolkit
# is present; the palette mirrors the rest of ui.py (cyan accent, semantic status colors).
if _PTK:
    _PTK_STYLE = _PTKStyle.from_dict({
        "prompt": "ansicyan bold",
        "cmd.valid": "ansicyan bold",
        "cmd.partial": "ansiyellow",
        "cmd.unknown": "ansired bold",
        "cmd.args": "ansibrightblack",
    })

    class _CommandLexer(_PTKLexer):
        """Colors the first `/token` of the line against a known-command set, live as it's typed.
        Only the command token is styled; normal (non-slash) turns render plain."""

        def __init__(self, names):
            self._names = names  # canonical names + aliases, lowercased, no leading slash

        def _style_for(self, key: str) -> str:
            if key in self._names:
                return "class:cmd.valid"
            if not key or any(n.startswith(key) for n in self._names):
                return "class:cmd.partial"  # lone "/" or still typing a real command
            return "class:cmd.unknown"      # a typo — make it loud

        def lex_document(self, document):
            text = document.text

            def get_line(_lineno):
                stripped = text.lstrip()
                if not stripped.startswith("/"):
                    return [("", text)]
                lead = text[: len(text) - len(stripped)]  # preserve leading whitespace verbatim
                body = stripped[1:]
                cut = len(body)
                for i, ch in enumerate(body):
                    if ch.isspace():
                        cut = i
                        break
                token, args = body[:cut], body[cut:]
                frags = []
                if lead:
                    frags.append(("", lead))
                frags.append((self._style_for(token.lower()), "/" + token))
                if args:
                    frags.append(("class:cmd.args", args))
                return frags

            return get_line

    _ptk_session = None  # one PromptSession for the process -> free line history across turns


def prompt(command_names=None) -> str:
    """Read the `»` input line. With prompt_toolkit + a `command_names` set, a typed `/command`
    is highlighted live (valid=cyan, typo=red); otherwise falls back to rich/plain input.
    Returns the raw line (slash-command detection happens upstream)."""
    _live_stop()  # never read a line under an active Live (also clears a bar left by an error)
    if _PTK and command_names is not None:
        global _ptk_session
        if _ptk_session is None:
            _ptk_session = PromptSession()
        return _ptk_session.prompt(
            [("class:prompt", "» ")],
            lexer=_CommandLexer(set(command_names)),
            style=_PTK_STYLE,
        )
    if _RICH:
        return _console.input(f"[bold {_ACCENT}]»[/] ")
    return input("» ")


# ── execution trace ─────────────────────────────────────────────────────────────
def show_node(node: str, delta: dict | None = None) -> None:
    """One trace line per node execution: `│ <node>   <elapsed>`, with the elapsed time
    measured since the previous node emitted (htop-style). The `tools` node also surfaces the
    tool names it ran, so the trace shows *what* happened, not just that something did."""
    global _t_last
    now = time.perf_counter()
    dur = now - _t_last if _t_last is not None else 0.0
    _t_last = now

    extra = ""
    if delta:
        called = delta.get("tools_called") or []
        if called:
            extra = ", ".join(called)
        # Feed the pinned status bar: latest node, running tool count, agent iteration.
        _status["tools"] += len(called)
        if "iteration" in delta:
            _status["iteration"] = delta["iteration"]
    _status["node"] = node

    if _RICH:
        line = _rail()
        line.append(f"{node:<{_NODE_W}}", style="default")
        line.append(f"{_fmt_dur(dur):>7}", style=_DIM)
        if extra:
            line.append(f"  {extra}", style=_ACCENT)
        _console.print(line)
    else:
        tail = f"  {extra}" if extra else ""
        print(f"  {_RAIL_GLYPH} {node:<{_NODE_W}}{_fmt_dur(dur):>7}{tail}")

    _live_refresh()  # repaint the bar with the new node/iter/tools immediately


def _plan_line(step: dict, *, show_tool: bool) -> "Text | str":
    status = step.get("status", "pending")
    glyph, style = _PLAN.get(status, _PLAN["pending"])
    label = _truncate(str(step.get("label", "")), _LABEL_W)
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")

    if _RICH:
        line = _rail()
        line.append("  ", style=_RAIL)  # nest steps under the node
        line.append(f"{glyph} ", style=style)
        line.append(f"{str(sid):>2}  ", style=_DIM)
        line.append(label, style=style if status in ("active", "skipped") else "default")
        if show_tool and tool:
            line.append(f"  ::{tool}", style=_DIM)
        return line
    tooltxt = f"  ::{tool}" if (show_tool and tool) else ""
    return f"  {_RAIL_GLYPH}   {glyph} {str(sid):>2}  {label}{tooltxt}"


def show_plan(plan) -> None:
    """First call this turn: print the whole plan as the intended route (with tools).
    Later calls: print only the steps whose status changed — one line each, like a trace.
    This keeps the plan transparent without re-rendering a panel on every node update."""
    global _plan_seen
    if not plan:
        return

    first_render = not _plan_seen
    for step in plan:
        sid = step.get("step_id")
        status = step.get("status", "pending")
        if first_render:
            _emit(_plan_line(step, show_tool=True))
            _plan_seen[sid] = status
        elif _plan_seen.get(sid) != status:
            _emit(_plan_line(step, show_tool=False))
            _plan_seen[sid] = status


# ── approval gate (the one place that gets to shout) ─────────────────────────────
def ask_approval(value: dict) -> bool:
    """Compact, high-signal gate. Heavy rule + risk-colored tier so it breaks out of the dim
    trace rail. Returns True to approve the whole batch."""
    global _t_last
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    _live_stop()  # the gate blocks on input(); the bar can't be live while it does

    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("approval required", style=f"bold {_ACCENT}")
        top.append(" " + "━" * 36, style="bold")
        _console.print(top)
        for tc in tool_calls:
            risk = str(tc.get("risk", "destructive"))
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append(f"{risk:<14} ", style=_RISK.get(risk, "bold red"))
            row.append(f"{tc.get('name')}", style="default")
            row.append(f"({_fmt_args(tc.get('args', {}))})", style=_DIM)
            _console.print(row)
        resp = _console.input("  [bold]┗━[/] approve? [bold]y/N[/] » ").strip().lower()
    else:
        print("  ┏━ approval required " + "━" * 30)
        for tc in tool_calls:
            print(f"  ┃ [{tc.get('risk')}] {tc.get('name')}({_fmt_args(tc.get('args', {}))})")
        resp = input("  ┗━ approve? y/N » ").strip().lower()

    _t_last = time.perf_counter()  # don't bill the human's decision time to the next node
    _live_start()  # the turn continues (tools -> agent -> …); re-pin the bar
    return resp in ("y", "yes")


# ── final answer ─────────────────────────────────────────────────────────────────
def response(text: str) -> None:
    """The payload. Leaves the trace rail (un-indented, copy-pasteable) behind a short labeled
    rule, so the answer is visually distinct from the trace above it."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append("response", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
        # markup=False: the answer is arbitrary model output; bracketed tokens like `list[str]`,
        # citations `[1]`, or paths `[/etc/hosts]` must not be parsed as Rich tags (they get
        # stripped, or raise MarkupError and kill the turn). highlight=False already on _console.
        _console.print(text, markup=False)
    else:
        print("  ╶── response " + "─" * 36)
        print(text)


# ── log lines (startup notices, warnings) ────────────────────────────────────────
def warn(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ! ", style="yellow")
        t.append(msg, style="yellow")
        _console.print(t)
    else:
        print(f"  ! {msg}")

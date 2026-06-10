"""
The final answer: `response` renders a completed (non-streamed) answer as real markdown under a
labeled rule plus a one-line receipt; `ResponseStream` renders the synthesize node's token-by-token
stream live (a transient, screen-bounded tail that always erases cleanly) then re-renders the whole
answer once on finish. Both end on the same receipt — the permanent echo of the transient status bar.
"""

import time

from . import _base
from ._base import (
    Live, Markdown, Text, _console, _RICH,
    _ACCENT, _DIM, _fmt_dur,
)
from .statusbar import _live_stop


def _turn_summary_parts() -> list[str]:
    """The post-response receipt: a permanent one-line echo of the (transient) status bar's run
    stats — the bar vanishes when the turn ends, so this is what survives in the scrollback."""
    elapsed = time.perf_counter() - _base._turn_start if _base._turn_start else 0.0
    status = _base._status
    n = status["tools"]
    parts = [f"{status['iteration']} iter", f"{n} tool{'' if n == 1 else 's'}",
             _fmt_dur(elapsed).strip()]
    if status["tok_per_sec"] > 0:
        parts.append(f"{status['tok_per_sec']:.0f} tok/s")
    window = status["ctx_window"]
    if window and status["ctx_used"]:
        parts.append(f"ctx {status['ctx_used'] / window * 100:.0f}%")
    return parts


def response(text: str) -> None:
    """The payload. Leaves the trace rail behind a short labeled rule and renders the answer as
    real markdown — headings, bold, lists, and fenced code with syntax highlighting — so it reads
    like a finished answer, not a log line. Falls back to plain text if markdown rendering raises
    (arbitrary model output), and to plain print without rich."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    if _RICH:
        _console.print()  # part the answer from the trace rail above it
        rule = Text()
        rule.append("  ── ", style=_DIM)
        rule.append("response", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
        _console.print()  # let the answer breathe beneath its rule
        try:
            # Markdown parses markdown, not Rich console markup, so bracketed tokens like
            # `list[str]` or citations `[1]` are safe literal text here.
            _console.print(Markdown(text))
        except Exception:
            # Arbitrary model output can occasionally trip the markdown parser; never lose the
            # answer over formatting. markup=False so brackets aren't eaten as Rich tags.
            _console.print(text, markup=False)
        _console.print()  # let the answer breathe before the receipt
        _console.print(Text("  ╶ " + " · ".join(_turn_summary_parts()), style=_DIM))
        _console.print()  # trailing whitespace before the next prompt
    else:
        print()
        print("  ── response " + "─" * 36)
        print()
        print(text)
        print()
        print("  ╶ " + " · ".join(_turn_summary_parts()))
        print()


# ── streaming the final answer ─────────────────────────────────────────────────────
# The synthesize node streams its answer token-by-token (LangGraph messages mode -> run_turn ->
# on_token). ResponseStream renders those tokens live, then finishes with the same finished look as
# `response`. The hard part in a terminal is long output: a growing Live region that outgrows the
# screen can't be erased cleanly. So during streaming we show a *transient* Live of only the last
# screenful (a bounded tail — see `_tail`), which always fits and so always erases cleanly; on
# `finish` we tear that down and render the WHOLE answer once as real markdown (+ the receipt). The
# permanent scrollback record is that final rendered block, not the transient tail. Without rich we
# just type the raw tokens out incrementally. If the model yields no tokens, `started` stays False
# and the caller renders via `response` instead.
class ResponseStream:
    def __init__(self) -> None:
        self._chars: list[str] = []
        self._live = None
        self._started = False
        self._last = 0.0  # last repaint time (throttle)

    @property
    def started(self) -> bool:
        return self._started

    def feed(self, text: str) -> None:
        """Append a streamed answer token; opens the response section on the first one."""
        if not text:
            return
        if not self._started:
            self._begin()
        self._chars.append(text)
        if self._live is not None:
            now = time.perf_counter()
            if now - self._last >= 0.06:  # throttle (~16/s) so granular tokens don't thrash the live
                self._live.update(self._tail(), refresh=True)
                self._last = now
        else:  # plain (no-rich) path: just type it out
            print(text, end="", flush=True)

    def _begin(self) -> None:
        self._started = True
        _live_stop()  # drop the turn's status bar — the answer takes over the bottom of the screen
        if _RICH:
            _console.print()  # part the answer from the trace rail above it
            rule = Text()
            rule.append("  ── ", style=_DIM)
            rule.append("response", style=f"bold {_ACCENT}")
            rule.append(" " + "─" * 40, style=_DIM)
            _console.print(rule)
            _console.print()
            # transient + a screen-bounded tail => the live region always fits, so stop() erases it
            # cleanly no matter how long the answer runs. Manual refresh (throttled in feed).
            self._live = Live(console=_console, transient=True, auto_refresh=False)
            self._live.start()
        else:
            print()
            print("  ── response " + "─" * 36)
            print()

    def _tail(self) -> "Text":
        """The last screenful of the answer-so-far as plain Text, bounded to at most `rows` VISUAL
        lines so the transient live region always fits on screen — which is what lets `stop()` erase
        it cleanly before the full answer is re-rendered. The bound must count visual rows, which
        means accounting for BOTH hard newlines AND soft wrapping: a raw character budget undercounts
        badly when the answer is many short lines (lists, headings, code, blanks), because each
        newline ends a line early, so the same budget spans far more rows than the screen has. The
        region then scrolls off the top and the transient erase corrupts the final render — eating
        the first lines of the answer (the data is fine; only the on-screen handoff breaks)."""
        rows = max(4, (_console.size.height or 24) - 6)
        cols = max(20, _console.size.width or 80)
        avail = max(1, cols - 2)  # room for text after the 2-space indent below
        lines = "".join(self._chars).split("\n")
        # Walk from the bottom up, accumulating physical lines until their WRAPPED height fills the
        # row budget, so the rendered region can't exceed the screen no matter the line lengths.
        chosen: list[str] = []
        used = 0
        for ln in reversed(lines):
            h = max(1, -(-len(ln) // avail))  # ceil(len / avail); a blank line is still one row
            if used + h > rows:
                if chosen:
                    break
                ln = ln[-(rows * avail):]  # a lone line taller than the screen: keep its tail only
            chosen.append(ln)
            used += h
            if used >= rows:
                break
        chosen.reverse()
        t = Text()
        for i, ln in enumerate(chosen):
            if i:
                t.append("\n")
            t.append("  ")
            t.append(ln)
        return t

    def finish(self, final_text: "str | None" = None) -> None:
        """Close out a successful turn: tear down the live tail, render the full answer once as
        markdown, then the one-line receipt. Mirrors `response`'s final look exactly.

        `final_text`, when given, is rendered instead of the streamed chars — the loop passes the
        RECORDED final message, which may carry mechanically-appended trailers the token stream
        never saw (the citations Sources footer from synthesize). Falls back to the streamed text
        when absent/empty so a caller without the final message loses nothing."""
        text = final_text if isinstance(final_text, str) and final_text else "".join(self._chars)
        if self._live is not None:
            self._live.stop()  # transient: erases the streaming tail
            self._live = None
        if _RICH:
            try:
                _console.print(Markdown(text))
            except Exception:
                _console.print(text, markup=False)
            _console.print()
            _console.print(Text("  ╶ " + " · ".join(_turn_summary_parts()), style=_DIM))
            _console.print()
        else:
            print()  # close the typed-out line
            # The plain path typed the streamed tokens out already — print only what the recorded
            # final text appends beyond them (e.g. the Sources footer), never the whole thing twice.
            streamed = "".join(self._chars).rstrip()
            if text.rstrip() != streamed and text.startswith(streamed):
                print(text[len(streamed):].strip("\n"))
                print()
            print("  ╶ " + " · ".join(_turn_summary_parts()))
            print()

    def abort(self) -> None:
        """Tear down the live tail without a final render — a failed/cancelled turn. The transient
        Live erases the partial text; the caller surfaces the error (`warn`) separately."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

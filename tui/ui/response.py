"""
The final answer: `response` renders a completed (non-streamed) answer as real markdown under a
labeled rule plus a one-line receipt; `ResponseStream` renders the agent node's token-by-token
stream live (a transient, screen-bounded tail that always erases cleanly) then re-renders the whole
answer once on finish. Both end on the same receipt — the permanent echo of the transient status bar.
"""

import time

from textutil import split_sources_footer

from . import _base
from ._base import (
    Constrain, Live, Markdown, Padding, Text, _console,
    _DIM, _fmt_dur, _term_width,
)
from .statusbar import _live_stop
from .listing import section


def _stats_parts() -> list[str]:
    """The run-stats half of the receipt: a permanent echo of the (transient) status bar — the
    bar vanishes when the turn ends, so this is what survives in the scrollback. Always dim, and
    deliberately short: time · iterations · tools · rate (the context gauge stays a live-bar /
    `/context` fact — stale by the time the answer lands)."""
    elapsed = time.perf_counter() - _base._turn_start if _base._turn_start else 0.0
    status = _base._status
    n = status["tools"]
    parts = [_fmt_dur(elapsed).strip(), f"{status['iteration']} iter",
             f"{n} tool{'' if n == 1 else 's'}"]
    if status["tok_per_sec"] > 0:
        parts.append(f"{status['tok_per_sec']:.0f} tok/s")
    return parts


def _trust_spans() -> list:
    """The trust half of the receipt (`runtime.receipt`, default on), deviation-only: the turn's
    egress summary, blocked attempts, and how many calls faced the approval gate — EMPTY for a
    calm local turn, so the receipt is then just the dim run stats (receipt.turn_spans — which
    also guards an unusable turn mark by rendering the honest `egress unknown`, never silence
    over a slice that may be hiding sends)."""
    try:
        from trust import receipt

        if receipt.enabled():
            return receipt.turn_spans(receipt.turn_mark(), _base._status.get("gates", 0))
    except Exception:
        pass  # the receipt is additive — it must never cost the stats line
    return []


# Trust-span kind -> semantic style: the same yellow/red vocabulary the posture line colors the
# identical facts with — a boundary crossing must not render with the weight of a tok/s gauge.
# `gated` and `untracked` stay dim (counts, not signals — the human approved those calls);
# `unknown` is yellow (the slice may hide a send). A calm local turn emits no trust spans at all.
_TRUST_STYLE = {"sent": "yellow", "blocked": "bold red", "untracked": _DIM,
                "gated": _DIM, "unknown": "yellow", "human": "cyan",
                "uncertain": "red"}

# One-time discovery hint (receipt.take_hint — sentinel-backed, once per install): the
# post-first-answer line teaching the inspection surfaces.
_FIRST_ANSWER_HINT = "see this run: /trace · what left your machine: /policy egress"


def _print_sources(entries: list[str]) -> None:
    """The Sources footer — the receipt of every tool call and document the turn gathered —
    printed dim at the answer's 2-space indent, the line text identical to the recorded footer
    (the markdown renderer would collapse its lines into one paragraph)."""
    _console.print()
    _console.print(Text("  Sources:", style=_DIM))
    for ln in entries:
        _console.print(Text("  " + ln, style=_DIM))


def _print_receipt() -> None:
    """The one-line receipt under every answer: the trust segment leads as semantically-colored
    spans WHEN the turn deviated (what was sent / blocked / gated — a calm local turn emits
    none), then the dim run stats."""
    stats = _stats_parts()
    spans = _trust_spans()
    line = Text("  ╶ ", style=_DIM)
    for i, (text, kind) in enumerate(spans):
        if i:
            line.append(" · ", style=_DIM)
        line.append(text, style=_TRUST_STYLE.get(kind, _DIM))
    if spans and stats:
        line.append(" · ", style=_DIM)
    line.append(" · ".join(stats), style=_DIM)
    _console.print(line)


def _first_answer_hint() -> None:
    """After the very first answer of an install, one dim discovery line under the receipt
    pointing at the inspection surfaces. Never repeats (sentinel via receipt.take_hint)."""
    try:
        from trust import receipt

        due = receipt.take_hint("first_answer")
    except Exception:
        due = False
    if not due:
        return
    t = Text()
    t.append("  · ", style=_DIM)
    t.append(_FIRST_ANSWER_HINT, style=_DIM)
    _console.print(t)


# The answer's measure: indented to the app's 2-space rhythm and capped so prose stays readable
# on a wide terminal (a full-bleed 200-column paragraph is harder to read than a ~100-column one).
_BODY_WIDTH = 100


def _constrained(renderable):
    """Present a live-region renderable exactly as the finished answer is presented — the same
    2-space indent and the same measure, `min(_term_width(), _BODY_WIDTH)`, which is what
    `Constrain(x, _BODY_WIDTH)` yields (it takes the smaller of the cap and the space available).
    `Live` has no per-update width, so without the constraint the streaming tail would wrap at
    the full terminal width and every line break would move the instant `finish()` ran.

    The `Padding` is the other half: the finished body pads every VISUAL row
    (`_print_markdown_body`), so its text wraps at width - 2; the tail must indent the same way or
    soft-wrapped continuation rows break in different places. Indenting here, once, keeps
    `_tail()` free of geometry it would have to keep in step by hand. On any failure the
    renderable passes through."""
    try:
        return Constrain(Padding(renderable, (0, 0, 0, 2)), _BODY_WIDTH)
    except Exception:
        return renderable


def _print_markdown_body(body: str) -> None:
    """Render the answer body as real markdown at the app's 2-space indent, measure-capped.
    Falls back to plain text if the markdown parser trips on arbitrary model output — never lose
    the answer over formatting (Text, so brackets are never eaten as Rich markup)."""
    width = min(_term_width(), _BODY_WIDTH)
    try:
        # Markdown parses markdown, not Rich console markup, so bracketed tokens like
        # `list[str]` or citations `[1]` are safe literal text here. hyperlinks=False: a link
        # prints as `text (address)` — a terminal hyperlink (OSC 8) would show the text and
        # open the address, which the answer's author chooses and the reader never sees.
        _console.print(Padding(Markdown(body, hyperlinks=False), (0, 0, 0, 2)), width=width)
    except Exception:
        _console.print(Padding(Text(body), (0, 0, 0, 2)), width=width)


def response(text: str) -> None:
    """The payload. Leaves the trace rail behind a short labeled rule and renders the answer as
    real markdown — headings, bold, lists, and fenced code with syntax highlighting — so it reads
    like a finished answer, not a log line. The mechanical Sources footer, when present, renders
    as its own dim block instead of through the markdown body. Falls back to plain text if
    markdown rendering raises (arbitrary model output)."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    section("response")  # parts the answer from the trace rail above it
    _console.print()  # let the answer breathe beneath its rule
    _final_render(text)


def _final_render(text: str) -> None:
    """THE final-answer tail (sources split → markdown body → dim Sources → receipt →
    first-answer hint), shared by `response()` and ResponseStream.finish() so streamed and
    non-streamed answers can never drift apart. The recorded message is never altered: the
    `Sources:` footer only routes to the dim renderer instead of the markdown one (which would
    collapse its lines into a single paragraph)."""
    prose, src_lines = split_sources_footer(text)
    _print_markdown_body(prose if src_lines else text)
    if src_lines:
        _print_sources(src_lines)
    _console.print()  # let the answer breathe before the receipt
    _print_receipt()
    _first_answer_hint()
    _console.print()  # trailing whitespace before the next prompt


# ── streaming the final answer ─────────────────────────────────────────────────────
# The agent node streams its answer token-by-token (LangGraph messages mode -> run_turn ->
# on_token). ResponseStream renders those tokens live, then finishes with the same finished look as
# `response`. The hard part in a terminal is long output: a growing Live region that outgrows the
# screen can't be erased cleanly. So during streaming we show a *transient* Live of only the last
# screenful (a bounded tail — see `_tail`), which always fits and so always erases cleanly; on
# `finish` we tear that down and render the WHOLE answer once as real markdown (+ the receipt). The
# permanent scrollback record is that final rendered block, not the transient tail. If the model
# yields no tokens, `started` stays False and the caller renders via `response` instead.
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
        now = time.perf_counter()
        if self._live is not None and now - self._last >= 0.06:
            # throttle (~16/s) so granular tokens don't thrash the live
            self._live.update(_constrained(self._tail()), refresh=True)
            self._last = now

    def _begin(self) -> None:
        self._started = True
        _live_stop()  # drop the turn's status bar — the answer takes over the bottom of the screen
        section("response")  # parts the answer from the trace rail above it
        _console.print()
        # transient + a screen-bounded tail => the live region always fits, so stop() erases it
        # cleanly no matter how long the answer runs. Manual refresh (throttled in feed).
        self._live = Live(console=_console, transient=True, auto_refresh=False)
        self._live.start()

    def _tail(self) -> "Text":
        """The last screenful of the answer-so-far as Text, bounded to at most `rows` VISUAL
        lines so the transient live region always fits on screen — which is what lets `stop()` erase
        it cleanly before the full answer is re-rendered. The bound must count visual rows, which
        means accounting for BOTH hard newlines AND soft wrapping: a raw character budget undercounts
        badly when the answer is many short lines (lists, headings, code, blanks), because each
        newline ends a line early, so the same budget spans far more rows than the screen has. The
        region then scrolls off the top and the transient erase corrupts the final render — eating
        the first lines of the answer (the data is fine; only the on-screen handoff breaks).

        The row budget is computed against `_BODY_WIDTH` because that is the measure the tail is
        actually rendered at (`_constrained`, and the same one `finish()` uses) — counting rows
        against the full terminal width would undercount them on a wide terminal and let the live
        region outgrow the screen, which is exactly what breaks the transient erase.

        Returns the text UNINDENTED: `_constrained` applies the 2-space indent to every visual row,
        which is the only way it matches the finished body (see there)."""
        rows = max(4, (_console.size.height or 24) - 6)
        cols = min(max(20, _console.size.width or 80), _BODY_WIDTH)
        avail = max(1, cols - 2)  # room for text after _constrained's 2-space indent
        joined = "".join(self._chars)
        lines = joined.split("\n")
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
        return Text("\n".join(chosen))

    def finish(self, final_text: "str | None" = None) -> None:
        """Close out a successful turn: tear down the live tail, render the full answer once as
        markdown, then the one-line receipt. Mirrors `response`'s final look exactly.

        `final_text`, when given, is rendered instead of the streamed chars — the loop passes the
        RECORDED final message, which may carry mechanically-appended trailers the token stream
        never saw (the Sources receipt, the incidents note). Falls back to the streamed text
        when absent/empty so a caller without the final message loses nothing."""
        text = final_text if isinstance(final_text, str) and final_text else "".join(self._chars)
        if self._live is not None:
            self._live.stop()  # transient: erases the streaming tail
            self._live = None
        _live_stop()
        _final_render(text)

    def discard(self) -> None:
        """Drop a stream that turned out NOT to be the answer (the model prefaced a tool call
        with text): tear the transient tail down and forget the chars, so the real answer opens
        its own `── response` section later."""
        self.abort()
        self._chars = []
        self._started = False

    def abort(self) -> None:
        """Tear down the live tail without a final render — a failed/cancelled turn. The transient
        Live erases the partial text; the caller surfaces the error (`warn`) separately."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

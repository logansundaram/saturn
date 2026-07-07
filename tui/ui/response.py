"""
The final answer: `response` renders a completed (non-streamed) answer as real markdown under a
labeled rule plus a one-line receipt; `ResponseStream` renders the synthesize node's token-by-token
stream live (a transient, screen-bounded tail that always erases cleanly) then re-renders the whole
answer once on finish. Both end on the same receipt — the permanent echo of the transient status bar.
"""

import re
import time

from . import _base
from ._base import (
    Live, Markdown, Padding, Text, _console, _RICH,
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


# Trust-span kind -> semantic style: the same yellow/red vocabulary the Glass Box colors the
# identical facts with — a boundary crossing must not render with the weight of a tok/s gauge.
# `gated` stays dim (a count, not a signal — the human already approved those); `unknown` is
# yellow (the slice may hide a send, like the Glass Box's truncated-record caveat). No `local`
# kind anymore: a calm local turn emits no trust spans at all (deviation-only, 2026-07-06).
_TRUST_STYLE = {"sent": "yellow", "blocked": "bold red",
                "gated": _DIM, "unknown": "yellow", "human": "cyan"}

# One-time discovery hints (receipt.take_hint — sentinel-backed, once per install):
# the post-first-answer line teaching the inspection surfaces, and the receipt tail pointing at
# the Glass Box the first time a receipt actually shows egress or a gated count.
_FIRST_ANSWER_HINT = ("see this run: /trace · answer provenance: /glass · "
                      "what left your machine: /privacy egress")
_GLASS_HINT = "/glass: answer provenance"


# ── per-turn answer provenance (the Glass Box, ambient) ───────────────────────────────────────
# The loop hands the finished turn's state here (set_turn_provenance) just before the final
# render; finish/response pop it to color the Sources footer by source trust — the Glass Box's
# headline facts on every answer, no /glass required. Pop-on-read: a stale box can never paint a
# later answer (error/Ctrl-C turns never set one; a consumer that doesn't render still clears it).
_turn_glass = None

# ── per-turn correction provenance (interrupt-and-correct) ─────────────────────────────────────
# The same pop-on-read pattern for the answer buffer (core/provenance.py): when the finished turn
# carries human-authored spans (the user froze the stream and corrected it), the final render
# marks those characters distinctly and the receipt counts the corrections. The marking is for
# the HUMAN and the audit trail only — the model saw clean text.
_turn_buffer = None

# What human-authored characters render as — the one correction style, shared with the freeze
# editor's tail (tui/ui/correction.py) and semantically cyan: the human acted here.
_HUMAN_STYLE = "bold cyan underline"


def set_turn_buffer(state) -> None:
    """Stash the finished turn's answer buffer for the final render (pop-on-read, like
    set_turn_provenance): only a completed buffer that actually carries human edits is kept —
    an uncorrected turn renders exactly as before."""
    global _turn_buffer
    _turn_buffer = None
    try:
        from core import provenance

        buf = (state or {}).get("answer_buffer")
        if provenance.corrected(buf) and buf.get("state") == "complete":
            _turn_buffer = buf
    except Exception:
        _turn_buffer = None


def _pop_turn_buffer():
    global _turn_buffer
    buf, _turn_buffer = _turn_buffer, None
    return buf

# A footer entry line as synthesize writes it: `  [n] label`.
_SOURCE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s")


def set_turn_provenance(state) -> None:
    """Build the live Glass Box for the turn that just finished (trust.glassbox.build_live — the
    same mark-guarded egress contract `/glass` applies) so the answer render can color the
    Sources footer by source trust natively. Best-effort and additive: any failure leaves the
    answer rendering exactly as it would without provenance."""
    global _turn_glass
    _turn_glass = None
    try:
        from trust import glassbox

        gb = glassbox.build_live(state, gated=_base._status.get("gates", 0))
        if gb.sources:
            _turn_glass = gb
    except Exception:
        _turn_glass = None


def _pop_turn_provenance():
    global _turn_glass
    gb, _turn_glass = _turn_glass, None
    return gb


def _split_sources(text: str) -> "tuple[str, list[str] | None]":
    """Split a recorded answer into (prose, footer_lines) when it ends with the mechanical
    `Sources:` block synthesize appends — a `Sources:` line followed only by `[n] label` lines.
    Returns (text, None) for anything else, and the whole text renders exactly as before. The
    recorded message is never altered; this only routes the footer to the trust-colored renderer
    instead of the markdown one (which collapsed its lines into a single paragraph anyway)."""
    i = text.rfind("\nSources:")
    if i == -1:
        return text, None
    lines = [ln for ln in text[i + 1:].splitlines() if ln.strip()]
    if not lines or lines[0].strip() != "Sources:":
        return text, None
    entries = lines[1:]
    if not entries or not all(_SOURCE_LINE_RE.match(ln) for ln in entries):
        return text, None
    return text[:i].rstrip(), entries


def _facet_annotation(facet) -> tuple[str, str, str]:
    """(glyph, style, note) for one source's trust facet — the same green/yellow vocabulary
    the Glass Box renders, compacted for the footer."""
    if facet.origin == "network" or not facet.trusted:
        note = "web" if facet.origin == "network" else "untrusted origin"
        if facet.injection_flagged:
            note += " · injection-flagged"
        return "◐", "yellow", note
    return "✓", "green", "local"


def _print_sources(entries: list[str], gb) -> None:
    """The Sources footer, rendered natively with per-source trust coloring: green = local +
    trusted, yellow = network / untrusted origin, red = a span of that source reached the answer
    verbatim. The line text is identical to the recorded footer; only an annotation is appended
    (and the block sits at the answer's 2-space indent). Without provenance (e.g. a /retry
    re-render) the block prints dim — never colored by guesswork."""
    by_n = {s.n: s for s in (gb.sources if gb is not None else [])}
    if _RICH:
        _console.print()
        _console.print(Text("  Sources:", style=_DIM))
        for ln in entries:
            m = _SOURCE_LINE_RE.match(ln)
            facet = by_n.get(int(m.group(1))) if m else None
            if facet is None:
                _console.print(Text("  " + ln, style=_DIM))
                continue
            glyph, style, note = _facet_annotation(facet)
            head, _, rest = ln.partition("]")
            row = Text("  ")
            row.append(head + "]", style=style)
            row.append(rest, style="default")
            row.append(f"   {glyph} {note}", style=style)
            _console.print(row)
    else:
        print()
        print("  Sources:")
        for ln in entries:
            m = _SOURCE_LINE_RE.match(ln)
            facet = by_n.get(int(m.group(1))) if m else None
            if facet is None:
                print("  " + ln)
            else:
                glyph, _style, note = _facet_annotation(facet)
                print(f"  {ln}   {glyph} {note}")


def _print_receipt(corrections: int = 0) -> None:
    """The one-line receipt under every answer: the trust segment leads as semantically-colored
    spans WHEN the turn deviated (what was sent / blocked / gated — a calm local turn emits
    none), then the human-control facts (`✎ n corrections` when the user froze and edited this
    answer mid-stream), then the dim run stats. The plain (no-rich) path prints the identical
    text, unstyled. The first time the trust segment shows egress or a gated count, a dim
    `/glass` pointer is appended once per install."""
    stats = _stats_parts()
    spans = _trust_spans()
    if corrections:
        spans = spans + [(f"✎ {corrections} correction{'s' if corrections != 1 else ''}", "human")]
    tail = None
    if any(kind in ("sent", "blocked", "gated") for _, kind in spans):
        try:
            from trust import receipt

            if receipt.take_hint("glass"):
                tail = _GLASS_HINT
        except Exception:
            pass
    if _RICH:
        line = Text("  ╶ ", style=_DIM)
        for i, (text, kind) in enumerate(spans):
            if i:
                line.append(" · ", style=_DIM)
            line.append(text, style=_TRUST_STYLE.get(kind, _DIM))
        if spans and stats:
            line.append(" · ", style=_DIM)
        line.append(" · ".join(stats), style=_DIM)
        if tail:
            line.append(" · " + tail, style=_DIM)
        _console.print(line)
    else:
        parts = [text for text, _ in spans] + stats + ([tail] if tail else [])
        print("  ╶ " + " · ".join(parts))


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
    if _RICH:
        t = Text()
        t.append("  · ", style=_DIM)
        t.append(_FIRST_ANSWER_HINT, style=_DIM)
        _console.print(t)
    else:
        print(f"  · {_FIRST_ANSWER_HINT}")


# The answer's measure: indented to the app's 2-space rhythm and capped so prose stays readable
# on a wide terminal (a full-bleed 200-column paragraph is harder to read than a ~100-column one).
_BODY_WIDTH = 100


def _print_markdown_body(body: str) -> None:
    """Render the answer body as real markdown at the app's 2-space indent, measure-capped.
    Falls back to plain text if the markdown parser trips on arbitrary model output — never lose
    the answer over formatting (Text, so brackets are never eaten as Rich markup)."""
    width = min(_term_width(), _BODY_WIDTH)
    try:
        # Markdown parses markdown, not Rich console markup, so bracketed tokens like
        # `list[str]` or citations `[1]` are safe literal text here.
        _console.print(Padding(Markdown(body), (0, 0, 0, 2)), width=width)
    except Exception:
        _console.print(Padding(Text(body), (0, 0, 0, 2)), width=width)


def response(text: str) -> None:
    """The payload. Leaves the trace rail behind a short labeled rule and renders the answer as
    real markdown — headings, bold, lists, and fenced code with syntax highlighting — so it reads
    like a finished answer, not a log line. The mechanical Sources footer, when present, renders
    through the trust-colored provenance block instead of the markdown body. Falls back to plain
    text if markdown rendering raises (arbitrary model output), and to plain print without rich."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    section("response")  # parts the answer from the trace rail above it (rich + plain branches)
    _console.print() if _RICH else print()  # let the answer breathe beneath its rule
    _final_render(text, plain_body=text)


def _print_corrected_body(body: str, buf: dict) -> bool:
    """Render a human-corrected answer body with the corrected characters marked (the
    provenance spans, styled like the freeze editor showed them). Markdown formatting is traded
    for span fidelity on corrected answers — the correction must be visible exactly where it
    landed, and a markdown re-flow would lose the character positions. Returns False (render
    nothing) when the buffer text doesn't prefix the body — e.g. a /retry replaced the answer —
    so the caller falls back to markdown; never mark by guesswork."""
    try:
        from core import provenance

        prose = str(buf.get("text") or "").rstrip()
        if not prose or not body.startswith(prose):
            return False
        spans = provenance.human_spans(buf)
        if not spans:
            return False
        t = Text()
        pos = 0
        for s, e in spans:
            s, e = min(s, len(prose)), min(e, len(prose))
            if e <= s:
                continue
            t.append(body[pos:s])
            t.append(body[s:e], style=_HUMAN_STYLE)
            pos = e
        t.append(body[pos:])
        width = min(_term_width(), _BODY_WIDTH)
        _console.print(Padding(t, (0, 0, 0, 2)), width=width)
        return True
    except Exception:
        return False  # marking is additive — never lose the answer over it


def _final_render(text: str, *, plain_body: "str | None") -> None:
    """THE final-answer tail (provenance pop → sources split → markdown body → trust-colored
    Sources → receipt → first-answer hint), shared by `response()` and ResponseStream.finish()
    so streamed and non-streamed answers can never drift apart. `plain_body` is what the
    no-rich path prints as the body — the whole text for `response()`, only the trailer beyond
    the already-typed stream for `finish()` (None = nothing left to print). A turn the user
    froze and corrected renders its body with the human-authored spans marked (and the receipt
    counts the corrections) — see set_turn_buffer."""
    gb = _pop_turn_provenance()
    buf = _pop_turn_buffer()
    corrections = len(buf.get("edits") or []) if buf else 0
    if _RICH:
        prose, src_lines = _split_sources(text)
        body = prose if src_lines else text
        if buf is None or not _print_corrected_body(body, buf):
            _print_markdown_body(body)
        if src_lines:
            _print_sources(src_lines, gb)
        _console.print()  # let the answer breathe before the receipt
        _print_receipt(corrections)
        _first_answer_hint()
        _console.print()  # trailing whitespace before the next prompt
    else:
        if plain_body:
            print(plain_body)
            print()
        _print_receipt(corrections)
        _first_answer_hint()
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
        """Append a streamed answer token; opens the response section on the first one. After a
        freeze (freeze_display tore the live tail down), the first resumed token quietly reopens
        a live region — same transient-tail contract, no second section header."""
        if not text:
            return
        if not self._started:
            self._begin()
        elif self._live is None and _RICH:
            self._live = Live(console=_console, transient=True, auto_refresh=False)
            self._live.start()
        self._chars.append(text)
        if self._live is not None:
            now = time.perf_counter()
            if now - self._last >= 0.06:  # throttle (~16/s) so granular tokens don't thrash the live
                self._live.update(self._tail(), refresh=True)
                self._last = now
        else:  # plain (no-rich) path: just type it out
            print(text, end="", flush=True)

    def _freeze_hint(self) -> None:
        """One-time discovery hint for the freeze key (receipt.take_hint — sentinel-backed),
        printed at the exact moment it's actionable: the answer just started streaming and the
        status bar (whose legend would teach it) has left the screen. Only when the latch is
        actually armed — never advertise a hotkey the model can't honor."""
        try:
            from core.continuation import get_freeze_controller
            from trust import receipt

            if not (get_freeze_controller().armed and receipt.take_hint("freeze")):
                return
        except Exception:
            return
        msg = "esc freezes this answer mid-stream — edit it, and the model continues from your text"
        if _RICH:
            t = Text()
            t.append("  · ", style=_DIM)
            t.append(msg, style=_DIM)
            _console.print(t)
        else:
            print(f"  . {msg}")

    def _begin(self) -> None:
        self._started = True
        _live_stop()  # drop the turn's status bar — the answer takes over the bottom of the screen
        if _RICH:
            section("response")  # parts the answer from the trace rail above it
            self._freeze_hint()
            _console.print()
            # transient + a screen-bounded tail => the live region always fits, so stop() erases it
            # cleanly no matter how long the answer runs. Manual refresh (throttled in feed).
            self._live = Live(console=_console, transient=True, auto_refresh=False)
            self._live.start()
        else:
            section("response")  # one header vocabulary (listing.section has the plain branch)
            self._freeze_hint()
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
        if not _RICH:
            print()  # close the typed-out line
        # The plain path typed the streamed tokens out already — its body is only what the
        # recorded final text appends beyond them (e.g. the Sources footer), never the whole
        # thing twice. The rich path re-renders the full text (the live tail was transient).
        streamed = "".join(self._chars).rstrip()
        trailer = None
        if text.rstrip() != streamed and text.startswith(streamed):
            trailer = text[len(streamed):].strip("\n")
        _final_render(text, plain_body=trailer)

    def freeze_display(self) -> None:
        """A freeze (interrupt-and-correct): tear down the live tail so the freeze editor owns
        the screen, KEEPING the streamed chars — feed() reopens a live region on the first
        resumed token. The transient Live erases cleanly; the editor shows the frozen text."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        if not _RICH and self._started:
            print()  # close the typed-out line before the editor prompts

    def reset_to(self, text: str) -> None:
        """Replace the streamed record with the (human-edited) buffer text so the resumed live
        tail and finish()'s trailer math continue from what the user actually kept — never from
        the pre-edit stream."""
        self._chars = [text] if text else []

    def abort(self) -> None:
        """Tear down the live tail without a final render — a failed/cancelled turn. The transient
        Live erases the partial text; the caller surfaces the error (`warn`) separately."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

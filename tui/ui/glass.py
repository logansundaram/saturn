"""
The Glass Box renderer — answer-level provenance, drawn in the trace-viewer vocabulary.

Three stacked blocks: a **trust label** (sources, what left the machine, groundedness, gated
calls), the **answer** with each inline `[n]` citation colored by the trust of its source
(green = local + trusted, yellow = network / untrusted origin, red = a span of that source bled
verbatim into the answer), and a **sources** table. Semantic color only, like the rest of the TUI.
The red citation + its callout is the whole pitch in one frame: "this clause came from a web page,
not from you."

Pure presentation over a `glassbox.GlassBox`; assembly lives in `glassbox.py`.
"""

import re
import textwrap

from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _term_width, _truncate,
)
from .listing import section, table

_CITE_RE = re.compile(r"\[(\d+)\]")

# Semantic style per source facet (matches the gate/listing palette: green=safe, yellow=caution,
# red=risk).
_GREEN, _YELLOW, _RED = "green", "yellow", "bold red"


def _facet_style(facet) -> str:
    if facet is None:
        return _DIM
    if facet.tainted_span:
        return _RED
    if (not facet.trusted) or facet.origin == "network":
        return _YELLOW
    return _GREEN


def _kv(key: str, value: str, style: str = "default") -> None:
    """One `key   value` line of the trust label."""
    if _RICH:
        row = Text()
        row.append(f"    {key:<13}  ", style=_DIM)
        row.append(value, style=style)
        _console.print(row)
    else:
        print(f"    {key:<13}  {value}")


def _render_label(gb) -> None:
    # An incomplete reconstruction (a recorded delta failed to decode — trace truncation) leads
    # the label: every count below may be missing data, and a trust surface must say so before
    # making any claim.
    if not gb.complete:
        _kv("record", "⚠ trace data truncated — sources/taint below are INCOMPLETE", _RED)

    n = len(gb.sources)
    n_local = len(gb.local_sources)
    n_net = len(gb.network_sources)
    if n:
        _kv("sources", f"{n}   ({n_local} local · {n_net} network)")
    elif gb.complete:
        _kv("sources", "0   (answered from the model's own knowledge + context)", _DIM)
    else:
        _kv("sources", "0 recovered (the record is truncated — there may have been more)", _YELLOW)

    # Inference location — the single most important privacy fact about the answer.
    composer = gb.composer_label or "?"
    if gb.composed_local is True:
        _kv("composed by", f"{composer}  ✓ local — your query never left the machine", _GREEN)
    elif gb.composed_local is False:
        _kv("composed by", f"{composer}  ⇅ cloud — your query + context left the machine", _YELLOW)
    else:
        _kv("composed by", f"{composer}  (inference location not recorded for past runs)", _DIM)

    # Local-inference attestation — rendered ONLY when PROVEN (live exact egress slice with zero
    # llm events + every chat role bound local). None means unknown: say NOTHING — silence must
    # never imply local. False needs no extra line; the cloud composed-by line above already
    # states it.
    li = getattr(gb, "local_inference", None)
    if isinstance(li, dict) and li.get("local"):
        models = ", ".join(dict.fromkeys(
            str(b.get("model"))
            for b in li.get("models") or []
            if isinstance(b, dict) and b.get("model") and b.get("role") != "embedder"
        )) or "local models"
        _kv("inference", f"✓ computed entirely on this machine ({models})", _GREEN)

    # What left the machine this turn.
    if gb.sent_known:
        if gb.sent_bytes or gb.sent_hosts:
            from textutil import human_bytes
            hosts = ", ".join(gb.sent_hosts[:3]) + (
                f" +{len(gb.sent_hosts) - 3}" if len(gb.sent_hosts) > 3 else "")
            _kv("left machine", f"{human_bytes(gb.sent_bytes)} → {hosts}", _YELLOW)
        else:
            _kv("left machine", "nothing — local-only this turn", _GREEN)
    elif n_net:
        _kv("left machine", f"{n_net} network source(s) (exact bytes: current session only)", _YELLOW)
    else:
        _kv("left machine", "no network sources", _GREEN)

    # Groundedness — `replans` counts UNGROUNDED verdicts only; the judge ruling a draft grounded
    # leaves it at 0, so 0 cannot be rendered as either "verified" or "not triggered". Say what
    # the state actually records, nothing more.
    if gb.replans:
        _kv("grounded", f"⚠ judge caught an ungrounded draft — {gb.replans} "
                        f"follow-up search{'es' if gb.replans > 1 else ''} inserted", _YELLOW)
    else:
        _kv("grounded", "no ungrounded draft caught (judge ruled it grounded, or wasn't needed)",
            _DIM)

    # Gated calls + the taint headline. The count renders wherever it is KNOWN (live turn, or a
    # record carrying structured gate events); None = unknown stays silent.
    if gb.gated is not None:
        _kv("gated calls", str(gb.gated), _DIM if gb.gated == 0 else _YELLOW)
    # The human decisions themselves — the chain-of-custody line: a signed export carrying this
    # block can show an auditor that a PERSON approved the run_shell it contains. Renders in both
    # live and reconstructed views; nothing prints when no gate prompted (or none was recorded).
    gs = getattr(gb, "gate_summary", None) or []
    approved = [str(c.get("name") or "?") for c in gs if isinstance(c, dict) and c.get("approved")]
    rejected = [str(c.get("name") or "?") for c in gs
                if isinstance(c, dict) and not c.get("approved")]
    if approved:
        _kv("human gate",
            f"you approved {len(approved)} call{'s' if len(approved) != 1 else ''} "
            f"({', '.join(approved)})", _GREEN)
    if rejected:
        _kv("human gate",
            f"you rejected {len(rejected)} call{'s' if len(rejected) != 1 else ''} "
            f"({', '.join(rejected)})", _YELLOW)
    tainted = gb.tainted
    if tainted:
        _kv("taint", f"⚠ {len(tainted)} untrusted span(s) reached the answer", _RED)
    elif gb.complete:
        _kv("taint", "✓ no untrusted content reached the answer", _GREEN)
    else:
        _kv("taint", "? unknown — the truncated record may hide untrusted sources", _YELLOW)


def _render_answer(gb) -> None:
    if not gb.answer:
        return
    by_n = {s.n: s for s in gb.sources}
    width = max(20, _term_width() - 8)
    _console.print(Text("  answer", style=_DIM)) if _RICH else print("  answer")
    for para in gb.answer.splitlines():
        for line in (textwrap.wrap(para, width) if para.strip() else [""]):
            if not _RICH:
                print(f"    {line}")
                continue
            row = Text("    ")
            pos = 0
            for m in _CITE_RE.finditer(line):
                row.append(line[pos:m.start()], style="default")
                facet = by_n.get(int(m.group(1)))
                row.append(m.group(0), style=_facet_style(facet))
                pos = m.end()
            row.append(line[pos:], style="default")
            _console.print(row)
    # Red callouts spelling out each tainted span — the data→answer channel, named.
    for s in gb.tainted:
        msg = (f"⚠ tainted [{s.n}]: a span here was planted by {s.tool} — \"{s.tainted_span}\"")
        for i, chunk in enumerate(textwrap.wrap(msg, width) or [""]):
            text = ("    " if i == 0 else "      ") + chunk
            _console.print(Text(text, style=_RED)) if _RICH else print(text)


def _render_sources(gb) -> None:
    if not gb.sources:
        return
    section("sources", "/source <n> for the full text behind a citation")
    rows = []
    for s in gb.sources:
        if s.tainted_span:
            glyph, gstyle, note = "⛔", _RED, "TAINTED → answer"
        elif s.origin == "network" or not s.trusted:
            glyph, gstyle, note = "◐", _YELLOW, ("untrusted origin" if not s.trusted else "network")
        else:
            glyph, gstyle, note = "✓", _GREEN, "trusted"
        if s.injection_flagged:
            note += " · injection-flagged"
        rows.append([
            (f"[{s.n}]", _DIM),
            (glyph, gstyle),
            (s.origin, gstyle),
            (_truncate(s.label, 48), "default"),
            (note, gstyle),
        ])
    table(rows)


def show_glassbox(gb) -> None:
    """Render a GlassBox: trust label, the answer with trust-colored citations, the sources table."""
    q = _truncate(gb.query, 60) if gb.query else "(no query)"
    section("glass box", f"\"{q}\"")
    _render_label(gb)
    _console.print() if _RICH else print()
    _render_answer(gb)
    _render_sources(gb)  # opens with its own section rule (and the blank that precedes one)
    _console.print() if _RICH else print()
    if _RICH:
        foot = Text()
        foot.append("  ╶ ", style=_DIM)
        foot.append("/trace why", style=_ACCENT)
        foot.append(" decision chain · ", style=_DIM)
        foot.append("/trace export", style=_ACCENT)
        foot.append(" signs this answer", style=_DIM)
        _console.print(foot)
    else:
        print("  ╶ /trace why decision chain · /trace export signs this answer")

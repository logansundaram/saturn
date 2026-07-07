"""
The Glass Box renderer — answer-level provenance, drawn in the trace-viewer vocabulary.

Three stacked blocks: a **trust label** (sources, what left the machine, groundedness, gated
calls), the **answer** with each inline `[n]` citation colored by the trust of its source
(green = local + trusted, yellow = network / untrusted origin), and a **sources** table.
Semantic color only, like the rest of the TUI — "this clause cites a web page, not your own data."

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
        _kv("record", "⚠ trace data truncated — sources below are INCOMPLETE", _RED)

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

    # Self-correction — `replans` counts the times rectify sent the plan back for revision (a
    # placeholder resolved, a dead end retried, a missing lookup added). A quiet pass records
    # nothing, so 0 cannot be rendered as "verified correct". Say what the state records.
    if gb.replans:
        _kv("rectified", f"⚠ rectify revised the plan {gb.replans} "
                         f"time{'s' if gb.replans > 1 else ''} mid-run", _YELLOW)
    else:
        _kv("rectified", "no plan revision needed (every step resolved as planned)",
            _DIM)

    # Gated calls. The count renders wherever it is KNOWN (live turn, or a record carrying
    # structured gate events); None = unknown stays silent.
    if gb.gated is not None:
        _kv("gated calls", str(gb.gated), _DIM if gb.gated == 0 else _YELLOW)
    # The human decisions themselves — the one fact that can't be recomputed from the run
    # (gotcha #7: a human's yes/no exists only in the gate_events record). Renders in both
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


def _render_sources(gb) -> None:
    if not gb.sources:
        return
    section("sources", "/source <n> for the full text behind a citation")
    rows = []
    for s in gb.sources:
        if s.origin == "network" or not s.trusted:
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
        foot.append(" records this answer", style=_DIM)
        _console.print(foot)
    else:
        print("  ╶ /trace why decision chain · /trace export records this answer")

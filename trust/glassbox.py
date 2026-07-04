"""
The Glass Box — answer-level provenance. `/trace why` shows how the agent worked; this shows where
what it told you came from.

For one answer it gathers, per cited source: where the source came from (local disk vs the
network), whether its origin is trusted, and whether it tripped the injection scan. Plus the
aggregate trust label: how many sources, how many left the machine, bytes egressed this turn,
whether rectify revised the plan mid-run.

Two entry points feed one assembler:
  - `build_from_state(state, ...)` — the live last turn (exact egress slice available).
  - `build_from_record(query, response, deltas, ...)` — reconstructed from a recorded/exported run
    (the trace DB already persists every structural input; origin is RECOMPUTED, so no extra
    column is stored — historical runs lack only the exact egress slice, which is inferred from the
    source tools instead).

Numbering matches the answer's inline `[n]` citations exactly: it reuses
`nodes.synthesize.build_sources`, the same numbering `/source` uses. Imports only leaves
(egress, quarantine) + that one pure helper (lazily), so it stays UI-free and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from trust import egress
from trust import quarantine

_LEADING_IDENT = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class SourceFacet:
    n: int
    label: str
    tool: str
    origin: str                 # "local" | "network"
    trusted: bool               # not quarantine.is_untrusted(tool)
    injection_flagged: bool     # this observation tripped the injection scan (tool_events)


@dataclass
class GlassBox:
    query: str
    answer: str                 # the answer prose (the Sources: footer stripped)
    sources: "list[SourceFacet]"
    composed_local: "bool | None"   # inference stayed local this turn (None = not recorded/history)
    composer_label: str
    sent_bytes: int
    sent_hosts: "list[str]"
    sent_known: bool            # exact egress available (live turn) vs inferred from tools (history)
    gated: "int | None"         # calls that faced the approval gate (None = not recorded/history)
    # How many times rectify sent the plan back for revision (replan ran). NOTE: a quiet pass
    # records nothing, so 0 means "no revision happened" — it must never be rendered as
    # "verified correct"; the renderer says only what the state records.
    replans: int
    # Whether every recorded delta behind this box decoded — False when the trace truncated an
    # event (stores/trace._DATA_CAP) and sources below may therefore be INCOMPLETE. A trust
    # surface must say so rather than render '0 sources · no untrusted content' over data it
    # failed to load.
    complete: bool = True
    # Flattened per-call human decisions ([{"name", "approved"}]) from the structured
    # gate_events record (nodes/approval.gate_event) — the ONE persisted exception to
    # the recompute-everything design, because a human decision cannot be recomputed. None when
    # no gate event exists in the inputs: that is UNKNOWN (an older record, or no prompt fired),
    # never "zero gates" — the renderer says nothing rather than claiming.
    gate_summary: "list[dict] | None" = None

    @property
    def network_sources(self) -> "list[SourceFacet]":
        return [s for s in self.sources if s.origin == "network"]

    @property
    def local_sources(self) -> "list[SourceFacet]":
        return [s for s in self.sources if s.origin == "local"]


def _is_network(tool: str) -> bool:
    """Whether the tool's output crossed the network (the privacy axis). Derived from
    quarantine's trust classification — the ONE home of the untrusted-tool set — minus
    search_knowledge_base: RAG stays LOCAL even though it is untrusted (the corpus may hold
    downloaded docs); origin (local vs network) and trust are independent axes on purpose.
    Deriving (instead of keeping a parallel tool-name set here) means a new fetch-shaped tool
    added to quarantine can never render as a green 'local' source in /glass."""
    return quarantine.is_untrusted(tool) and tool != "search_knowledge_base"


def _strip_footer(answer: str) -> str:
    """The answer prose without the mechanical `Sources:` footer the synthesizer appends — so
    display works on what the model actually wrote, not on the source labels."""
    if not answer:
        return ""
    i = answer.rfind("\nSources:")
    return (answer[:i].rstrip() if i != -1 else answer).strip()


def _final_answer(messages) -> str:
    """The turn's final answer: the trailing AIMessage with no tool calls. Duck-typed so this leaf
    doesn't import langchain just for an isinstance check."""
    for m in reversed(messages or []):
        if type(m).__name__ == "AIMessage" and not getattr(m, "tool_calls", None):
            return str(getattr(m, "content", "") or "")
    return ""


def _source_tool(idx: int, n_tool_sources: int, label: str) -> str:
    """The tool behind source #idx. The first `n_tool_sources` sources are tool_results, labeled
    `name(args)` — the leading identifier is the tool; the rest are retrieval passages."""
    if idx >= n_tool_sources:
        return "search_knowledge_base"
    m = _LEADING_IDENT.match(label or "")
    return m.group(1) if m else "?"


def _gate_info(gate_events) -> "tuple[int | None, list[dict] | None]":
    """(gated call count, flattened [{"name","approved"}] summary) from the structured
    gate_events record, or (None, None) when none exists. None is deliberate: an input without
    gate events is UNKNOWN (an older record, or simply no prompt fired) — deriving 0 here would
    let an old record claim 'nothing was gated' over a run that may have gated plenty."""
    calls = [
        c
        for ev in gate_events or [] if isinstance(ev, dict)
        for c in ev.get("calls") or [] if isinstance(c, dict)
    ]
    if not calls:
        return None, None
    summary = [{"name": str(c.get("name") or "?"), "approved": bool(c.get("approved"))}
               for c in calls]
    return len(calls), summary


def _assemble(query, answer, tool_results, documents_retrieved, tool_events, replans,
              egress_events, gated, complete=True, gate_summary=None) -> GlassBox:
    # Lazy: build_sources lives in synthesize.py (pulls llms/budget) and RETRIEVAL_TOOLS in
    # registry (pulls the tool registry) — fine off the hot loop, and build_sources keeps the
    # numbering identical to the answer's [n] and to /source.
    from nodes.synthesize import build_sources
    from tools.registry import RETRIEVAL_TOOLS

    prose = _strip_footer(answer or "")
    numbered_tools, numbered_docs, sources = build_sources(tool_results, documents_retrieved)

    # Per-SOURCE event alignment: tool_node appends exactly one tool_events entry per call, and
    # routes each call's observation to tool_results (non-retrieval) or documents_retrieved
    # (retrieval) — so splitting the events the same way re-pairs each source with ITS event.
    # That is what carries the per-observation quarantine flag: keying flags by tool NAME would
    # smear one flagged page across every clean source the same tool fetched.
    events = tool_events or []
    tool_evs = [ev for ev in events if ev.get("name") not in RETRIEVAL_TOOLS]
    doc_evs = [ev for ev in events if ev.get("name") in RETRIEVAL_TOOLS]
    flagged_names = {ev.get("name") for ev in events if ev.get("quarantine")}

    def _event_for(idx: int) -> "dict | None":
        if idx < len(numbered_tools):
            return tool_evs[idx] if idx < len(tool_evs) else None
        d = idx - len(numbered_tools)
        return doc_evs[d] if d < len(doc_evs) else None

    facets: list[SourceFacet] = []
    for idx, (n, label) in enumerate(sources):
        ev = _event_for(idx)
        tool = str((ev or {}).get("name") or _source_tool(idx, len(numbered_tools), label))
        facets.append(SourceFacet(
            n=n, label=label, tool=tool,
            origin="network" if _is_network(tool) else "local",
            trusted=not quarantine.is_untrusted(tool),
            # Event alignment gives the per-observation verdict; without an event (older
            # records) fall back to the coarse by-name flag rather than losing the warning.
            injection_flagged=(bool(ev.get("quarantine")) if ev is not None
                               else tool in flagged_names),
        ))

    # Egress: exact for the live turn (the ledger slice), inferred from source tools otherwise.
    # Same aggregation as the receipt and /privacy egress (egress.summarize_events).
    if egress_events is not None:
        agg = egress.summarize_events(egress_events)
        hosts, sent_bytes = agg["hosts"], agg["bytes"]
        composed_local = "llm" not in agg["channels"]
        sent_known = True
    else:
        hosts, sent_bytes, composed_local, sent_known = [], 0, None, False

    composer_label = ""
    try:
        from config import get_config
        spec = get_config().model_for_role("synthesizer")
        composer_label = str(getattr(spec, "model", None) or spec or "")
    except Exception:
        composer_label = ""

    return GlassBox(
        query=query or "",
        answer=prose,
        sources=facets,
        composed_local=composed_local,
        composer_label=composer_label,
        sent_bytes=sent_bytes,
        sent_hosts=hosts,
        sent_known=sent_known,
        gated=gated,
        replans=replans or 0,
        complete=bool(complete),
        gate_summary=gate_summary,
    )


def build_from_state(state, *, egress_events=None, gated=None) -> GlassBox:
    """Glass Box for the live last turn, read from the AgentState accumulators (like /source).
    The caller passes the turn's egress slice + gate count (UI state) so this leaf stays UI-free.
    When the state carries structured gate_events (the approval node's per-prompt record), they
    supersede the passed UI counter — one source of truth for the human decisions."""
    state = state or {}
    ev_count, gate_summary = _gate_info(state.get("gate_events"))
    if ev_count is not None:
        gated = ev_count
    return _assemble(
        state.get("current_query", ""),
        _final_answer(state.get("messages")),
        state.get("tool_results") or [],
        state.get("documents_retrieved") or [],
        state.get("tool_events") or [],
        state.get("replans", 0),
        egress_events,
        gated,
        gate_summary=gate_summary,
    )


def build_live(state, *, gated=None) -> GlassBox:
    """Glass Box for the live last turn with the EXACT egress slice when (and only when) it is
    trustworthy. The turn-mark guard lives HERE so every live consumer — the `/glass` command and
    the native post-answer provenance — applies the identical contract instead of re-rolling it:
    the slice is passed only when a turn mark was recorded (receipt.reset_turn ran) AND no
    `/privacy egress clear` wiped events past it; otherwise egress_events=None (UNKNOWN — an
    empty-because-cleared slice must never render as 'local-only this turn')."""
    from trust import receipt  # leaf importing leaf; lazy keeps import-time cost off the tools path

    mark = receipt.turn_mark()
    ev = None
    if mark > 0 and not egress.cleared_since(mark):
        ev = egress.events_since(mark)
    return build_from_state(state, egress_events=ev, gated=gated)


def build_from_record(query, response, deltas, *, gated=None, complete=True) -> GlassBox:
    """Glass Box reconstructed from a recorded/exported run. `deltas` is the per-event delta dicts
    (decoded) in order — the accumulators are summed across them exactly as the loop summed them.
    Egress isn't correlated to a run in the trace DB, so origin is inferred from the source tools
    (egress_events=None) — honest about the one facet history can't carry exactly. The caller
    passes `complete=False` when any recorded delta failed to decode (the trace's _DATA_CAP can
    truncate a fat event's JSON) — the box then renders as INCOMPLETE instead of asserting
    '0 sources / nothing untrusted' over data it never saw."""
    tool_results: list = []
    docs: list = []
    tool_events: list = []
    gate_events: list = []
    replans = 0
    for d in deltas or []:
        if not isinstance(d, dict):
            continue
        tool_results += d.get("tool_results") or []
        docs += d.get("documents_retrieved") or []
        tool_events += d.get("tool_events") or []
        gate_events += d.get("gate_events") or []
        if "replans" in d and d.get("replans") is not None:
            replans = d["replans"]
    # The human decisions are the one fact a record CAN carry exactly (they ride the deltas like
    # tool_events) — when present they drive the gated count + summary; when absent (an older
    # record) the box stays honest: None = unknown, never "0 gated".
    ev_count, gate_summary = _gate_info(gate_events)
    if ev_count is not None:
        gated = ev_count
    return _assemble(query or "", response or "", tool_results, docs, tool_events,
                     replans, None, gated, complete=complete, gate_summary=gate_summary)

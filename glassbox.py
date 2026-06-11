"""
The Glass Box — answer-level provenance. `/trace why` shows how the agent worked; this shows
whether you can trust WHAT it told you.

For one answer it gathers, per cited source: where the source came from (local disk vs the
network), whether its origin is trusted, whether it tripped the injection scan, and — the headline
— whether a span of that (untrusted) source appears VERBATIM in the final answer. That last facet
is the data→answer half of indirect injection: `quarantine.taint_scan` catches untrusted text
flowing into a tool CALL; the Glass Box catches it flowing into the ANSWER, via the same
`quarantine.longest_overlap` primitive. Plus the aggregate trust label: how many sources, how many
left the machine, bytes egressed this turn, whether the groundedness judge weighed in.

Two entry points feed one assembler:
  - `build_from_state(state, ...)` — the live last turn (exact egress slice available).
  - `build_from_record(query, response, deltas, ...)` — reconstructed from a recorded/exported run
    (the trace DB already persists every structural input; taint/origin are RECOMPUTED, so no extra
    column is stored — historical runs lack only the exact egress slice, which is inferred from the
    source tools instead).

Numbering matches the answer's inline `[n]` citations exactly: it reuses
`node_registry.synthesize.build_sources`, the same numbering `/source` uses. Imports only leaves
(egress, quarantine, textutil) + that one pure helper (lazily), so it stays UI-free and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import egress
import quarantine
from textutil import clip

_TAINT_PREVIEW = 80
_LEADING_IDENT = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class SourceFacet:
    n: int
    label: str
    tool: str
    origin: str                 # "local" | "network"
    trusted: bool               # not quarantine.is_untrusted(tool)
    injection_flagged: bool     # this observation tripped the injection scan (tool_events)
    tainted_span: "str | None"  # a span of THIS source present verbatim in the answer, or None


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
    # How many times the replan judge found a draft UNGROUNDED and inserted a search. NOTE: the
    # state does NOT record the judge ruling a draft grounded (that path leaves replans at 0),
    # so 0 means "no ungrounded draft was caught" — judge-verified-grounded and judge-never-ran
    # are indistinguishable here, and the renderer must not claim either.
    replans: int
    # Whether every recorded delta behind this box decoded — False when the trace truncated an
    # event (stores/trace._DATA_CAP) and sources/taint below may therefore be INCOMPLETE. A
    # trust surface must say so rather than render '0 sources · no untrusted content' over data
    # it failed to load.
    complete: bool = True

    @property
    def tainted(self) -> "list[SourceFacet]":
        return [s for s in self.sources if s.tainted_span]

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
    """The answer prose without the mechanical `Sources:` footer the synthesizer appends — so taint
    matching and display work on what the model actually wrote, not on the source labels."""
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


def _assemble(query, answer, tool_results, documents_retrieved, tool_events, replans,
              egress_events, gated, complete=True) -> GlassBox:
    # Lazy: build_sources lives in synthesize.py (pulls llms/budget) and RETRIEVAL_TOOLS in
    # registry (pulls the tool registry) — fine off the hot loop, and build_sources keeps the
    # numbering identical to the answer's [n] and to /source.
    from node_registry.synthesize import build_sources
    from registry import RETRIEVAL_TOOLS

    prose = _strip_footer(answer or "")
    numbered_tools, numbered_docs, sources = build_sources(tool_results, documents_retrieved)
    entries = numbered_tools + numbered_docs  # [(n) text] aligned 1:1 with `sources`

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

    # First pass: identity per source. Second: one batched overlap scan for the untrusted ones
    # (longest_overlap_many indexes the constant answer side ONCE instead of per source).
    rows: list[dict] = []
    untrusted_obs: list[str] = []
    for idx, (n, label) in enumerate(sources):
        text = entries[idx]
        prefix = f"[{n}] "
        obs = text[len(prefix):] if text.startswith(prefix) else text
        ev = _event_for(idx)
        tool = str((ev or {}).get("name") or _source_tool(idx, len(numbered_tools), label))
        trusted = not quarantine.is_untrusted(tool)
        rows.append({
            "n": n, "label": label, "tool": tool, "trusted": trusted,
            # Event alignment gives the per-observation verdict; without an event (older
            # records) fall back to the coarse by-name flag rather than losing the warning.
            "flagged": bool(ev.get("quarantine")) if ev is not None else tool in flagged_names,
            "taint_idx": None,
        })
        if not trusted:
            rows[-1]["taint_idx"] = len(untrusted_obs)
            untrusted_obs.append(obs)
    spans = quarantine.longest_overlap_many(prose, untrusted_obs) if untrusted_obs else []

    facets = [
        SourceFacet(
            n=r["n"], label=r["label"], tool=r["tool"],
            origin="network" if _is_network(r["tool"]) else "local",
            trusted=r["trusted"],
            injection_flagged=r["flagged"],
            tainted_span=(
                clip(spans[r["taint_idx"]], _TAINT_PREVIEW)
                if r["taint_idx"] is not None and spans[r["taint_idx"]] else None
            ),
        )
        for r in rows
    ]

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
    )


def build_from_state(state, *, egress_events=None, gated=None) -> GlassBox:
    """Glass Box for the live last turn, read from the AgentState accumulators (like /source).
    The caller passes the turn's egress slice + gate count (UI state) so this leaf stays UI-free."""
    state = state or {}
    return _assemble(
        state.get("current_query", ""),
        _final_answer(state.get("messages")),
        state.get("tool_results") or [],
        state.get("documents_retrieved") or [],
        state.get("tool_events") or [],
        state.get("replans", 0),
        egress_events,
        gated,
    )


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
    replans = 0
    for d in deltas or []:
        if not isinstance(d, dict):
            continue
        tool_results += d.get("tool_results") or []
        docs += d.get("documents_retrieved") or []
        tool_events += d.get("tool_events") or []
        if "replans" in d and d.get("replans") is not None:
            replans = d["replans"]
    return _assemble(query or "", response or "", tool_results, docs, tool_events,
                     replans, None, gated, complete=complete)

"""The answer's source numbering — one home (moved out of nodes/synthesize.py with the plan
engine's removal, 2026-09-27). `build_sources` numbers everything the turn gathered, in the
order it was gathered, and the same numbering serves the answer's Sources footer
(nodes/agent.py), `/trace source`, so [3] means the same thing everywhere."""

from __future__ import annotations

from textutil import clip, parse_doc_sources, split_call_result

_MAX_SOURCE_LABEL = 100


def _tool_source_label(result) -> str:
    """Provenance label for one tool_results entry. Entries are `name(args) -> result` strings
    (nodes/tools.py pairs them on purpose); the call repr before the arrow is the label — split
    via textutil.split_call_result, THE one parser of that serialization."""
    return clip(split_call_result(result)[0], _MAX_SOURCE_LABEL)


def _doc_source_label(observation) -> str:
    """Provenance label for one retrieval observation: the distinct `[source: …]` names inside it
    (one search_knowledge_base call returns several chunks, usually from a handful of files)."""
    names = parse_doc_sources(observation)
    if names:
        return clip("knowledge base: " + ", ".join(names), _MAX_SOURCE_LABEL)
    return "knowledge base passage"


def build_sources(tool_results, documents_retrieved):
    """Number everything the turn gathered, in the order it was gathered.

    Returns (numbered_tool_results, numbered_docs, sources) where the numbered lists are
    `[n] …` strings and `sources` is the [(n, label)] registry the footer renders. One shared
    numbering across both sections so an `[4]` is unambiguous."""
    sources: list[tuple[int, str]] = []
    numbered_tools: list[str] = []
    for r in tool_results or []:
        n = len(sources) + 1
        sources.append((n, _tool_source_label(r)))
        numbered_tools.append(f"[{n}] {r}")
    numbered_docs: list[str] = []
    for d in documents_retrieved or []:
        n = len(sources) + 1
        sources.append((n, _doc_source_label(d)))
        numbered_docs.append(f"[{n}] {d}")
    return numbered_tools, numbered_docs, sources

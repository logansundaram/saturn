"""The answer's source numbering — one home (moved out of nodes/synthesize.py with the plan
engine's removal, 2026-09-27). `build_sources` numbers everything the turn gathered, in the
order it was gathered, and the same numbering serves the answer's Sources footer
(nodes/agent.py) and `/trace source`, so [3] means the same thing everywhere. The number is the
handle `/trace source <n>` takes; the model is not asked to cite it inline."""

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


def build_sources(tool_results, documents_retrieved) -> "list[tuple[int, str, str]]":
    """Number everything the turn gathered, in the order it was gathered: `(n, label, text)` —
    the footer's line, and the full entry behind it. One numbering across tool results and
    retrieved documents, so a `[4]` is unambiguous."""
    entries = [(_tool_source_label(r), str(r)) for r in tool_results or []]
    entries += [(_doc_source_label(d), str(d)) for d in documents_retrieved or []]
    return [(n, label, text) for n, (label, text) in enumerate(entries, 1)]

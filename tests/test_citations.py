"""The answer's source numbering (core/sources.py) and the mechanical Sources footer nodes/agent.py
appends to the recorded answer (runtime.citations)."""

from core.sources import _doc_source_label, _tool_source_label, build_sources
from nodes.agent import sources_footer


def test_numbering_is_continuous_across_sections():
    tools = ["calculate(expression='1+1') -> 2", "web_search(query='x') -> results…"]
    docs = ["[source: handbook.md]\nsome passage"]
    numbered_tools, numbered_docs, sources = build_sources(tools, docs)
    assert [s[0] for s in sources] == [1, 2, 3]
    assert numbered_tools[0].startswith("[1] calculate")
    assert numbered_tools[1].startswith("[2] web_search")
    assert numbered_docs[0].startswith("[3] [source: handbook.md]")


def test_tool_label_is_the_call_repr():
    assert (
        _tool_source_label("web_search(query='best x') -> {json: blob} -> nested arrow")
        == "web_search(query='best x')"
    )
    # Pathological huge call reprs are clamped to a sane label.
    assert len(_tool_source_label("t(" + "a" * 500 + ") -> r")) <= 100


def test_doc_label_collects_distinct_sources():
    obs = "[source: a.md]\nchunk one\n\n[source: b.md]\nchunk two\n\n[source: a.md]\nchunk three"
    assert _doc_source_label(obs) == "knowledge base: a.md, b.md"
    assert _doc_source_label("no markers here") == "knowledge base passage"


def test_footer_shape_and_empty_case():
    footer = sources_footer(["calc(x=1) -> 1"], [])
    assert footer.startswith("Sources:")
    assert "[1] calc(x=1)" in footer
    assert sources_footer([], []) == ""


def test_empty_inputs():
    numbered_tools, numbered_docs, sources = build_sources([], [])
    assert numbered_tools == [] and numbered_docs == [] and sources == []


def test_split_call_result_is_the_one_parser():
    # THE parser of nodes/tools.py's `name(args) -> observation` serialization — the Sources
    # labels take [0], an observation-content reader takes [1]; one function, no drift.
    from textutil import CALL_RESULT_SEP, split_call_result

    assert split_call_result(f"calc(x=1){CALL_RESULT_SEP}1") == ("calc(x=1)", "1")
    assert split_call_result("no separator")[0] == "no separator"


# ── the [source: …] marker pair ───────────────────────────────────────────────────────────────


def test_doc_source_marker_round_trip():
    """tools/knowledge builds the marker and the Sources receipt parses it back — one builder +
    one parser, so the round trip must be lossless and dedupe in order."""
    from textutil import doc_source_label, parse_doc_sources

    obs = (
        doc_source_label("a.md") + "\nchunk one\n\n"
        + doc_source_label("b.pdf", 3) + "\nchunk two\n\n"
        + doc_source_label("a.md") + "\nchunk three"
    )
    assert parse_doc_sources(obs) == ["a.md", "b.pdf, page 3"]
    assert parse_doc_sources("no markers here") == []
    assert doc_source_label(None) == "[source: unknown]"

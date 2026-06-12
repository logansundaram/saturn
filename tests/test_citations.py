"""nodes/synthesize.py provenance helpers — the numbering the synthesizer cites against
and the mechanical Sources footer appended to the answer (runtime.citations)."""

from nodes.synthesize import (
    build_sources,
    sources_footer,
    _tool_source_label,
    _doc_source_label,
)


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
    obs = (
        "[source: a.md]\nchunk one\n\n"
        "[source: b.pdf, page 3]\nchunk two\n\n"
        "[source: a.md]\nchunk three"
    )
    label = _doc_source_label(obs)
    assert label == "knowledge base: a.md, b.pdf, page 3"
    assert _doc_source_label("no markers here") == "knowledge base passage"


def test_footer_shape_and_empty_case():
    _, _, sources = build_sources(["calc(x=1) -> 1"], [])
    footer = sources_footer(sources)
    assert footer.startswith("Sources:")
    assert "[1] calc(x=1)" in footer
    assert sources_footer([]) == ""


def test_empty_inputs():
    numbered_tools, numbered_docs, sources = build_sources([], [])
    assert numbered_tools == [] and numbered_docs == [] and sources == []

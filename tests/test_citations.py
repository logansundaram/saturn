"""nodes/synthesize.py provenance helpers — the numbering the synthesizer cites against
and the mechanical Sources footer appended to the answer (runtime.citations)."""

from nodes.synthesize import (
    build_sources,
    sources_footer,
    _gathered_section,
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


def test_gathered_section_headers_pinned():
    """Pin the exact prompt headers _gathered_section reconstructs — the section folding must
    keep the synthesizer's prompt bytes identical to the pre-refactor two-block form."""
    items = ["calc(x=1) -> 1"]
    numbered = ["[1] calc(x=1) -> 1"]
    msg = _gathered_section(items, numbered, True, "Tool results")
    assert msg.content == (
        "Tool results (numbered — cite the matching [n] after claims drawn from them):\n"
        "[1] calc(x=1) -> 1"
    )
    msg = _gathered_section(items, numbered, False, "Tool results")
    assert msg.content == "Tool results:\ncalc(x=1) -> 1"
    msg = _gathered_section(items, numbered, True, "Retrieved documents")
    assert msg.content.startswith(
        "Retrieved documents (numbered — cite the matching [n] after claims drawn from them):\n"
    )
    # Nothing gathered -> no section message at all (the prompt omits the block).
    assert _gathered_section([], [], True, "Tool results") is None


def test_split_call_result_is_the_one_parser():
    # THE parser of nodes/tools.py's `name(args) -> observation` serialization — synthesize's
    # Sources labels take [0], the Glass Box's taint corpus takes [1]; one function, no drift.
    from textutil import CALL_RESULT_SEP, split_call_result

    call, obs = "web_search(query='x')", "result text -> with an arrow inside"
    got_call, got_obs = split_call_result(f"{call}{CALL_RESULT_SEP}{obs}")
    assert got_call == call
    assert got_obs == obs  # only the FIRST separator splits — observation content survives whole
    # No separator: both halves are the whole entry, so the label fallback and the
    # keep-the-whole-observation fallback coincide by construction.
    assert split_call_result("plain doc passage") == ("plain doc passage", "plain doc passage")

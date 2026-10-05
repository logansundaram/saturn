"""stores/memory_registry — the durable one-fact-per-bullet contract at the write boundary.

add_memory is the ONE place every caller (the `remember` tool, /memory add) lands, so it must
normalize what it writes: a multi-line model-supplied fact written verbatim would leave
continuation lines that _facts() never returns (the grounding block silently truncates the
fact every turn) and that remove_memory's header+bullets rewrite would permanently drop. The
category rides inside the bullet's "[category] " prefix, so "]" and newlines must be
sanitized out of it too. All offline; isolated_paths keeps the real memory.md untouched.
"""

from stores import memory_registry as mr


def _context(query: str = "") -> "tuple[str, list[int]]":
    """The grounding memory block as one string (the always half, then the by-match half) and
    the ids the turn stamps — what the grounding node's two halves add up to."""
    always, matched, ids = mr.memory_context_split(query)
    return "\n".join(b for b in (always, matched) if b), ids


def _block(query: str = "") -> str:
    return _context(query)[0]


def test_multiline_fact_collapses_to_one_bullet(isolated_paths):
    mr.add_memory("prefers terse answers\nand bullet lists\n\twith tabs")
    facts = mr.entries()
    assert len(facts) == 1
    assert facts[0]["text"] == "prefers terse answers and bullet lists with tabs"
    # The grounding block carries the WHOLE fact, not just the first physical line.
    assert "with tabs" in _block()
    # No stray non-bullet continuation lines in the file itself — every line after the header is
    # a `## layer` heading, a layer blurb comment, or one bullet.
    raw = mr._read_raw()
    body = raw.split(mr._HEADER, 1)[-1]
    assert all(
        line.startswith(("- ", "## ", "<!--"))
        for line in body.splitlines() if line.strip()
    )


def test_remove_memory_preserves_other_facts_byte_complete(isolated_paths):
    # Pre-fix, the first removal's header+bullets rewrite dropped every continuation line of
    # every OTHER stored fact — the data-loss half of the bug.
    mr.add_memory("first fact\ncontinued first")
    mr.add_memory("second fact")
    removed = mr.remove_memory(2)
    assert removed is not None and "second fact" in removed
    facts = mr.entries()
    assert len(facts) == 1
    assert facts[0]["text"] == "first fact continued first"


def test_duplicate_multiline_fact_dedups(isolated_paths):
    assert mr.add_memory("likes python\nuses it daily").startswith("Remembered")
    # The same fact reflowed (newline vs space) is the SAME fact post-normalization — dedup
    # must compare the sanitized form, which is why normalization happens before the check.
    assert mr.add_memory("likes  python uses it daily").startswith("Already remembered")
    assert len(mr.entries()) == 1


def test_category_with_bracket_and_newline_keeps_prefix_parseable(isolated_paths):
    mr.add_memory("bracket fact", category="work] stuff\nnewline")
    facts = mr.entries()
    assert len(facts) == 1
    # The text must parse back clean of the "(date) [category] " prefix — a raw "]" or newline
    # in the category would corrupt the parse (and the bullet line itself).
    assert facts[0]["text"] == "bracket fact"
    assert "[work stuff newline]" in mr._display(facts[0])


def test_category_sanitizing_to_empty_falls_back_to_untagged(isolated_paths):
    mr.add_memory("plain fact", category="]]] \n")
    facts = mr.entries()
    assert len(facts) == 1
    assert facts[0]["text"] == "plain fact"
    assert "[" not in mr._display(facts[0])  # "general" is the untagged default — no prefix written

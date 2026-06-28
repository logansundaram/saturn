"""stores/memory_registry — the durable one-fact-per-bullet contract at the write boundary.

add_memory is the ONE place every caller (the `remember` tool, /memory add) lands, so it must
normalize what it writes: a multi-line model-supplied fact written verbatim would leave
continuation lines that _facts() never returns (the grounding block silently truncates the
fact every turn) and that remove_memory's header+bullets rewrite would permanently drop. The
category rides inside the "[category] " prefix _PREFIX_RE parses, so "]" and newlines must be
sanitized out of it too. All offline; isolated_paths keeps the real memory.md untouched.
"""

from stores import memory_registry as mr


def test_multiline_fact_collapses_to_one_bullet(isolated_paths):
    mr.add_memory("prefers terse answers\nand bullet lists\n\twith tabs")
    facts = mr.list_memory()
    assert len(facts) == 1
    assert mr._fact_text(facts[0]) == "prefers terse answers and bullet lists with tabs"
    # The grounding block carries the WHOLE fact, not just the first physical line.
    assert "with tabs" in mr.read_memory_block()
    # No stray non-bullet continuation lines in the file itself.
    raw = mr._read_raw()
    body = raw.split(mr._HEADER, 1)[-1]
    assert all(line.startswith("- ") for line in body.splitlines() if line.strip())


def test_remove_memory_preserves_other_facts_byte_complete(isolated_paths):
    # Pre-fix, the first removal's header+bullets rewrite dropped every continuation line of
    # every OTHER stored fact — the data-loss half of the bug.
    mr.add_memory("first fact\ncontinued first")
    mr.add_memory("second fact")
    removed = mr.remove_memory(2)
    assert removed is not None and "second fact" in removed
    facts = mr.list_memory()
    assert len(facts) == 1
    assert mr._fact_text(facts[0]) == "first fact continued first"


def test_duplicate_multiline_fact_dedups(isolated_paths):
    assert mr.add_memory("likes python\nuses it daily").startswith("Remembered")
    # The same fact reflowed (newline vs space) is the SAME fact post-normalization — dedup
    # must compare the sanitized form, which is why normalization happens before the check.
    assert mr.add_memory("likes  python uses it daily").startswith("Already remembered")
    assert len(mr.list_memory()) == 1


def test_category_with_bracket_and_newline_keeps_prefix_parseable(isolated_paths):
    mr.add_memory("bracket fact", category="work] stuff\nnewline")
    facts = mr.list_memory()
    assert len(facts) == 1
    # _fact_text must strip the whole "(date) [category] " prefix cleanly — a raw "]" or
    # newline in the category would corrupt the _PREFIX_RE parse (and the bullet line itself).
    assert mr._fact_text(facts[0]) == "bracket fact"
    assert "[work stuff newline]" in facts[0]


def test_category_sanitizing_to_empty_falls_back_to_untagged(isolated_paths):
    mr.add_memory("plain fact", category="]]] \n")
    facts = mr.list_memory()
    assert len(facts) == 1
    assert mr._fact_text(facts[0]) == "plain fact"
    assert "[" not in facts[0]  # "general" is the untagged default — no prefix written

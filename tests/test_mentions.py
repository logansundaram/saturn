"""mentions.py — @file resolution (quoted paths, trailing punctuation), the drag-and-drop
detector, and the clamped read that keeps one @file from blowing the context window."""

from core import mentions


def test_find_mentions_resolves_existing_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hello", encoding="utf-8")
    found = mentions.find_mentions(f"summarize @{f}")
    assert found == [str(f)]


def test_find_mentions_quoted_path_with_spaces(tmp_path):
    f = tmp_path / "my docs.md"
    f.write_text("hello", encoding="utf-8")
    found = mentions.find_mentions(f'read @"{f}" please')
    assert found == [str(f)]


def test_find_mentions_trailing_punctuation(tmp_path):
    import os

    f = tmp_path / "a.py"
    f.write_text("x = 1", encoding="utf-8")
    # Windows' isfile() tolerates a trailing dot, so the as-typed candidate may win there —
    # compare by file identity, not string equality.
    found = mentions.find_mentions(f"look at @{f}.")
    assert len(found) == 1 and os.path.samefile(found[0], f)
    found = mentions.find_mentions(f"(see @{f})")
    assert len(found) == 1 and os.path.samefile(found[0], f)


def test_unresolvable_mentions_ignored(tmp_path):
    assert mentions.find_mentions("email @handle and @nope/missing.txt") == []
    # A directory is not a file mention.
    assert mentions.find_mentions(f"@{tmp_path}") == []


def test_dropped_path(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF")
    assert mentions.dropped_path(str(f)) == str(f)
    assert mentions.dropped_path(f'"{f}"') == str(f)  # quoted drag shape
    assert mentions.dropped_path("just a sentence") is None
    assert mentions.dropped_path("") is None


def test_expand_block_and_extra_paths(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("alpha-contents", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("beta-contents", encoding="utf-8")
    block, paths = mentions.expand(f"check @{a}", extra_paths=[str(b)])
    assert paths == [str(a), str(b)]
    assert "alpha-contents" in block and "beta-contents" in block
    block, paths = mentions.expand("no mentions here")
    assert block == "" and paths == []


def test_expand_clamps_large_files(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("z" * (mentions._MAX_FILE_CHARS + 5000), encoding="utf-8")
    block, paths = mentions.expand(f"@{big}")
    assert paths == [str(big)]
    assert "truncated" in block
    assert len(block) < mentions._MAX_FILE_CHARS + 1000

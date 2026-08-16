"""The supported-family gate: the predicate, the size ladder, and the migration map."""

import pytest

from core import model_family as mf


class TestInFamily:
    def test_every_ladder_tag_is_in_family(self):
        for _key, tag in mf.SIZE_LADDER:
            assert mf.in_family(tag), tag

    def test_bare_family_name_matches(self):
        assert mf.in_family("qwen3.6")

    def test_case_insensitive(self):
        assert mf.in_family("QWEN3.5:0.8B")

    def test_matching_is_anchored_not_a_loose_prefix(self):
        # qwen3.50 must NOT satisfy a qwen3.5 test — the whole point of anchoring.
        assert not mf.in_family("qwen3.50:1b")

    @pytest.mark.parametrize(
        "tag",
        ["gemma4:e4b", "qwen3-coder:30b", "qwen3-embedding:8b", "qwen3.7:9b", "", "   "],
    )
    def test_outsiders_rejected(self, tag):
        assert not mf.in_family(tag)

    def test_none_is_not_in_family(self):
        assert not mf.in_family(None)


class TestLadder:
    def test_classes_match_the_ladder_order(self):
        assert mf.classes() == ("0.8b", "2b", "4b", "9b", "27b", "35b")

    def test_tag_for_round_trips(self):
        for key, tag in mf.SIZE_LADDER:
            assert mf.tag_for(key) == tag

    def test_tag_for_is_case_insensitive(self):
        assert mf.tag_for("0.8B") == "qwen3.5:0.8B"

    def test_tag_for_preserves_the_capital_b_tag(self):
        # Ollama tags are case-sensitive: the 0.8B tag must survive verbatim.
        assert mf.tag_for("0.8b") == "qwen3.5:0.8B"

    def test_tag_for_unknown_class_raises(self):
        with pytest.raises(KeyError):
            mf.tag_for("13b")

    def test_default_class_is_on_the_ladder(self):
        assert mf.DEFAULT_CLASS in mf.classes()


class TestMigrate:
    @pytest.mark.parametrize(
        "old,expected",
        [
            ("gemma4:e2b", "2b"),
            ("gemma4:e4b", "4b"),
            ("gemma4:12b", "9b"),
            ("gemma4:26b", "27b"),
            ("gemma4:31b", "27b"),
            ("qwen3-coder:30b", "27b"),
        ],
    )
    def test_legacy_table_is_exact(self, old, expected):
        assert mf.migrate(old) == expected

    def test_legacy_lookup_is_case_insensitive(self):
        assert mf.migrate("GEMMA4:E4B") == "4b"

    def test_unknown_tag_falls_back_to_the_size_parse(self):
        # |33 - 27.3| = 5.7 vs |33 - 36.0| = 3.0 -> nearest class is 35b, not 27b.
        assert mf.migrate("mystery:33b") == "35b"
        assert mf.migrate("mystery:3b") == "2b"

    def test_unparseable_tag_falls_back_to_the_default_class(self):
        assert mf.migrate("devstral-small-2:latest") == mf.DEFAULT_CLASS

    def test_migrate_always_returns_a_real_class(self):
        for tag in ["gemma4:e4b", "mystery:33b", "junk", "", None]:
            assert mf.migrate(tag) in mf.classes()

"""The per-model confidence overlay: what /confidence tune and /confidence set write."""

import json

import pytest


@pytest.fixture
def store(isolated_paths):
    from core import confidence_store

    confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
    confidence_store._reset_cache()
    yield confidence_store
    confidence_store._reset_cache()


class TestRoundTrip:
    def test_written_values_come_back(self, store):
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="tuned", tokens=1200, prompts=55)
        rec = store.entry_for("qwen3.8:27b")
        assert rec["enter"] == 0.31
        assert rec["exit"] == 0.52
        assert rec["source"] == "tuned"
        assert rec["tokens"] == 1200
        assert rec["prompts"] == 55
        assert rec["at"]                       # stamped by the store

    def test_lookup_is_case_insensitive(self, store):
        store.write_entry("QWEN3.8:27B", 0.31, 0.52, source="manual")
        assert store.entry_for("qwen3.8:27b")["enter"] == 0.31

    def test_unknown_model_is_none(self, store):
        assert store.entry_for("qwen3.5:9b") is None

    def test_writing_one_model_leaves_another_alone(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="manual")
        assert store.entry_for("qwen3.5:9b")["enter"] == 0.20
        assert store.entry_for("qwen3.8:27b")["enter"] == 0.31

    def test_rewriting_a_model_replaces_its_entry(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.5:9b", 0.44, 0.66, source="tuned")
        rec = store.entry_for("qwen3.5:9b")
        assert (rec["enter"], rec["exit"], rec["source"]) == (0.44, 0.66, "tuned")


class TestClear:
    def test_clear_removes_only_the_named_model(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="manual")
        assert store.clear_entry("qwen3.5:9b") is True
        assert store.entry_for("qwen3.5:9b") is None
        assert store.entry_for("qwen3.8:27b") is not None

    def test_clearing_an_absent_model_reports_false(self, store):
        assert store.clear_entry("qwen3.5:2b") is False


class TestFailSoft:
    def test_a_garbled_file_reads_empty_and_records_the_problem(self, store):
        store.store_path().write_text("{not json", encoding="utf-8")
        store._reset_cache()
        assert store.read() == {}
        assert store.entry_for("qwen3.8:27b") is None
        assert store.load_problem()          # a confidence failure never costs the answer

    def test_a_missing_file_is_simply_empty(self, store):
        if store.store_path().exists():
            store.store_path().unlink()
        store._reset_cache()
        assert store.read() == {}
        assert store.load_problem() == ""

    def test_a_non_mapping_payload_is_ignored(self, store):
        store.store_path().write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        store._reset_cache()
        assert store.read() == {}

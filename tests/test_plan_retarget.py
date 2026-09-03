"""The namespace guard: read_file steps aimed at ingested documents (2026-09-02).

Two namespaces, one tool each — workspace files (read_file) and the knowledge base
(search_knowledge_base). A small planner reads "read X" as read_file even when the grounding
context lists X under the knowledge base (measured twice on qwen3.5:9b, before and after the
prompt rule). The engine knows both manifests exactly, so `plan_ops.retarget_knowledge_base_reads`
swaps the tool deterministically at plan and replan time. Offline: isolated manifests.
"""

import pytest

from core import plan_ops
from core import structured as st
from nodes import plan as pl
from stores import document_registry as dr


@pytest.fixture
def env(isolated_paths, monkeypatch):
    monkeypatch.setattr(dr, "_summarize", lambda content, filename: "stub")
    ws = isolated_paths / "database" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    dr.register_rag_document("welcome-to-saturn.md", "Welcome to Saturn")
    dr.register_rag_document("guides/onboarding.pdf", "Onboarding")
    return ws


def _step(label, tool):
    return {"step_id": 1, "label": label, "status": "pending", "intended_tool": tool,
            "result": None, "needs_resolution": False}


def test_read_of_an_ingested_document_becomes_a_knowledge_base_search(env):
    steps = [_step("Read welcome-to-saturn.md and summarize it", "read_file")]
    out = plan_ops.retarget_knowledge_base_reads(steps)
    assert out[0]["intended_tool"] == "search_knowledge_base"
    assert out[0]["label"] == "Read welcome-to-saturn.md and summarize it"


def test_basename_match_is_case_insensitive_for_nested_documents(env):
    steps = [_step("Read ONBOARDING.PDF", "read_file")]
    assert plan_ops.retarget_knowledge_base_reads(steps)[0]["intended_tool"] == "search_knowledge_base"


def test_a_same_named_workspace_file_keeps_read_file(env):
    (env / "welcome-to-saturn.md").write_text("local copy", encoding="utf-8")
    steps = [_step("Read welcome-to-saturn.md", "read_file")]
    assert plan_ops.retarget_knowledge_base_reads(steps)[0]["intended_tool"] == "read_file"


def test_unrelated_reads_and_other_tools_are_untouched(env):
    steps = [_step("Read notes.txt", "read_file"),
             _step("Search for welcome-to-saturn.md", "find_files"),
             _step("Think about welcome-to-saturn.md", None)]
    out = plan_ops.retarget_knowledge_base_reads(steps)
    assert [s["intended_tool"] for s in out] == ["read_file", "find_files", None]


def test_a_stem_never_claims_a_document(env):
    dr.register_rag_document("notes.md", "n")
    steps = [_step("Read the notes file", "read_file")]  # 'notes' alone is not 'notes.md'
    assert plan_ops.retarget_knowledge_base_reads(steps)[0]["intended_tool"] == "read_file"


def test_no_knowledge_base_means_no_change(isolated_paths):
    steps = [_step("Read welcome-to-saturn.md", "read_file")]
    assert plan_ops.retarget_knowledge_base_reads(steps)[0]["intended_tool"] == "read_file"


def test_plan_node_applies_the_guard(env, monkeypatch):
    draft = st._PlanOut.model_validate({"plan": [
        {"description": "Read welcome-to-saturn.md", "tool": "read_file", "needs_resolution": False}
    ]})
    monkeypatch.setattr(pl, "structured", lambda *a, **k: draft)
    out = pl.plan_node({"plan": [], "context": "", "current_query": "read welcome to saturn"})
    assert out["plan"][0]["intended_tool"] == "search_knowledge_base"

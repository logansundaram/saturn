"""
Plan recipes (stores/recipes.py + node_registry.plan.seed_next_plan) — save/load/list/delete,
step normalization, and the one-shot planner seed.
"""

import pytest

from stores.recipes import (
    delete_recipe,
    list_recipes,
    load_recipe,
    normalize_steps,
    save_recipe,
)

_PLAN = [
    {"step_id": 1, "label": "search the web", "status": "done", "intended_tool": "web_search"},
    {"step_id": 2, "label": "read the page", "status": "skipped", "intended_tool": "web_extract"},
    {"step_id": 3, "label": "", "status": "pending", "intended_tool": None},  # dropped
    {"step_id": 4, "label": "synthesize", "status": "pending", "intended_tool": None},
]


def test_normalize_strips_run_state_keeps_shape():
    steps = normalize_steps(_PLAN)
    assert steps == [
        {"label": "search the web", "intended_tool": "web_search"},
        {"label": "read the page", "intended_tool": "web_extract"},
        {"label": "synthesize", "intended_tool": None},
    ]


def test_save_load_list_delete_round_trip(isolated_paths):
    path = save_recipe("Weekly Brief!", "what changed this week?", _PLAN)
    assert path.exists()
    assert path.stem == "Weekly-Brief"  # unsafe chars sanitized

    payload = load_recipe("Weekly Brief!")
    assert payload is not None
    assert payload["query"] == "what changed this week?"
    assert len(payload["steps"]) == 3

    listed = list_recipes()
    assert [r["name"] for r in listed] == ["Weekly-Brief"]

    assert delete_recipe("Weekly-Brief")
    assert load_recipe("Weekly-Brief") is None
    assert not delete_recipe("Weekly-Brief")


def test_save_refuses_empty_plan(isolated_paths):
    with pytest.raises(ValueError):
        save_recipe("empty", "q", [{"label": "  ", "intended_tool": None}])


def test_seed_next_plan_is_one_shot():
    from node_registry import plan as plan_mod

    n = plan_mod.seed_next_plan(
        [{"label": "search", "intended_tool": "web_search"}, {"label": ""}]
    )
    assert n == 1
    # the next plan_node pass consumes the seed without touching a model
    delta = plan_mod.plan_node({"context": "", "current_query": "q"})
    assert delta["plan"] == [
        {"step_id": 1, "label": "search", "status": "pending", "intended_tool": "web_search"}
    ]
    assert plan_mod._SEEDED_STEPS is None  # consumed


def test_seed_with_no_usable_steps_arms_nothing():
    from node_registry import plan as plan_mod

    assert plan_mod.seed_next_plan([{"label": "   "}]) == 0
    assert plan_mod._SEEDED_STEPS is None


def test_seed_consumed_for_its_own_query():
    from node_registry import plan as plan_mod

    plan_mod.seed_next_plan([{"label": "search", "intended_tool": "web_search"}], "the query")
    delta = plan_mod.plan_node({"context": "", "current_query": "the query"})
    assert delta["plan"][0]["label"] == "search"
    assert plan_mod._SEEDED_STEPS is None


def test_stale_seed_does_not_hijack_a_different_query(monkeypatch):
    # If the armed recipe turn dies before plan_node runs (Ctrl-C/exception in grounding), the
    # leftover seed must not silently become the plan for the user's NEXT, unrelated question —
    # in lockstep mode that would strongly direct the agent at recipe steps with no UI hint why.
    from node_registry import plan as plan_mod

    class _NoModel:
        def invoke(self, prompt):
            raise RuntimeError("tests never reach a model")

    monkeypatch.setattr(plan_mod, "get_plan_model", lambda: _NoModel())
    n = plan_mod.seed_next_plan(
        [{"label": "search", "intended_tool": "web_search"}], "the recipe query"
    )
    assert n == 1
    delta = plan_mod.plan_node({"context": "", "current_query": "something else entirely"})
    # drafted (here: the no-model generic fallback), NOT the recipe's seeded steps
    assert delta["plan"][0]["label"] == "Resolve the user's request"
    assert plan_mod._SEEDED_STEPS is None  # the stale seed is dropped, not left armed


def test_seed_with_no_valid_steps_disarms():
    from node_registry import plan as plan_mod

    plan_mod.seed_next_plan([{"label": "search", "intended_tool": "web_search"}], "q")
    # Re-arming with nothing valid drops the previous seed instead of leaving it dangling.
    assert plan_mod.seed_next_plan([{"label": "   "}], "q2") == 0
    assert plan_mod._SEEDED_STEPS is None and plan_mod._SEEDED_QUERY is None

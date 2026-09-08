"""
The quick path (2026-09-08 — PLAN.md "The common-case contract"):

  - core/complexity.py : the request-side check — which turns skip the planner and the judge.
  - nodes/quick.py     : the router node (answer / one read-only call / hand-over), its routing,
                         and the update_plan return route.
  - core/prime.py      : the quick lineage's user message is byte-identical to the node's.

No test reaches an LLM: the `structured` seam is monkeypatched at the node's namespace.
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from core import complexity as cx
from core.structured import QuickDecision
from nodes import quick as qk


# ── core/complexity: the request-side check ──────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "hello",
    "explain to me the axioms of probability",
    "Who is the CEO of OpenAI?",
    "What is the price of gold per ounce?",
    "what time is it",
    "review the emails I received today",
    "read notes.md and tell me who is on call",
    "Search the knowledge base: what is the standard maintenance window?",
    "What is my favorite text editor?",
    "write me a story",                       # prose, not a file
    "draft us a toast for the wedding",
    "what is 892.5 divided by 3.4?",          # one calculate call — traceable on the quick path
])
def test_simple_requests_take_the_quick_path(text):
    assert cx.plan_reason(text) == ""
    assert cx.is_simple(text)


@pytest.mark.parametrize("text, reason", [
    ("write me a story and save it to story.txt", "change"),
    ("write me a story to a file", "change"),
    ("Create a file called gate_probe.txt containing exactly this text: gate check one", "change"),
    ("In the workspace file gate_probe.txt, change the word 'one' to 'two'.", "edit"),
    ("rename report.md to final.md", "change"),
    ("replace the title in report.md", "edit"),
    ("Remember that my favorite text editor is Helix.", "change"),
    ("make an appointment for me tomorrow to get groceries", "change"),
    ("remind me at 5pm to call mom", "change"),
    ("delete this file /Users/me/Downloads/x.dmg", "change"),
    ("Read east.csv and tell me the total", "computed figure"),
    ("what is (742 + 96) * 0.85?", "computed figure"),
    ("Read locator.txt, then open the file it names.", "reference"),
    ("Ask me which colour to use", "asked"),
    ("Read east.csv and west.csv and tell me which is longer", "several paths"),
    ("list the workspace, then read the newest file", "several parts"),
    ("search my notes for the recycling day and tell me the pickup time; also check the calendar",
     "several parts"),
])
def test_requests_that_need_the_engine_are_planned(text, reason):
    why = cx.plan_reason(text)
    assert reason in why, (text, why)
    assert not cx.is_simple(text)


def test_the_prose_exemption_does_not_cover_a_persisted_write():
    assert cx.is_simple("write me a poem")
    assert not cx.is_simple("write me a poem and save it")
    assert not cx.is_simple("write me a poem into poem.txt")
    assert not cx.is_simple("write me a note in notes.md")


def test_benchmark_shapes_route_as_the_benchmark_expects():
    """Grounding baits and injection probes are lookups (quick); every gate, fabrication and
    memory-write probe still reaches the engine — the gate coverage grade depends on it."""
    import benchmark as b

    assert all(cx.is_simple(q) for q in b.INJECTION_PROBES)
    assert cx.is_simple(b.MEMORY_RECALL_ASK) and cx.is_simple(b.MEMORY_PLANT_PROBE)
    for q in b.GATE_PROBES + b.FABRICATION_PROBES + [
        b.MEMORY_RECALL_REMEMBER, b.MEMORY_SUPERSEDE_FIRST, b.MEMORY_SUPERSEDE_SECOND,
    ]:
        assert not cx.is_simple(q), q
    # Five of six grounding baits are quick; "the next TOTAL solar eclipse" trips the
    # aggregation vocabulary and plans — the safe direction, so it is tolerated, not fixed.
    assert sum(cx.is_simple(q) for q in b.GROUNDING_BAIT) == 5


# ── nodes/quick: the router node ─────────────────────────────────────────────────────────────


def _state(plan=None, query="who is the CEO of OpenAI?", **kw):
    base = {"messages": [HumanMessage(query)], "plan": plan or [], "current_query": query,
            "context": "", "context_stable": "STABLE", "context_dynamic": "", "iteration": 0,
            "replans": 0, "route": "", "reasoning": ""}
    base.update(kw)
    return base


def _decide(monkeypatch, decisions):
    """Feed the router's verdicts in order; record every prompt it was asked with."""
    seen = []
    it = iter(decisions)

    def fake(role, messages, schema, fmt, shape, default=None, attempts=3):
        seen.append({"role": role, "messages": messages, "fmt": fmt, "shape": shape})
        return next(it)

    monkeypatch.setattr(qk, "structured", fake)
    return seen


def test_answer_lands_at_synthesize_with_no_call(monkeypatch):
    seen = _decide(monkeypatch, [QuickDecision(tool="answer")])
    out = qk.quick_node(_state(query="hello"))
    assert out["route"] == "quick" and out["iteration"] == 1
    assert "messages" not in out and "plan" not in out
    assert qk.route_after_quick({**_state(query="hello"), **out}) == "synthesize"
    # Prompt-cache order: [quick system][stable grounding][request] — the first two are the
    # primed lineage, the request its own message.
    msgs = seen[0]["messages"]
    assert msgs[1].content == "STABLE"
    assert msgs[2].content == "User request:\nhello"
    assert seen[0]["role"] == "tool_caller"
    assert "answer" in seen[0]["fmt"]["properties"]["tool"]["enum"]


def test_a_read_only_tool_becomes_a_pending_step_and_a_gated_call(monkeypatch):
    _decide(monkeypatch, [QuickDecision(tool="web_search", arguments={"q": "CEO of OpenAI"})])
    st = _state()
    out = qk.quick_node(st)
    step = out["plan"][-1]
    assert step["intended_tool"] == "web_search" and step["result"] is None
    assert step["label"].startswith(qk.QUICK_LABEL)
    call = out["messages"][0].tool_calls[0]
    assert call["name"] == "web_search"
    assert call["args"] == {"query": "CEO of OpenAI"}  # alias coerced onto the real schema
    assert qk.route_after_quick({**st, **out, "messages": st["messages"] + out["messages"]}) == "approval"


def test_the_round_comes_back_to_quick_and_the_observation_rides_the_next_prompt(monkeypatch):
    seen = _decide(monkeypatch, [QuickDecision(tool="answer")])
    plan = [{"step_id": 1, "label": qk.QUICK_LABEL + "web_search(query='x')", "status": "done",
             "intended_tool": "web_search", "result": "Sam Altman is the CEO", "needs_resolution": False}]
    st = _state(plan=plan, route="quick")
    assert qk.route_after_update_plan(st) == "quick"
    assert qk.route_after_update_plan({**st, "route": "plan"}) == "rectify"
    out = qk.quick_node(st)
    msgs = seen[0]["messages"]
    assert msgs[3].content.startswith("Result of quick lookup: web_search")
    assert "Sam Altman" in msgs[3].content
    assert qk.route_after_quick({**st, **out}) == "synthesize"


def test_an_out_of_set_tool_hands_the_turn_over(monkeypatch):
    _decide(monkeypatch, [QuickDecision(tool="create_calendar_event", arguments={"title": "x"})])
    st = _state(query="make an appointmenet for me tomrrow")
    out = qk.quick_node(st)
    assert out["route"] == "plan"
    assert out["reasoning"].startswith(qk.ESCALATE_TOOL)
    assert "messages" not in out
    # Nothing gathered yet -> the planner drafts fresh.
    assert qk.route_after_quick({**st, **out}) == "plan"


def test_a_hand_over_after_a_lookup_keeps_the_gathered_steps_through_replan(monkeypatch):
    _decide(monkeypatch, [QuickDecision(tool="write_file", arguments={})])
    plan = [{"step_id": 1, "label": "quick lookup: read_file(file_path='a.txt')", "status": "done",
             "intended_tool": "read_file", "result": "hello", "needs_resolution": False}]
    st = _state(plan=plan, route="quick")
    out = qk.quick_node(st)
    assert out["route"] == "plan"
    assert qk.route_after_quick({**st, **out}) == "replan"


def test_nothing_parseable_fails_toward_the_engine(monkeypatch):
    _decide(monkeypatch, [QuickDecision(tool="", arguments={})])
    out = qk.quick_node(_state())
    assert out["route"] == "plan" and out["reasoning"].startswith(qk.ESCALATE_UNPARSED)


def test_a_guarded_outcome_ends_the_turn_without_a_retry(monkeypatch):
    seen = _decide(monkeypatch, [QuickDecision(tool="web_search", arguments={"query": "x"})])
    plan = [{"step_id": 1, "label": "quick lookup: web_search(query='x')", "status": "skipped",
             "intended_tool": "web_search", "result": "Execution declined by the user.",
             "needs_resolution": False}]
    st = _state(plan=plan, route="quick")
    out = qk.quick_node(st)
    assert seen == []  # no model call
    assert out["route"] == "quick" and "messages" not in out
    assert qk.route_after_quick({**st, **out}) == "synthesize"


def test_an_error_observation_hands_the_turn_over(monkeypatch):
    seen = _decide(monkeypatch, [])
    plan = [{"step_id": 1, "label": "quick lookup: web_search(query='x')", "status": "error",
             "intended_tool": "web_search", "result": "Error calling web_search: boom",
             "needs_resolution": False}]
    st = _state(plan=plan, route="quick")
    out = qk.quick_node(st)
    assert seen == []
    assert out["route"] == "plan" and out["reasoning"].startswith(qk.ESCALATE_ERROR)
    assert qk.route_after_quick({**st, **out}) == "replan"


def test_the_call_budget_lets_the_model_answer_but_not_call(monkeypatch):
    seen = _decide(monkeypatch, [QuickDecision(tool="read_mail", arguments={"id": "3"})])
    plan = [
        {"step_id": i, "label": f"quick lookup: read_mail(id='{i}')", "status": "done",
         "intended_tool": "read_mail", "result": f"mail {i}", "needs_resolution": False}
        for i in (1, 2, 3)
    ]
    st = _state(plan=plan, route="quick", query="review my emails")
    out = qk.quick_node(st)
    assert "budget is spent" in seen[0]["messages"][-1].content
    assert out["route"] == "plan" and out["reasoning"].startswith(qk.ESCALATE_BUDGET)
    assert qk.route_after_quick({**st, **out}) == "replan"
    # ... and "answer" at the budget still lands.
    _decide(monkeypatch, [QuickDecision(tool="answer")])
    out = qk.quick_node(st)
    assert out["route"] == "quick" and qk.route_after_quick({**st, **out}) == "synthesize"


def test_arguments_that_do_not_fit_the_schema_hand_over_instead_of_calling(monkeypatch):
    _decide(monkeypatch, [QuickDecision(tool="read_file", arguments={"nonsense": 1})])
    out = qk.quick_node(_state(query="read it"))
    assert out["route"] == "plan" and "messages" not in out


def test_the_iteration_cap_lands_a_runaway_quick_loop(monkeypatch):
    from config import get_config

    _decide(monkeypatch, [QuickDecision(tool="answer")])
    st = _state(iteration=get_config().max_iterations)
    out = qk.quick_node(st)
    assert qk.route_after_quick({**st, **out}) == "synthesize"


# ── routing out of ground ────────────────────────────────────────────────────────────────────


def test_route_after_ground_honors_overrides_seeded_plans_and_the_knob(monkeypatch):
    from config import get_config

    assert qk.route_after_ground(_state(query="hello")) == "quick"
    assert qk.route_after_ground(_state(query="save this to a.txt")) == "plan"
    assert qk.route_after_ground(_state(query="save this to a.txt", route="quick")) == "quick"
    assert qk.route_after_ground(_state(query="hello", route="plan")) == "plan"
    seeded = [{"step_id": 1, "label": "x", "status": "pending", "intended_tool": None,
               "result": None, "needs_resolution": False}]
    assert qk.route_after_ground(_state(query="hello", plan=seeded)) == "plan"
    cfg = get_config()
    monkeypatch.setitem(cfg._data.setdefault("runtime", {}), "quick_path", False)
    assert qk.route_after_ground(_state(query="hello")) == "plan"


def test_the_graph_edges_agree_with_the_node_and_the_state_resets_route():
    from app import graph as g
    from app.session import _initial_state

    assert _initial_state()["route"] == ""
    # The wiring names every destination route_after_quick / route_after_ground can return.
    import inspect

    src = inspect.getsource(g.build_agent)
    for dest in ("quick", "plan", "approval", "synthesize", "replan", "rectify"):
        assert f'"{dest}"' in src


# ── the primed lineage ───────────────────────────────────────────────────────────────────────


def test_the_quick_lineage_is_byte_identical_to_the_nodes_prompt(monkeypatch):
    from core import prime

    seen = _decide(monkeypatch, [QuickDecision(tool="answer")])
    qk.quick_node(_state(query="hello"))
    names = [n for n, _r, _m in prime.lineages("STABLE")]
    assert names[0] == "quick"
    _n, role, msgs = prime.lineages("STABLE")[0]
    assert role == "tool_caller"
    assert msgs[0].content == seen[0]["messages"][0].content
    assert msgs[1].content == seen[0]["messages"][1].content == "STABLE"


def test_the_quick_prompt_lists_only_read_only_tools_and_names_the_rest():
    from core.messages import quick_sys_msg, quick_tool_names
    from core.plan_context import QUICK_TOOLS
    from tools.registry import risk_of

    names = quick_tool_names()
    assert set(names) <= set(QUICK_TOOLS)
    assert all(risk_of(n) == "read_only" for n in names)
    assert "ask_user" not in names
    text = quick_sys_msg().content
    assert "- web_search {query: string}" in text
    for other in ("write_file", "edit_file", "ask_user", "create_calendar_event"):
        assert other in text  # named as hand-over choices, never described as callable


def test_a_policy_raise_drops_a_tool_from_the_quick_set(monkeypatch):
    from core.messages import quick_tool_names
    from tools import registry

    monkeypatch.setitem(registry.TOOL_RISK, "web_search", "side_effecting")
    assert "web_search" not in quick_tool_names()


# ── the REPL / CLI overrides ─────────────────────────────────────────────────────────────────


def test_quick_and_plan_commands_hand_a_forced_turn_to_the_repl(capsys):
    from commands._framework import CommandContext
    from commands.plan import _plan, _quick

    ctx = CommandContext(state={}, make_initial_state=dict, db_path="")
    _quick(ctx, ["what", "does", "total", "mean?"])
    assert ctx.pending_turn == ("quick", "what does total mean?")
    ctx.pending_turn = None
    _plan(ctx, ["who", "is", "the", "CEO", "of", "OpenAI?"])
    assert ctx.pending_turn == ("plan", "who is the CEO of OpenAI?")
    ctx.pending_turn = None
    _plan(ctx, ["review"])  # a real subcommand is never a request
    assert ctx.pending_turn is None
    _quick(ctx, [])
    assert ctx.pending_turn is None
    assert "usage" in capsys.readouterr().out


def test_headless_flags_force_the_route():
    from app.cli import _build_parser

    args = _build_parser().parse_args(["-p", "hello", "--plan"])
    assert args.plan and not args.quick
    args = _build_parser().parse_args(["-q", "hello", "--quick"])
    assert args.quick

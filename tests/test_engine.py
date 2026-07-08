"""
The plan/execute engine (2026-07-03 transplant) — offline coverage of the load-bearing seams:

  - core/structured.py : the hardened parse layer (JSON salvage, lenient models, tool-name
                         normalization, planner-output -> step dicts).
  - core/tool_args.py  : alias arg coercion + text-format call recovery + the schema hint.
  - core/plan_context.py : the curated per-step contexts built off the plan data bus.
  - nodes/execute.py   : step dispatch (reasoning / write gate / tool call) + routing.
  - nodes/rectify.py   : every deterministic short-circuit, in priority order, + routing.
  - nodes/replan.py    : done-steps-kept merge, renumbering, rectify reset, empty-redraft
                         degradation.

No test reaches an LLM: the `structured`/text seams are monkeypatched at each node's namespace.
"""

import types

import pytest
from langchain.messages import AIMessage, HumanMessage

from core import plan_context, structured as st, tool_args
from nodes import execute as ex
from nodes import rectify as rc
from nodes import replan as rp


def _step(step_id, tool=None, result=None, status="pending", needs_resolution=False, label=None):
    return {"step_id": step_id, "label": label or f"step {step_id}", "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": needs_resolution}


def _state(plan, **kw):
    base = {"messages": [HumanMessage("the request")], "plan": plan,
            "current_query": "the request", "context": "", "iteration": 0, "replans": 0}
    base.update(kw)
    return base


# ── core/structured: the hardened parse layer ─────────────────────────────────────────────────


def test_extract_json_salvages_prose_wrapped_output():
    assert st._extract_json('Sure! {"rectify": true, "reasoning": "x"} Hope that helps') == (
        '{"rectify": true, "reasoning": "x"}'
    )
    assert st._extract_json("no braces at all") == "no braces at all"


def test_norm_tool_maps_synonyms_and_rejects_junk():
    valid = {"calculate", "list_directory", "search_files", "web_search"}
    assert st.norm_tool("calc", valid) == "calculate"
    assert st.norm_tool("Calculator", valid) == "calculate"
    assert st.norm_tool("ls", valid) == "list_directory"
    assert st.norm_tool("grep", valid) == "search_files"
    assert st.norm_tool("none", valid) is None
    assert st.norm_tool("made_up_tool", valid) is None
    assert st.norm_tool(None, valid) is None
    # A pipe-separated echo of the enum ("read_file|write_file") takes the first token.
    assert st.norm_tool("web_search|calculate", valid) == "web_search"


def test_norm_tool_degenerate_emissions_never_crash():
    # A leading pipe / bare "=" / whitespace once raised IndexError out of to_steps and killed
    # the whole turn — exactly the ignores-the-grammar input class this layer exists to absorb.
    valid = {"calculate", "web_search"}
    assert st.norm_tool("|web_search", valid) is None
    assert st.norm_tool("=", valid) is None
    assert st.norm_tool("   ", valid) is None


def test_invoke_kwargs_carry_num_ctx_for_ollama_roles():
    # Invoke-time `options` REPLACES ChatOllama's constructor options (the only home of the
    # configured num_ctx), so the window must ride every options dict or the daemon silently
    # reverts to its ~2048 default and front-truncates the prompt.
    kw = st._invoke_kwargs("planner", None, 0.0)
    assert kw, "the shipped config binds planner to Ollama"
    assert kw["options"]["temperature"] == 0.0
    assert kw["options"].get("num_ctx", 0) > 0


def test_to_steps_builds_data_bus_dicts():
    draft = st._PlanOut(plan=[
        st._PlanItem(description="Read a.txt", tool="read_file", needs_resolution=False),
        st._PlanItem(description="", tool="calc"),  # blank descriptions drop
        st._PlanItem(description="Total it", tool="calc", needs_resolution=True),
    ])
    steps = st.to_steps(draft)
    assert [s["step_id"] for s in steps] == [1, 2]
    assert steps[0]["intended_tool"] == "read_file" and steps[0]["result"] is None
    assert steps[1]["intended_tool"] == "calculate"  # synonym normalized against the registry
    assert steps[1]["needs_resolution"] is True
    assert all(s["status"] == "pending" for s in steps)


def test_plan_format_enum_tracks_given_names():
    fmt = st.plan_format(["read_file", "web_search"])
    enum = fmt["properties"]["plan"]["items"]["properties"]["tool"]["enum"]
    assert "none" in enum and "read_file" in enum and "web_search" in enum


# ── core/tool_args: alias coercion + text-call recovery ───────────────────────────────────────


def test_coerce_args_maps_aliases_onto_real_schema():
    assert tool_args.coerce_args("read_file", {"path": "notes.md"}) == {"file_path": "notes.md"}
    assert tool_args.coerce_args("calculate", {"expr": "1+2"}) == {"expression": "1+2"}
    assert tool_args.coerce_args("run_shell", {"cmd": "dir"}) == {"command": "dir"}
    out = tool_args.coerce_args(
        "edit_file", {"file": "a.txt", "old": "x", "new": "y", "replace_all": True}
    )
    assert out == {"file_path": "a.txt", "old_string": "x", "new_string": "y",
                   "replace_all": True}


def test_coerce_args_missing_required_returns_none():
    assert tool_args.coerce_args("write_file", {"file_path": "a.txt"}) is None  # no content
    assert tool_args.coerce_args("read_file", {"nonsense": "x"}) is None
    assert tool_args.coerce_args("read_file", "not-a-dict") is None


def test_coerce_args_empty_string_is_a_value_where_it_means_something():
    # Deleting text and creating an empty file are legitimate calls — "" must count as present
    # for edit_file.new_string / write_file.content, not as a missing value to retry forever.
    assert tool_args.coerce_args(
        "edit_file", {"file_path": "a.txt", "old_string": "TODO", "new_string": ""}
    ) == {"file_path": "a.txt", "old_string": "TODO", "new_string": ""}
    assert tool_args.coerce_args(
        "write_file", {"file_path": "empty.txt", "content": ""}
    ) == {"file_path": "empty.txt", "content": ""}
    # But an empty ANCHOR is still missing (edit_file would refuse it anyway).
    assert tool_args.coerce_args(
        "edit_file", {"file_path": "a.txt", "old_string": "", "new_string": "x"}
    ) is None


def test_coerce_args_zero_required_and_unknown_tools():
    # Tools with no required args succeed on empty input; optionals ride along when present.
    assert tool_args.coerce_args("current_time", {}) == {}
    assert tool_args.coerce_args("recall", {"query": "tz"}) == {"query": "tz"}
    # An unknown (MCP) tool passes its dict through unchanged — the remote schema is not ours.
    assert tool_args.coerce_args("mcp_gh_search", {"q": "x"}) == {"q": "x"}


def test_parse_text_call_recovers_both_dialects():
    assert tool_args.parse_text_call('file_path: <|"|>notes.md<|"|>') == {"file_path": "notes.md"}
    assert tool_args.parse_text_call('{"query": "solar eclipse"}') == {"query": "solar eclipse"}
    assert tool_args.parse_text_call("no call here") is None


def test_schema_hint_has_shape_for_known_and_generic_for_unknown():
    assert "read_file(file_path=" in tool_args.schema_hint("read_file", "bad args")
    assert "mcp_x(" in tool_args.schema_hint("mcp_x", "bad args")


# ── core/plan_context: the curated contexts off the data bus ─────────────────────────────────


def test_results_block_caps_and_numbers():
    plan = [_step(1, "read_file", result="A" * 1000, status="done"), _step(2)]
    block = plan_context.results_block(plan)
    assert block.startswith("Results from earlier steps")
    assert "…(truncated)" in block
    assert plan_context.results_block([_step(1)]) == ""  # nothing run -> empty


def test_exec_context_carries_previous_step_callout():
    plan = [_step(1, "read_file", result="42 rows", status="done"), _step(2, "calculate")]
    ctx = plan_context.exec_context(_state(plan), plan[1])
    assert "User's overall request: the request" in ctx
    assert '"the previous step"' in ctx and "42 rows" in ctx
    assert ctx.rstrip().endswith("Your current step: step 2")


def test_plan_txt_marks_pending_and_done():
    plan = [_step(1, "read_file", result="data", status="done"), _step(2, "calculate")]
    txt = plan_context.plan_txt(plan)
    assert "[DONE] tool=read_file" in txt and "result: data" in txt
    assert "[PENDING] tool=calculate" in txt


def test_plan_txt_caps_done_results():
    # Several ~12k clamped observations rendered uncapped would overflow the judge/replan
    # prompts on a small window and front-truncate the system prompt.
    plan = [_step(1, "read_file", result="A" * 20000, status="done")]
    txt = plan_context.plan_txt(plan)
    assert "…(truncated)" in txt and len(txt) < 2000


def test_exec_context_callout_is_bounded():
    plan = [_step(1, "read_file", result="B" * 20000, status="done"), _step(2, "calculate")]
    ctx = plan_context.exec_context(_state(plan), plan[1])
    assert len(ctx) < 8000, "the previous-step callout must not re-send a full clamped observation"


# ── nodes/execute: step dispatch + routing ────────────────────────────────────────────────────


def test_execute_reasoning_step_records_result_inline(monkeypatch):
    monkeypatch.setattr(ex, "_reasoning_call", lambda ctx: ("east is larger", None))
    plan = [_step(1, None)]
    out = ex.execute_node(_state(plan))
    step = out["plan"][0]
    assert step["result"] == "east is larger" and step["status"] == "done"
    assert out["iteration"] == 1
    assert isinstance(out["messages"][0], AIMessage)
    # No tool call -> rectify, not approval.
    assert ex.route_after_execute({"messages": out["messages"]}) == "rectify"


def test_execute_tool_step_emits_corrected_call_for_approval(monkeypatch):
    monkeypatch.setattr(
        ex, "_generate_tool_call", lambda tool, ctx: ({"file_path": "a.txt"}, None, None)
    )
    plan = [_step(1, "read_file")]
    out = ex.execute_node(_state(plan))
    (msg,) = out["messages"]
    (call,) = msg.tool_calls
    assert call["name"] == "read_file" and call["args"] == {"file_path": "a.txt"}
    assert out["plan"][0]["result"] is None, "the recorder (update_plan) writes the result"
    assert out["plan"][0]["status"] == "active"
    assert ex.route_after_execute({"messages": out["messages"]}) == "approval"


def test_execute_write_gate_skip_short_circuits(monkeypatch):
    monkeypatch.setattr(ex, "_write_gate", lambda state, step: "skipped write: value missing")
    called = []
    monkeypatch.setattr(ex, "_generate_tool_call",
                        lambda *a: called.append(1) or (None, "x", None))
    plan = [_step(1, "read_file", result="data", status="done"), _step(2, "write_file")]
    out = ex.execute_node(_state(plan))
    step = out["plan"][1]
    assert step["status"] == "skipped" and step["result"].startswith("skipped write")
    assert not called, "a gated write never generates a call"


def test_execute_arg_failure_lands_as_error(monkeypatch):
    monkeypatch.setattr(
        ex, "_generate_tool_call", lambda tool, ctx: (None, "error: no tool call emitted", None)
    )
    plan = [_step(1, "read_file")]
    out = ex.execute_node(_state(plan))
    assert out["plan"][0]["status"] == "error"
    assert out["plan"][0]["result"].startswith("error:")


def test_execute_unknown_tool_degrades_to_reasoning(monkeypatch):
    monkeypatch.setattr(ex, "_reasoning_call", lambda ctx: ("reasoned through it", None))
    plan = [_step(1, "not_a_registered_tool")]
    out = ex.execute_node(_state(plan))
    assert out["plan"][0]["result"] == "reasoned through it"


def test_execute_no_step_left_is_a_noop():
    assert ex.execute_node(_state([_step(1, result="x", status="done")])) == {}


def test_execute_does_not_mutate_state_plan(monkeypatch):
    monkeypatch.setattr(ex, "_reasoning_call", lambda ctx: ("r", None))
    plan = [_step(1, None)]
    before = [dict(s) for s in plan]
    ex.execute_node(_state(plan))
    assert plan == before


# ── the semantic write gate (execute._write_gate) ─────────────────────────────────────────────


def test_write_gate_passes_when_nothing_gathered():
    plan = [_step(1, "write_file")]
    assert ex._write_gate(_state(plan), plan[0]) is None


def test_write_gate_skips_on_empty_upstream():
    plan = [_step(1, "search_files", result="[]", status="done"), _step(2, "write_file")]
    blocked = ex._write_gate(_state(plan), plan[1])
    assert blocked and "upstream result was empty" in blocked


def test_write_gate_mechanical_plan_never_asks_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no LLM for a read-and-compute plan")

    monkeypatch.setattr(ex, "structured", boom)
    plan = [_step(1, "read_file", result="rows: 1,2,3", status="done"), _step(2, "write_file")]
    assert ex._write_gate(_state(plan), plan[1]) is None


def test_write_gate_blocks_when_judge_says_absent(monkeypatch):
    monkeypatch.setattr(
        ex, "structured",
        lambda *a, **k: st.WriteGate(present=False, evidence="nothing matches"),
    )
    plan = [_step(1, "search_files", result="unrelated hit", status="done"),
            _step(2, "write_file")]
    blocked = ex._write_gate(_state(plan), plan[1])
    assert blocked and "not present in the gathered" in blocked


def test_write_gate_mechanical_zero_is_a_value_not_an_absence(monkeypatch):
    # "compute 17-17, write the result" — a computed 0 (or an empty-looking literal) on a
    # mechanical read-and-compute plan must write, not skip: the gate only arms when a search
    # ran or a step failed.
    def boom(*a, **k):
        raise AssertionError("no LLM for a read-and-compute plan")

    monkeypatch.setattr(ex, "structured", boom)
    plan = [_step(1, "calculate", result="0", status="done"), _step(2, "write_file")]
    assert ex._write_gate(_state(plan), plan[1]) is None


def test_write_gate_armed_zero_goes_to_the_judge_not_the_skip(monkeypatch):
    # With a search upstream, a computed 0 is judged (present/absent), never mechanically
    # skipped as "empty" — 0 is a value.
    monkeypatch.setattr(
        ex, "structured", lambda *a, **k: st.WriteGate(present=True, evidence="count is 0")
    )
    plan = [_step(1, "search_files", result="3 hits", status="done"),
            _step(2, "calculate", result="0", status="done"),
            _step(3, "write_file")]
    assert ex._write_gate(_state(plan), plan[2]) is None


def test_write_gate_fails_closed_when_judge_unavailable(monkeypatch):
    # When the judge can't produce a verdict, structured() returns the caller's `default`. That
    # default must be fail-CLOSED (present=False) so an unverifiable write is skipped, not waved
    # through — and the skip must disclose it was the gate, not a confirmed absence.
    monkeypatch.setattr(ex, "structured", lambda *a, default=None, **k: default)
    plan = [_step(1, "search_files", result="unrelated hit", status="done"),
            _step(2, "write_file")]
    blocked = ex._write_gate(_state(plan), plan[1])
    assert blocked and "fail-closed" in blocked and "could not verify" in blocked


def test_write_gate_default_is_fail_closed():
    # The WriteGate parse model itself defaults present=False (a partial parse missing the field
    # blocks, not allows) — belt-and-suspenders with the constrained decoder's required fields.
    assert st.WriteGate().present is False


# ── nodes/rectify: the deterministic short-circuits, in priority order ────────────────────────


def _no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("this branch must not reach the LLM")

    monkeypatch.setattr(rc, "structured", boom)


def test_rectify_guarded_outcome_cancels_remaining(monkeypatch):
    _no_llm(monkeypatch)
    plan = [_step(1, "write_file", result="skipped write: missing", status="skipped"),
            _step(2, "read_file")]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is False
    assert out["plan"][1]["status"] == "cancelled"
    assert "guarded" in out["reasoning"]
    # The original plan is untouched (the cancel works on a copy).
    assert plan[1]["result"] is None


def test_rectify_concrete_pending_passes_without_llm(monkeypatch):
    _no_llm(monkeypatch)
    plan = [_step(1, "read_file", result="data", status="done"), _step(2, "calculate")]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is False and "pending" in out["reasoning"]


def test_rectify_resolution_found_asks_replan(monkeypatch):
    monkeypatch.setattr(
        rc, "structured",
        lambda *a, **k: st.ResolutionCheck(found=True, evidence="listing names b.csv"),
    )
    plan = [_step(1, "search_files", result="b.csv:1: data", status="done"),
            _step(2, "read_file", needs_resolution=True)]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is True and "Resolve the next step's reference" in out["reasoning"]


def test_rectify_resolution_absent_cancels(monkeypatch):
    monkeypatch.setattr(
        rc, "structured",
        lambda *a, **k: st.ResolutionCheck(found=False, evidence="nothing matches"),
    )
    plan = [_step(1, "search_knowledge_base", result="unrelated chunk", status="done"),
            _step(2, "read_file", needs_resolution=True)]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is False
    assert out["plan"][1]["status"] == "cancelled"
    assert "not found" in out["plan"][1]["result"]


def test_rectify_mechanical_resolution_skips_the_check(monkeypatch):
    _no_llm(monkeypatch)  # a read-only pointer chain must resolve without the LLM presence check
    plan = [_step(1, "read_file", result="next: b.txt", status="done"),
            _step(2, "read_file", needs_resolution=True)]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is True


def test_rectify_write_step_exempt_from_resolution(monkeypatch):
    _no_llm(monkeypatch)
    # A deferred WRITE is not force-resolved (the write gate judges it) — with pending steps and
    # no failure this falls through to the concrete-pending pass.
    plan = [_step(1, "read_file", result="data", status="done"),
            _step(2, "write_file", needs_resolution=True)]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is False and "pending" in out["reasoning"]


def test_rectify_dead_end_retry_once(monkeypatch):
    _no_llm(monkeypatch)
    plan = [_step(1, "search_files", result="No matches for /tokn/ in '.'", status="done")]
    out = rc.rectify_node(_state(plan, replans=0))
    assert out["rectify"] is True and "came up empty" in out["reasoning"]
    # Budget: past 2 replans the dead end falls through to the LLM verdict instead.
    monkeypatch.setattr(rc, "structured",
                        lambda *a, **k: st.RectifyBool(rectify=False, reasoning="stop"))
    out = rc.rectify_node(_state(plan, replans=2))
    assert out["rectify"] is False


def test_rectify_read_file_miss_is_genuine_absence(monkeypatch):
    # A read_file miss is NOT retryable — it falls to the LLM verdict.
    monkeypatch.setattr(rc, "structured",
                        lambda *a, **k: st.RectifyBool(rectify=False, reasoning="absent"))
    plan = [_step(1, "read_file", result="Error calling read_file: not found", status="error")]
    out = rc.rectify_node(_state(plan))
    assert out["rectify"] is False


def test_rectify_budget_spent_short_circuits(monkeypatch):
    _no_llm(monkeypatch)
    out = rc.rectify_node(_state([_step(1)], replans=rc.MAX_REPLANS))
    assert out["rectify"] is False and "budget" in out["reasoning"]


def test_rectify_guarded_cancel_wins_over_spent_budget(monkeypatch):
    # Branch order is load-bearing (gotcha #8): a gate rejection recorded AFTER the replan
    # budget is spent must still cancel the remaining steps ("cancelled: a prior guarded
    # action…"), not leave them mislabeled "never ran".
    _no_llm(monkeypatch)
    plan = [_step(1, "run_shell", result="Execution declined by the user.", status="skipped"),
            _step(2, "read_file")]
    out = rc.rectify_node(_state(plan, replans=rc.MAX_REPLANS))
    assert out["rectify"] is False
    assert out["plan"][1]["status"] == "cancelled"
    assert "guarded" in out["reasoning"]


def test_rectify_shell_exit_code_is_anchored_to_the_header(monkeypatch):
    # A SUCCESSFUL run whose output merely mentions "exit code 128" mid-transcript is not a
    # dead end — only the [exit code N] header run_shell itself prepends counts.
    monkeypatch.setattr(rc, "structured",
                        lambda *a, **k: st.RectifyBool(rectify=False, reasoning="fine"))
    ok_run = "[exit code 0]\nci log: previous job failed with exit code 128"
    plan = [_step(1, "run_shell", result=ok_run, status="done")]
    assert rc.rectify_node(_state(plan))["rectify"] is False
    # A genuinely failed run IS the retryable dead end.
    _no_llm(monkeypatch)
    plan = [_step(1, "run_shell", result="[exit code 2] (no output)", status="done")]
    out = rc.rectify_node(_state(plan, replans=0))
    assert out["rectify"] is True and "came up empty" in out["reasoning"]


def test_route_after_rectify_bounds_and_routes():
    assert rc.route_after_rectify(_state([_step(1)], rectify=True)) == "replan"
    assert rc.route_after_rectify(_state([_step(1)], rectify=False)) == "plan_gate"
    assert rc.route_after_rectify(
        _state([_step(1, result="x", status="done")], rectify=False)) == "synthesize"
    assert rc.route_after_rectify(_state([_step(1)], iteration=999)) == "synthesize"
    assert rc.route_after_rectify(_state([_step(1)], replans=rc.MAX_REPLANS)) == "synthesize"


# ── nodes/replan: done steps kept, pending redrafted ──────────────────────────────────────────


def test_replan_keeps_done_and_redrafts_pending(monkeypatch):
    draft = st._PlanOut(plan=[
        st._PlanItem(description="Read b.csv", tool="read_file"),
        st._PlanItem(description="step 1", tool="read_file"),  # dup of a done step -> dropped
    ])
    monkeypatch.setattr(rp, "structured", lambda *a, **k: draft)
    plan = [_step(1, "read_file", result="names b.csv", status="done"),
            _step(2, "read_file", needs_resolution=True)]
    out = rp.replan_node(_state(plan, reasoning="resolve the reference", replans=0))
    labels = [s["label"] for s in out["plan"]]
    assert labels == ["step 1", "Read b.csv"], "done kept verbatim; pending replaced; dup dropped"
    assert [s["step_id"] for s in out["plan"]] == [1, 2]
    assert out["plan"][0]["result"] == "names b.csv"
    assert out["replans"] == 1 and out["rectify"] is False


def test_replan_empty_redraft_keeps_plan(monkeypatch):
    monkeypatch.setattr(rp, "structured", lambda *a, **k: st._PlanOut())
    plan = [_step(1, "read_file", result="x", status="done"), _step(2, "calculate")]
    out = rp.replan_node(_state(plan, reasoning="fix it"))
    assert "plan" not in out, "an empty redraft leaves the plan untouched"
    assert out["replans"] == 1 and out["rectify"] is False

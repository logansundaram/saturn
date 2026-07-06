"""
Unit tests for the brittle pure-logic core of the loop — the parts whose bugs don't crash but
silently corrupt behaviour (the plan data bus, history compaction, observation clamping).

Deliberately dependency-light: every function under test is pure (or near-pure), so these need no
Ollama, no network, no checkpointer. Runnable two ways:
    python tests/test_core.py     # standalone, no pytest required
    pytest tests/                 # if pytest is installed

Coverage (2026-07-03 engine transplant):
  - current_step / update_plan_node : the plan-as-data-bus pointer + the mechanical recorder
    (result written onto the current step, status derived from the observation).
  - _compact_history : older turns collapse but the most recent scratchpad is retained.
  - _clamp_observation : large tool output can't overflow the context window.
  - planner tool catalog : built from the live registry (no drift when tools are added).
"""

import os
import sys

# Repo root on the path so `import state` etc. resolve when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.messages import HumanMessage, AIMessage, ToolMessage

from core.state import current_step, unfinished_steps, incident_steps
from nodes.update_plan import update_plan_node
from nodes.tools import _clamp_observation, _MAX_OBSERVATION


def _step(step_id, tool=None, result=None, status="pending"):
    return {"step_id": step_id, "label": f"step {step_id}", "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


# --- the data-bus pointer: result-is-None marks the step being executed ---------------------
def test_current_step_is_first_without_result():
    plan = [_step(1, result="done a", status="done"), _step(2), _step(3)]
    assert current_step(plan)["step_id"] == 2
    assert current_step([_step(1, result="x", status="done")]) is None
    assert current_step([]) is None and current_step(None) is None


def test_unfinished_and_incident_views():
    plan = [
        _step(1, result="ok", status="done"),
        _step(2, result="declined", status="skipped"),
        _step(3),
    ]
    assert [s["step_id"] for s in unfinished_steps(plan)] == [3]
    assert [s["step_id"] for s in incident_steps(plan)] == [2]


# --- the mechanical recorder: observation -> current step's result + stamped status ----------
def _tool_round(plan, observation, name="web_search", stamp=None):
    """A state as it looks after tools ran: the call answered by a trailing ToolMessage.
    `stamp` mirrors the producer's structural outcome (nodes/tools.py `saturn_status`)."""
    kwargs = {"additional_kwargs": {"saturn_status": stamp}} if stamp else {}
    return {
        "plan": plan,
        "messages": [
            HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": name, "args": {}, "id": "c1"}]),
            ToolMessage(observation, tool_call_id="c1", name=name, **kwargs),
        ],
    }


def test_update_plan_records_result_on_current_step():
    plan = [_step(1, "web_search"), _step(2, "calculate")]
    out = update_plan_node(_tool_round(plan, "search results here"))["plan"]
    assert out[0]["result"] == "search results here"
    assert out[0]["status"] == "done"
    assert out[1]["result"] is None, "only the current step records"


def test_update_plan_reads_stamped_incident_statuses():
    plan = [_step(1, "run_shell")]
    out = update_plan_node(
        _tool_round(plan, "Error calling run_shell: boom", stamp="error")
    )["plan"]
    assert out[0]["status"] == "error"
    plan = [_step(1, "web_search")]
    out = update_plan_node(
        _tool_round(plan, "Air-gap is ON — this operation would send data.", stamp="blocked")
    )["plan"]
    assert out[0]["status"] == "blocked"
    plan = [_step(1, "write_file")]
    decline = ("Execution declined by the user. Do not retry this action; tell the user you "
               "did not perform it.")
    out = update_plan_node(
        _tool_round(plan, decline, name="write_file", stamp="skipped")
    )["plan"]
    assert out[0]["status"] == "skipped", "a gate rejection records as a skipped incident"


def test_update_plan_never_sniffs_status_from_observation_text():
    """A successful read of a file whose CONTENT starts with an error/blocked word must stay
    `done` — the status is the producer's stamp, never the observation text (the old prefix
    sniffing failed a step over its own data)."""
    plan = [_step(1, "read_file")]
    out = update_plan_node(
        _tool_round(plan, "ERROR: disk full at 03:12\nrest of the log", name="read_file")
    )["plan"]
    assert out[0]["status"] == "done"
    plan = [_step(1, "read_file")]
    out = update_plan_node(
        _tool_round(plan, "Blocked IPs: 10.0.0.1, 10.0.0.2", name="read_file")
    )["plan"]
    assert out[0]["status"] == "done"


def test_update_plan_decline_prefix_fallback_without_stamp():
    """Belt-and-braces: an UNSTAMPED decline still records as skipped off the DECLINE_TEXT
    prefix (the one textual fallback kept)."""
    from nodes.approval import DECLINE_TEXT

    plan = [_step(1, "write_file")]
    out = update_plan_node(_tool_round(plan, DECLINE_TEXT, name="write_file"))["plan"]
    assert out[0]["status"] == "skipped"


def test_set_status_keeps_the_pointer_pairing():
    """The plan-review editor's status verb must keep gotcha #6 intact: a TERMINAL status on an
    un-run step also stamps a result (else execute re-selects it by `result is None` and RUNS
    the step the user just skipped), and pending clears the result so a step is runnable."""
    from core import plan_ops

    plan = [_step(1), _step(2)]
    out = plan_ops.set_status(plan, 1, "skipped")
    assert out[0]["result"] is not None
    assert current_step(out)["step_id"] == 2, "the skipped step is no longer the pointer"
    back = plan_ops.set_status(out, 1, "pending")
    assert back[0]["result"] is None, "back to pending -> runnable again"
    # A completed step marked done keeps its recorded result untouched.
    done = [_step(1, result="real output", status="done")]
    kept = plan_ops.set_status(done, 1, "done")
    assert kept[0]["result"] == "real output"


def test_update_plan_does_not_mutate_input():
    plan = [_step(1, "web_search")]
    before = [dict(s) for s in plan]
    update_plan_node(_tool_round(plan, "result"))
    assert plan == before, "update_plan must work on a copy, not mutate state in place"


def test_update_plan_noop_without_observation_or_pending_step():
    # No trailing ToolMessage -> nothing to record.
    assert update_plan_node({"plan": [_step(1)], "messages": [HumanMessage("q")]}) == {}
    # Every step already has a result -> nothing to record onto.
    done = [_step(1, result="x", status="done")]
    assert update_plan_node(_tool_round(done, "obs")) == {}


# --- _compact_history: keep the most recent scratchpad, collapse older turns ----------------
def test_compact_history_keeps_recent_scratchpad_drops_old():
    from agent import _compact_history

    history = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "a"}]),
        ToolMessage(content="old-result", tool_call_id="a", name="web_search"),
        AIMessage(content="answer1"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "b"}]),
        ToolMessage(content="recent-result", tool_call_id="b", name="web_search"),
        AIMessage(content="answer2"),
    ]
    kept = _compact_history(history)  # keep_recent_turns=1
    blob = "|".join(str(m.content) for m in kept)
    assert "old-result" not in blob, "old turn's tool output should be dropped"
    assert "recent-result" in blob, "most recent turn's scratchpad must be retained"
    assert "q1" in blob and "answer1" in blob, "older turn collapses to its Q&A, not vanishes"
    # No orphaned tool-call AIMessage from the old turn survived.
    assert not any(getattr(m, "tool_calls", None) and "old" in str(m.content) for m in kept)


# --- _clamp_observation: large tool output can't blow the context window --------------------
def test_clamp_short_observation_untouched():
    s = "small result"
    assert _clamp_observation(s) == s


def test_clamp_long_observation_truncated_with_marker():
    s = "x" * (_MAX_OBSERVATION + 5000)
    out = _clamp_observation(s)
    assert len(out) < len(s)
    assert "truncated" in out
    assert out.startswith("x") and out.endswith("x")  # head + tail preserved


# --- planner prompt stays in sync with the live registry ------------------------------------
def test_planner_prompt_lists_every_registered_tool():
    from core import messages
    from tools import registry

    prompt = messages.planner_sys_msg().content
    for t in registry.tool:
        assert t.name in prompt, f"{t.name} missing from planner prompt (drift!)"
    # ...and the same names reach the constrained decoder's enum + the normalizer.
    from core.structured import plan_format, registered_tools

    enum = plan_format(sorted(registered_tools()))
    enum = enum["properties"]["plan"]["items"]["properties"]["tool"]["enum"]
    for t in registry.tool:
        assert t.name in enum, f"{t.name} missing from the plan schema enum (drift!)"


# --- #5: registration decorator keeps the registry views consistent -------------------------
def test_registry_views_consistent():
    from tools import registry

    # Every registered tool has a risk tier, and tools_by_name covers the whole list.
    assert {t.name for t in registry.tool} == set(registry.tools_by_name)
    assert all(t.name in registry.TOOL_RISK for t in registry.tool)
    # Risk tiers are valid, and unknown tools fail safe to the strictest tier.
    from tools.toolspec import RISK_TIERS

    assert all(v in RISK_TIERS for v in registry.TOOL_RISK.values())
    assert registry.risk_of("a-tool-that-does-not-exist") == "destructive"
    # The retrieval flag rode along with the tool that declared it.
    assert "search_knowledge_base" in registry.RETRIEVAL_TOOLS


# --- #8: model-presence normalization for the startup health check --------------------------
def test_model_present_normalizes_latest_tag():
    from core.llms import _model_present

    assert _model_present("qwen3.5:9b", {"qwen3.5:9b"})
    assert _model_present("foo", {"foo:latest"})        # implicit :latest
    assert _model_present("foo:latest", {"foo"})        # and the reverse
    assert not _model_present("missing:9b", {"other:9b"})


# --- #3: calculate tames float epsilon without capping real precision -----------------------
def test_calculate_tames_epsilon_keeps_precision():
    from tools.calculator import calculate

    call = lambda e: calculate.invoke({"expression": e})
    assert call("672.34999999999999 + 0") == "672.35"   # epsilon artifact removed
    assert call("37.0") == "37"                          # whole-number float -> int
    assert call("2+3*4") == "14"
    assert call("1/3") == "0.333333333333"               # real precision preserved (not 0.3333)


# --- Tier 2 #7: LLM compaction folds old turns, keeps the recent one, and fails safe -----------
def test_compaction_folds_old_keeps_recent_turn():
    from core import compaction

    orig = compaction._llm_summary
    compaction._llm_summary = lambda older: f"SUMMARY({len(older)})"  # stub: offline + deterministic
    try:
        msgs = [
            HumanMessage("q1"), AIMessage("a1"),
            HumanMessage("q2"), AIMessage("a2"),
            HumanMessage("q3"), AIMessage("a3"),
        ]
        new, stats = compaction.summarize_messages(msgs)  # keep_recent_turns=1
        assert stats["summarized_turns"] == 2, "the two older turns fold"
        assert compaction.is_summary(new[0]), "folded turns become one summary message at the head"
        assert new[-2].content == "q3" and new[-1].content == "a3", "recent turn kept verbatim"
        # A prior summary chains forward (folded into the next) rather than accreting.
        chained = new + [HumanMessage("q4"), AIMessage("a4")]
        n2, _ = compaction.summarize_messages(chained)
        assert sum(compaction.is_summary(m) for m in n2) == 1, "summaries don't pile up"
        # Only the recent turn present -> nothing to fold.
        _, s2 = compaction.summarize_messages([HumanMessage("only"), AIMessage("a")])
        assert s2["summarized_turns"] == 0
    finally:
        compaction._llm_summary = orig


def test_compaction_llm_failure_leaves_history_intact():
    from core import compaction

    orig = compaction._llm_summary

    def boom(older):
        raise RuntimeError("model down")

    compaction._llm_summary = boom
    try:
        msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2")]
        new, stats = compaction.summarize_messages(msgs)
        assert new is msgs and stats["summarized_turns"] == 0, "a failed summary must not lose history"
    finally:
        compaction._llm_summary = orig


def test_compaction_steer_note_is_not_a_turn_boundary():
    """plan_gate's standalone mid-turn steer note is a HumanMessage but NOT a turn boundary
    (core.state.is_turn_start — the same rule _compact_history already obeys): with
    keep_recent_turns=1 the boundary must land on the steered turn's REAL question. Slicing at
    the steer note would fold the question + its pre-steer scratchpad into the lossy summary
    while the correction survives as the apparent current question."""
    from core import compaction
    from core.state import STEER_PREFIX

    orig = compaction._llm_summary
    compaction._llm_summary = lambda older: f"SUMMARY({len(older)})"  # stub: offline + deterministic
    try:
        msgs = [
            HumanMessage("q1"), AIMessage("a1"),
            HumanMessage("q2"),
            AIMessage("", tool_calls=[{"name": "web_search", "args": {}, "id": "t1"}]),
            ToolMessage("results", tool_call_id="t1"),
            HumanMessage(f"{STEER_PREFIX} no, the OTHER repo"),
            AIMessage("final"),
        ]
        new, stats = compaction.summarize_messages(msgs)
        # Boundary = q2: only the q1 turn folds, never the steered turn's question.
        assert stats["summarized_turns"] == 1
        assert compaction.is_summary(new[0])
        assert new[1].content == "q2"
        # The steered turn survives verbatim: its scratchpad AND the steer note itself.
        assert any(isinstance(m, ToolMessage) for m in new)
        assert any(str(m.content).startswith(STEER_PREFIX) for m in new)

        # Stats side of the same predicate: once the steered turn ages into the older slice it
        # must count as ONE folded turn — the steer note (and the prior summary) must not
        # inflate the user-facing "compacted N earlier turn(s)" notice.
        chained = new + [HumanMessage("q3"), AIMessage("a3")]
        n2, s2 = compaction.summarize_messages(chained)
        assert s2["summarized_turns"] == 1
        assert sum(compaction.is_summary(m) for m in n2) == 1
    finally:
        compaction._llm_summary = orig


# --- Tier 2 #6: type-ahead queue + Esc steering (InputQueue char handling, no console) ----------
def test_typeahead_queues_enter_terminated_lines_fifo():
    from tui import typeahead

    q = typeahead.InputQueue()
    for ch in "first":
        q._on_char(ch)
    q._on_char("\r")
    for ch in "second":
        q._on_char(ch)
    q._on_char("\n")
    assert q.pending()
    assert q.pop() == "first" and q.pop() == "second", "queue drains FIFO"
    assert q.pop() is None


def test_typeahead_blank_not_queued_and_backspace_edits():
    from tui import typeahead

    q = typeahead.InputQueue()
    for ch in "   ":
        q._on_char(ch)
    q._on_char("\r")
    assert not q.pending(), "a blank line never queues"
    for ch in "abx":
        q._on_char(ch)
    q._on_char("\x08")  # backspace removes the x
    q._on_char("c")
    q._on_char("\r")
    assert q.pop() == "abc"


def test_escape_with_text_steers_empty_reviews():
    from core import plan_ops as interrupts
    from tui import typeahead

    c = interrupts.get_pause_controller()
    c.clear()
    q = typeahead.InputQueue()
    for ch in "use the 2023 figures":
        q._on_char(ch)
    q._on_escape()
    req = c.peek()
    assert req.source == "steer" and req.reason == "use the 2023 figures"
    assert q._buffer == "", "the typed line is consumed as a steer, not left to queue"
    c.clear()
    q._on_escape()  # empty buffer
    assert c.peek().source == "user", "empty Esc asks for a plan-review pause"
    c.clear()


def test_plan_gate_injects_steer_and_consumes_request():
    from core import plan_ops as interrupts
    from nodes.plan_gate import plan_gate_node, route_after_gate

    c = interrupts.get_pause_controller()
    c.clear()
    c.request("steer", "focus on cost, not schedule")
    upd = plan_gate_node({"messages": [HumanMessage("q")], "plan": [], "iteration": 1})
    assert "messages" in upd, "a steer is injected as a message update"
    assert "focus on cost" in upd["messages"][0].content
    assert not c.pending(), "the steer request is consumed (won't re-inject next boundary)"
    # The steer arms a replan with the correction as the revision instruction, and the gate's
    # router honors it (the remaining steps are redrafted around the user's words).
    assert upd["rectify"] is True and "focus on cost" in upd["reasoning"]
    assert route_after_gate({"rectify": True}) == "replan"
    assert route_after_gate({}) == "execute"
    assert route_after_gate({"aborted": True}) == "synthesize"


# --- Tier 2 #5: write_file diff preview (pure diff classification) ------------------------------
def test_write_diff_new_file_is_all_additions():
    from tui import ui

    rows, is_new, _hidden = ui._diff_lines("___does_not_exist___.txt", "alpha\nbeta\n", True)
    assert is_new
    assert [k for k, _ in rows] == ["hunk", "add", "add"]


def test_write_diff_overwrite_shows_delete_and_add():
    from config import get_config
    from tui import ui

    ws = get_config().path("workspace")
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / "___difftest___.txt"
    p.write_text("one\ntwo\n", encoding="utf-8")
    try:
        rows, is_new, _hidden = ui._diff_lines("___difftest___.txt", "one\nTWO\n", True)
        kinds = [k for k, _ in rows]
        assert not is_new
        assert "del" in kinds and "add" in kinds, "a changed line shows as a delete + an add"
    finally:
        p.unlink()


# --- config.persist: surgical YAML edit keeps comments + round-trips, fails safe ---------------
def test_set_yaml_scalar_replaces_value_and_keeps_comments():
    import yaml
    from config import _set_yaml_scalar, _CONFIG_PATH

    text = _CONFIG_PATH.read_text(encoding="utf-8")
    out = _set_yaml_scalar(text, "runtime.max_iterations", 12)
    assert yaml.safe_load(out)["runtime"]["max_iterations"] == 12, "the value is updated on parse"
    # The change is surgical: only the one line differs, every comment/line else is untouched.
    a, b = text.splitlines(), out.splitlines()
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 1, "exactly one line changed"
    assert b[diff[0]].lstrip().startswith("max_iterations:")


def test_set_yaml_scalar_preserves_inline_comment():
    from config import _set_yaml_scalar

    src = "runtime:\n  num_ctx: null  # the context window comment\n"
    out = _set_yaml_scalar(src, "runtime.num_ctx", 8192)
    assert out == "runtime:\n  num_ctx: 8192  # the context window comment\n"


def test_set_yaml_scalar_quotes_stringy_values():
    import yaml
    from config import _set_yaml_scalar

    src = "web:\n  provider: auto\n"
    # A value that looks like a bool must round-trip as a string, not True.
    out = _set_yaml_scalar(src, "web.provider", "true")
    assert isinstance(yaml.safe_load(out)["web"]["provider"], str)


def test_set_yaml_scalar_missing_key_raises():
    from config import _set_yaml_scalar

    try:
        _set_yaml_scalar("runtime:\n  max_iterations: 8\n", "runtime.nope", 1)
    except KeyError:
        return
    raise AssertionError("a missing leaf must raise KeyError, not silently no-op")


def test_set_yaml_scalar_container_value_raises():
    from config import _set_yaml_scalar

    try:
        _set_yaml_scalar("tiers:\n  workstation: x\n", "tiers", {"a": 1})
    except ValueError:
        return
    raise AssertionError("persisting a container must raise ValueError")


def test_dump_scalar_renders_yaml_literals():
    from config import _dump_scalar

    assert _dump_scalar(None) == "null"
    assert _dump_scalar(True) == "true" and _dump_scalar(False) == "false"
    assert _dump_scalar(12) == "12"
    assert _dump_scalar("auto") == "auto"
    assert _dump_scalar("true").startswith('"'), "a stringy bool gets quoted"
    assert _dump_scalar("a:b").startswith('"'), "punctuation forces quoting"


# --- LLM-call tracing: faithful (de)serialization + durable storage ----------------------------
def test_trace_msg_to_dict_normalizes_role_and_keeps_toolcalls():
    from langchain_core.messages import SystemMessage
    from stores.trace import _msg_to_dict

    d = _msg_to_dict(SystemMessage(content="the context"))
    assert d["role"] == "system" and d["content"] == "the context"
    ai = _msg_to_dict(AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"q": "x"}, "id": "1"}]))
    assert ai["role"] == "ai" and ai["tool_calls"][0]["name"] == "web_search"


def test_trace_msg_to_dict_flags_truncation():
    from langchain_core.messages import HumanMessage as HM
    from stores.trace import _msg_to_dict, _LLM_MSG_CAP

    d = _msg_to_dict(HM(content="x" * (_LLM_MSG_CAP + 100)))
    assert len(d["content"]) == _LLM_MSG_CAP and d["truncated"] == _LLM_MSG_CAP + 100


def test_trace_llm_output_extracts_content_and_tokens():
    from langchain_core.outputs import LLMResult, ChatGeneration
    from stores.trace import _llm_output

    msg = AIMessage(content="hello", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
    out, ptok, otok = _llm_output(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert out["content"] == "hello" and ptok == 12 and otok == 3


def test_tracer_records_and_reads_back_llm_calls():
    import os
    import sqlite3
    import tempfile
    from stores.trace import Tracer

    db = os.path.join(tempfile.mkdtemp(), "trace.sqlite")
    t = Tracer(db)
    rid = t.start_run("thread", "a query")
    t.log_llm_call(rid, "agent", "qwen3.5:9b", 1.5, 100, 20, "[]", '{"content":"hi"}', "ok")
    t.log_llm_call(rid, "synthesize", "qwen3.5:9b", 0.4, 50, 8, "[]", '{"content":"done"}', "ok")
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT seq, node, prompt_tokens FROM llm_calls WHERE run_id = ? ORDER BY seq", (rid,)
        ).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == [1, 2], "per-run seq increments"
    assert rows[0][1] == "agent" and rows[1][1] == "synthesize"


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())

"""KV-cache stability of the engine's prompts (2026-09-04).

Measured against the live daemon (Ollama 0.33 / qwen3.5:9b): llama-server restores a prompt
only up to a saved context checkpoint (N-1024 and N-4 of an earlier prompt, the restore point of
a lineage), so a prompt reuses its predecessor's prefill exactly when everything that changed
sits AFTER such a checkpoint. Three things broke that for every node: the per-result caps were
recomputed from the count of results (an earlier result's rendering changed when a later one
landed), the request rode INSIDE the grounding message (only N-1024 was ever reachable), and the
native tool bind rendered the tool schema into the system message (a full re-prefill per step).
This file pins the prefix-stable layouts, the stable/dynamic grounding split, the idle primes,
and the grammar-constrained argument path. Fully offline: every model seam is a stub.
"""

import json

from langchain.messages import AIMessage, HumanMessage

from core import llms


def _step(step_id, tool=None, result=None, status="pending", label=None):
    return {"step_id": step_id, "label": label or f"step {step_id}", "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


def _state(plan, **kw):
    base = {"messages": [HumanMessage("the request")], "plan": plan,
            "current_query": "the request", "context": "", "iteration": 0, "replans": 0}
    base.update(kw)
    return base


# ── results block: prefix-stable per-result caps ────────────────────────────────────────────


def _long_plan(n, size=5000):
    return [_step(i, "read_file", result=f"R{i}:" + "B" * size, status="done")
            for i in range(1, n + 1)]


# ── execute context: message-boundary layout ───────────────────────────────────────────────


# ── grounding: stable / dynamic split ───────────────────────────────────────────────────────


def test_grounding_node_splits_stable_and_per_turn_sections(isolated_paths, monkeypatch):
    from nodes import ground

    (isolated_paths / "database" / "workspace").mkdir(parents=True)
    (isolated_paths / "database" / "workspace" / "SATURN.md").write_text("be terse")
    monkeypatch.setattr(ground, "memory_context_split",
                        lambda q: ("- #1 likes tea", "- #2 [entities] tea shop", [2]))
    monkeypatch.setattr(ground, "mark_used", lambda ids: 0)
    msgs = [HumanMessage("earlier q"), AIMessage("earlier a"), HumanMessage("now")]
    out = ground.grounding_node({"messages": msgs, "current_query": "now", "attachments": "ATT"})
    stable, dynamic = out["context_stable"], out["context_dynamic"]
    assert out["context"] == stable + "\n\n" + dynamic
    assert stable.startswith("## Grounding context")
    assert "be terse" in stable and "likes tea" in stable
    for per_turn in ("tea shop", "ATT", "### Now"):  # (the recap section left with the plan engine)
        assert per_turn in dynamic and per_turn not in stable
    # the stable half is exactly what the idle prime rebuilds between turns
    assert ground.stable_grounding() == stable


def test_grounding_node_reads_the_memory_store_once_a_turn(isolated_paths, monkeypatch):
    """One selection returns both halves; a second call re-read and re-parsed the memory file
    on the first node of every turn."""
    from nodes import ground

    asked = []

    def split(q=""):
        asked.append(q)
        return "- #1 likes tea", "- #2 [entities] tea shop", [2]

    monkeypatch.setattr(ground, "memory_context_split", split)
    monkeypatch.setattr(ground, "mark_used", lambda ids: 0)
    out = ground.grounding_node({"messages": [], "current_query": "tea", "attachments": ""})
    assert asked == ["tea"]
    assert "likes tea" in out["context_stable"] and "tea shop" in out["context_dynamic"]


def test_memory_context_split_keeps_the_always_layers_query_independent(isolated_paths):
    from stores import memory_registry as mr

    mr.add_memory("prefers tabs", layer="user")
    mr.add_memory("the tea shop closes at five", layer="entities")
    always_a, matched_a, ids_a = mr.memory_context_split("when does the tea shop close")
    always_b, matched_b, ids_b = mr.memory_context_split("unrelated question")
    assert always_a == always_b and "prefers tabs" in always_a
    assert "tea shop" in matched_a and matched_b == ""
    assert ids_a and not ids_b


# ── plan / synthesize prompt layouts ────────────────────────────────────────────────────────


# ── execute: tool arguments under a grammar, no native bind ─────────────────────────────────


# ── the idle primes ─────────────────────────────────────────────────────────────────────────


def test_prime_sends_one_boundary_request_per_lineage(monkeypatch):
    """One lineage since the v2 loop (2026-09-27): the agent's, through the BOUND model, so the
    tool schemas the chat template renders are inside the cached prefix."""
    from core import prime

    class M:
        def __init__(self):
            self.calls = []
            self.bound = 0

        def bind_tools(self, tools):
            self.bound += 1
            return self

        def invoke(self, msgs, **kw):
            self.calls.append((msgs, kw))
            return AIMessage(content="")

    model = M()
    monkeypatch.setattr("core.llms.get_model", lambda: model)
    monkeypatch.setattr(prime, "ENABLED", True)
    n = prime.prime("STABLE")
    assert n == 1 == len(model.calls) == model.bound
    from core.messages import agent_sys_msg

    msgs, kw = model.calls[0]
    assert [m.content for m in msgs] == [agent_sys_msg().content, "STABLE"]
    assert kw["options"]["num_predict"] == 1
    assert kw["reasoning"] is True  # think ON: think-off adds tokens past the boundary
    assert kw["options"]["num_ctx"] == llms.invoke_kwargs(None, 0.0)["options"]["num_ctx"]


def test_prime_never_raises_and_reports_zero_when_the_daemon_is_down(monkeypatch):
    from core import prime

    class Down:
        def invoke(self, msgs, **kw):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("core.llms.get_model", lambda: Down())
    monkeypatch.setattr(prime, "ENABLED", True)
    assert prime.prime("STABLE") == 0


def test_priming_is_off_under_tests_and_the_config_knob(monkeypatch):
    from core import prime

    assert prime.ENABLED is False  # conftest: no test may reach a model
    assert prime.start_priming() is None
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: False)
    assert prime.start_priming() is None


def test_start_priming_runs_the_rebuild_on_a_daemon_thread(monkeypatch):
    from core import prime

    seen = []
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: True)
    monkeypatch.setattr(prime, "prime_now", lambda only=None: seen.append("primed") or 4)
    t = prime.start_priming()
    t.join(timeout=5)
    assert t.daemon and not t.is_alive() and seen == ["primed"]


def test_warm_up_thread_primes_after_the_weights_load(monkeypatch):
    from app import startup
    from core import prime

    seen = []
    monkeypatch.setattr(startup, "warm_model", lambda: seen.append("warm") or True)
    monkeypatch.setattr(prime, "prime_now", lambda only=None: seen.append(("prime", only)) or 1)
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: True)
    t = startup.start_warm_up()
    t.join(timeout=5)
    assert seen == ["warm", ("prime", ("agent",))]


def test_prime_stops_between_lineages_when_a_turn_starts(monkeypatch):
    from core import prime

    class M:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, msgs, **kw):
            self.calls += 1
            prime.set_busy(True)  # the user typed mid-sequence
            return AIMessage(content="")

    model = M()
    monkeypatch.setattr("core.llms.get_model", lambda: model)
    monkeypatch.setattr(prime, "ENABLED", True)
    try:
        assert prime.prime("STABLE") == 1 == model.calls
    finally:
        prime.set_busy(False)


def test_now_section_names_the_weekday_date_and_time():
    from datetime import datetime, timedelta, timezone

    from nodes.ground import now_section

    now = datetime(2026, 9, 29, 14, 5, tzinfo=timezone(timedelta(hours=-7)))
    assert now_section(now) == (
        "### Now\nTuesday 2026-09-29 (29 September 2026), 14:05 local time (UTC-07:00)")

# Tool-Call Output Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `write_file` / `edit_file` steps actually emit their tool call when the payload is long, stop the engine from burning a minute on retries that cannot succeed, and stop a transient write failure from being disclosed as "fabrication".

**Architecture:** The serving layer gains a second tool-call task (`tool_payload`) with a larger `num_predict` circuit breaker that the execute node selects for the payload-carrying tools; the execute node learns to read Ollama's `done_reason == "length"` and abandon the retry ladder with an honest error instead of re-rolling at a hotter temperature; the write gate stops arming on a prior failed *write* (only a failed *producer* can leave a value to bridge). Two adjacent robustness holes found in the same trace are closed: the judge's cap and a missing `numpy` dependency.

**Tech Stack:** Python 3.11+, LangGraph, langchain-ollama 1.1 (`response_metadata["done_reason"]`), pytest (fully offline — every model is a scripted stub).

**Spec:** This plan (the evidence section below is the spec; there is no separate design doc).

## Evidence (why these five tasks and not others)

Run 15 in `database/db.sqlite` (`write me a story and save it to a file named story.txt`, 2026-09-02 20:44) and run 10 (`give me some ideas for a social function and write it to juan.txt`, 20:38):

- Every failed `write_file` attempt in `llm_calls` has `output_tokens = 512`, `content = ""`, `tool_calls = []`. 512 is exactly `serving.TASKS["tool_args"].num_predict`. The model was writing the story *inside the tool-call arguments* and the daemon cut it off mid-JSON, so no call parsed.
- Live replay of `llm_calls.id = 61` (same messages, same model `qwen3.5:9b`): `num_predict=512` → `done_reason=length`, 0 tool calls, 19.7 s. `num_predict=4096` → `done_reason=stop`, `eval_count=690`, one `write_file` call with a 2,956-character story, 18.4 s. Script: `scratchpad/repro_write.py` from the diagnosing session.
- `nodes/execute._generate_tool_call` then retried twice more at temperature 0.5 and 0.7 with the **same cap** — three identical truncations, 30–45 s per step. `rectify` sent it to `replan`, which redrafted the identical step, which truncated again. Run 10 spent ~2 minutes this way before the user hit Ctrl-C.
- On the redraft in run 15, `_write_gate` armed because a prior step had `status == "error"` (the failed write itself), asked the judge whether "an original story" is present in the gathered results, got `present=false`, and skipped the step. The answer then told the user "I must not fabricate a story". The gate is meant to catch a value bridged over a failed *search/read*; a failed write of the same file is not an upstream data source.
- Run 16 `rectify` judge attempt 1: `output_tokens = 512`, unparseable (truncated JSON); attempt 2 happened to finish in 241. Same cap class, smaller blast radius.
- Run 19: `search_knowledge_base` → `Error calling search_knowledge_base: cosine_similarity requires numpy to be installed`. `numpy` is not installed in `.venv` and not declared in `pyproject.toml` or `requirements.txt`; `langchain_core.vectorstores.InMemoryVectorStore` (used by `stores/rag.py`) needs it at query time. Tests never reach the vector store, so CI cannot see this.
- The trust benchmark's only writes (`benchmark.py:106,108`) are one-line payloads, which is why the cap never bit there.

Not in scope (noted for later): `core/plan_context._RESULT_CAP = 800` truncates each earlier result in the results block, so a write step whose source is not the *immediately preceding* producer (the callout carries 4,000 chars) would copy a truncated value. Separate plan.

## Global Constraints

- Tests are fully offline: no test may reach Ollama, the network, or the embedder (`tests/conftest.py`; monkeypatch at each node's namespace).
- `core/serving.py` is a stdlib-only leaf. It must not import `core/plan_context` or anything project-side.
- `core/llms.py` is the egress chokepoint module; add only pure helpers there, no new network calls.
- `pyproject.toml` `dependencies` and `requirements.txt` must stay in sync (CI installs from the former; clone installs from the latter).
- Commit messages: `area: what changed`, lowercase. Every commit ends with the two trailers below.
- User-visible changes go under `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog).

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe
```

Run the whole suite before each commit: `python -m pytest tests/ -q` (≈4 s, ≈1300 tests, all must pass).

---

### Task 1: A `tool_payload` task with a larger cap, selected per tool

**Files:**
- Modify: `core/serving.py:44-51` (the `TASKS` table) and the module docstring
- Modify: `nodes/execute.py:342-352` (`_generate_tool_call`, the `_invoke_kwargs(...)` call)
- Test: `tests/test_serving.py`

**Interfaces:**
- Produces: `serving.TASKS["tool_payload"]` — `Task("tool_payload", strict=True, num_predict=4096, think=False)`; `serving.num_predict("tool_payload") == 4096`.
- Produces: `nodes.execute.PAYLOAD_TOOLS` — `frozenset(WRITE_TOOLS)` (i.e. `{"write_file", "edit_file"}`), and `nodes.execute._task_for_tool(name: str) -> str` returning `"tool_payload"` for those and `"tool_args"` otherwise. Task 2 reads the cap through `_task_for_tool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serving.py` (the file already imports `serving`, `structured`, `ex`, and `tools_by_name`; reuse the scripted-model shape from `test_generate_tool_call_arms_the_repeat_penalty_only_after_a_degenerate_draw` at ~line 170):

```python
def test_tool_payload_task_exists_and_is_larger_than_tool_args():
    """write_file/edit_file carry the WHOLE payload in their arguments (measured 2026-09-02:
    a 2,956-char story is ~690 tokens; the 512 tool_args cap cut every attempt at exactly 512
    with done_reason=length and no call parsed). The payload cap is still a circuit breaker."""
    assert serving.num_predict("tool_payload") >= 4096
    assert serving.num_predict("tool_payload") > serving.num_predict("tool_args")
    assert serving.thinks("tool_payload") is False
    assert serving.task_of("tool_payload").strict is True


def test_task_for_tool_routes_write_tools_to_the_payload_cap():
    assert ex._task_for_tool("write_file") == "tool_payload"
    assert ex._task_for_tool("edit_file") == "tool_payload"
    assert ex._task_for_tool("calculate") == "tool_args"
    assert ex._task_for_tool("mcp_remote_thing") == "tool_args"


def test_generate_tool_call_uses_the_payload_cap_for_write_file(monkeypatch):
    from langchain.messages import AIMessage

    seen = []

    class M:
        def bind_tools(self, tools):
            return self

        def invoke(self, msgs, **kw):
            seen.append(kw)
            return AIMessage(content="", tool_calls=[{
                "name": "write_file", "args": {"file_path": "a.txt", "content": "hi"},
                "id": "c1", "type": "tool_call"}])

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["write_file"], "ctx")
    assert failure is None and args["content"] == "hi"
    assert seen[0]["options"]["num_predict"] == serving.num_predict("tool_payload")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_serving.py -q -k "tool_payload or task_for_tool or payload_cap"`
Expected: 3 failures — `KeyError`/fallback for `tool_payload` (the fallback `Task` has `num_predict=512`, so the `>= 4096` assert fails), `AttributeError: module 'nodes.execute' has no attribute '_task_for_tool'`, and the `num_predict` assert.

- [ ] **Step 3: Add the task to `core/serving.py`**

In the `TASKS` dict, after the `"tool_args"` line:

```python
    "tool_args": Task("tool_args", strict=True, num_predict=512, think=False),
    # The payload-carrying tools (write_file, edit_file): the file's CONTENT rides inside the
    # arguments, so the bound is the size of a file, not of an argument list. Still a circuit
    # breaker — 4096 tokens is ~12-16 KB of text; a longer write is refused honestly by the
    # execute node's truncation branch instead of looping (measured 2026-09-02: every write of a
    # story/idea list cut at exactly 512 with done_reason=length and no call parsed).
    "tool_payload": Task("tool_payload", strict=True, num_predict=4096, think=False),
```

And in the module docstring's `num_predict` bullet, append one sentence: `The one exception to "well above healthy" was the tool-argument cap for write_file/edit_file, whose healthy generation IS the file — those use the tool_payload task.`

- [ ] **Step 4: Select the task per tool in `nodes/execute.py`**

Near the top of the module, after `_ATTEMPT_TEMPS`:

```python
# The tools whose ARGUMENTS carry the whole payload (a file's content). Their tool-call
# generation runs under the serving layer's `tool_payload` bound instead of `tool_args`; every
# other tool's arguments are a path, a pattern, a query, an expression — small by construction.
PAYLOAD_TOOLS = frozenset(WRITE_TOOLS)


def _task_for_tool(tool_name: str) -> str:
    """The serving task that bounds this tool's call generation."""
    return "tool_payload" if tool_name in PAYLOAD_TOOLS else "tool_args"
```

In `_generate_tool_call`, replace `task="tool_args"` in the `_invoke_kwargs(...)` call:

```python
    task = _task_for_tool(tool.name)
    for temp in _ATTEMPT_TEMPS:
        try:
            resp = generate(
                bound,
                [EXECUTE_TOOL_SYS, HumanMessage(content=block)],
                tag=_model_tag("tool_caller"),
                **_invoke_kwargs("tool_caller", None, temp, task=task,
                                 repetition=repetition),
            )
```

- [ ] **Step 5: Run the tests and the whole suite**

Run: `python -m pytest tests/test_serving.py -q -k "tool_payload or task_for_tool or payload_cap"` → 3 passed.
Run: `python -m pytest tests/ -q` → all pass (the existing `test_generate_tool_call_arms_the_repeat_penalty...` binds `calculate`, which still resolves to `tool_args`).

- [ ] **Step 6: Commit**

```bash
git add core/serving.py nodes/execute.py tests/test_serving.py
git commit -m "serving: tool_payload task — write_file/edit_file generate under a file-sized cap

The 512-token tool_args bound cut every long write_file call at exactly 512
(done_reason=length, no call parsed); the payload tools now run under 4096.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

### Task 2: The retry ladder recognises truncation and stops

**Files:**
- Modify: `core/llms.py` (next to `extract_tok_per_sec`, ~line 560)
- Modify: `nodes/execute.py` (`_generate_tool_call`, the `if not calls:` / `else:` arms)
- Test: `tests/test_engine.py` (after `test_execute_arg_failure_lands_as_error`, ~line 277)

**Interfaces:**
- Consumes: `nodes.execute._task_for_tool` (Task 1), `core.serving.num_predict`.
- Produces: `core.llms.was_truncated(response) -> bool` — True iff `response.response_metadata.get("done_reason") == "length"`.
- Produces: `nodes.execute.TRUNCATED_TEXT` — the `error:` result prefix `"error: the tool call was cut off at the output limit"`; `nodes/synthesize` needs no change (any `error:` result is already disclosed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py` (helpers `_step`, `_state`, `ex`, `st` exist at the top of the file; add `from core import llms` and `from core import serving` next to the existing `from core import ...` import if absent):

```python
def test_was_truncated_reads_done_reason():
    from langchain.messages import AIMessage
    assert llms.was_truncated(AIMessage(content="", response_metadata={"done_reason": "length"}))
    assert not llms.was_truncated(AIMessage(content="", response_metadata={"done_reason": "stop"}))
    assert not llms.was_truncated(AIMessage(content=""))
    assert not llms.was_truncated(None)


def test_generate_tool_call_stops_after_a_truncated_attempt(monkeypatch):
    """A tool call cut off at num_predict (Ollama done_reason=length) cannot be fixed by a
    hotter temperature — the 2026-09-02 traces show three identical 512-token truncations per
    step, 30-45 s each. One attempt, then an honest error naming the limit."""
    from langchain.messages import AIMessage

    calls = []

    class M:
        def bind_tools(self, tools):
            return self

        def invoke(self, msgs, **kw):
            calls.append(kw)
            return AIMessage(content="", response_metadata={"done_reason": "length"})

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["write_file"], "ctx")
    assert args is None
    assert len(calls) == 1, "a truncated draw must not be re-rolled at a hotter temperature"
    assert failure.startswith(ex.TRUNCATED_TEXT)
    assert str(serving.num_predict("tool_payload")) in failure
    assert "write_file" in failure


def test_generate_tool_call_truncation_is_not_the_text_fallback(monkeypatch):
    """A truncated draw that also carried some prose must NOT be recorded as 'the model answered
    in text instead' — the prose is a cut-off generation, not an answer."""
    from langchain.messages import AIMessage

    class M:
        def bind_tools(self, tools):
            return self

        def invoke(self, msgs, **kw):
            return AIMessage(content="Once upon a", response_metadata={"done_reason": "length"})

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    _, failure, _ = ex._generate_tool_call(tools_by_name["write_file"], "ctx")
    assert failure.startswith(ex.TRUNCATED_TEXT)
    assert "answered in text" not in failure
```

`tools_by_name` — add `from tools.registry import tools_by_name` at the top of `tests/test_engine.py` if it is not already imported there (check with `grep -n tools_by_name tests/test_engine.py`).

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_engine.py -q -k "truncat"`
Expected: 3 failures — `AttributeError: module 'core.llms' has no attribute 'was_truncated'`, then `AttributeError: ... 'TRUNCATED_TEXT'` / `len(calls) == 3`.

- [ ] **Step 3: Add the helper to `core/llms.py`**

Directly after `extract_prompt_tokens`:

```python
def was_truncated(response) -> bool:
    """Whether the daemon stopped this generation at `num_predict` rather than at a natural end
    (Ollama's `done_reason == "length"`). A truncated TOOL CALL never parses — the JSON is cut
    mid-argument — and re-rolling it at a hotter temperature reproduces the cut, so callers on a
    retry ladder read this to stop instead of spending the remaining rungs."""
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("done_reason") == "length"
```

Add `was_truncated` to the `from core.llms import ...` line in `nodes/execute.py`.

- [ ] **Step 4: Teach `_generate_tool_call` the branch**

Add the constant near `WRITE_GATE_SKIP_PREFIX`:

```python
# The truncation refusal: the daemon hit the task's num_predict before the call closed. One
# producer (here); the synthesizer discloses any `error:` result, so no parser needs the text.
TRUNCATED_TEXT = "error: the tool call was cut off at the output limit"
```

Inside the loop, replace the `if not calls: parsed = ...` block and the `else:` text-fallback arm so the truncation check comes FIRST:

```python
        content = getattr(resp, "content", "")
        content = content if isinstance(content, str) else str(content)
        calls = [{"args": tc.get("args")} for tc in (getattr(resp, "tool_calls", None) or [])]
        if not calls and was_truncated(resp):
            # The call was cut mid-JSON at num_predict. No temperature fixes that — the
            # remaining rungs would reproduce the cut (measured: three identical 512-token
            # truncations per step). Refuse now with the limit named, so the answer can say
            # WHY and a redraft can split the write instead of retrying it whole.
            cap = serving.num_predict(task)
            return None, (
                f"{TRUNCATED_TEXT} ({cap} tokens) — the {tool.name} call's arguments are too "
                "long to generate in one call; write a shorter version, or write the first part "
                "with write_file and append the rest with edit_file"
            ), resp
        if not calls:
            parsed = parse_text_call(content)
            if parsed:
                calls = [{"args": parsed}]
```

Add `from core import serving` to the imports of `nodes/execute.py`. Leave the `else:` text-fallback arm as it is — it is now reached only for non-truncated prose, which is what its comment describes.

- [ ] **Step 5: Run the tests and the whole suite**

Run: `python -m pytest tests/test_engine.py -q -k "truncat"` → 3 passed.
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add core/llms.py nodes/execute.py tests/test_engine.py
git commit -m "execute: a truncated tool call ends the retry ladder with the limit named

done_reason=length cannot be fixed by temperature; one attempt, then an
honest error instead of three identical 30-45 s truncations.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

### Task 3: A failed write does not arm the fabrication gate

**Files:**
- Modify: `nodes/execute.py` (`_write_gate`, the `failed = ...` line, ~line 249)
- Test: `tests/test_engine.py` (next to `test_write_gate_error_status_still_arms`, ~line 567)

**Interfaces:**
- Consumes: `core.plan_context.WRITE_TOOLS` (already imported in `nodes/execute.py`).
- Produces: nothing new; `_write_gate(state, step) -> str | None` keeps its signature.

- [ ] **Step 1: Write the failing test**

Insert after `test_write_gate_error_status_still_arms`:

```python
def test_write_gate_failed_write_does_not_arm(monkeypatch):
    """The gate guards a value bridged over a failed PRODUCER (a search/read that returned an
    error). A prior WRITE that errored (a truncated call, a daemon timeout) produced no value
    anything could bridge from — arming on it turned every transient write failure into a
    permanent 'fabrication' skip on the redraft (run 15, 2026-09-02: 'I must not fabricate a
    story'). With no producer failed and nothing searched, the plan is mechanical: no judge."""
    def boom(*a, **k):
        raise AssertionError("a plan whose only failure is a write must not consult the judge")

    monkeypatch.setattr(ex, "structured", boom)
    plan = [_step(1, "write_file", result="error: the tool call was cut off at the output limit",
                  status="error"),
            _step(2, "write_file")]
    assert ex._write_gate(_state(plan), plan[1]) is None


def test_write_gate_failed_producer_still_arms_even_after_a_failed_write(monkeypatch):
    monkeypatch.setattr(
        ex, "structured", lambda *a, **k: st.WriteGate(present=False, evidence="not there")
    )
    plan = [_step(1, "read_file", result="error: read failed", status="error"),
            _step(2, "write_file", result="error: no tool call emitted", status="error"),
            _step(3, "write_file")]
    blocked = ex._write_gate(_state(plan), plan[2])
    assert blocked and "not present" in blocked
```

- [ ] **Step 2: Run them to verify the first fails**

Run: `python -m pytest tests/test_engine.py -q -k "failed_write_does_not_arm or failed_producer_still_arms"`
Expected: `test_write_gate_failed_write_does_not_arm` FAILS with the `AssertionError` from `boom`; the second passes already.

- [ ] **Step 3: Narrow the arming condition**

In `_write_gate`, replace the `failed = any(s.get("status") == "error" for s in done)` line and its comment tail:

```python
    # Failure is the STRUCTURAL stamp only (status == "error", gotcha #6) — never sniffed from
    # observation text: a successful read of a log that begins "ERROR:" is a done step with an
    # error-looking result, and text-sniffing it armed the gate on purely mechanical plans (the
    # exact false positive the saturn_status contract removed from update_plan, 2026-07-04).
    # And only a failed PRODUCER arms it: the hazard is a value bridged over a search/read that
    # never returned one. A prior WRITE that errored (a truncated call, a daemon timeout) left
    # no value to bridge from, and arming on it turned a transient write failure into a
    # permanent 'fabrication' skip on the redraft (2026-09-02).
    failed = any(
        s.get("status") == "error" and s.get("intended_tool") not in WRITE_TOOLS for s in done
    )
```

- [ ] **Step 4: Run the tests and the whole suite**

Run: `python -m pytest tests/test_engine.py -q -k write_gate` → all pass (the existing `test_write_gate_error_status_still_arms` uses a failed `read_file`, still armed).
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/execute.py tests/test_engine.py
git commit -m "execute: the write gate arms on a failed producer, not on a failed write

A prior write that errored left no value to bridge; arming on it skipped
the redraft as fabrication and told the user the story could not be written.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

### Task 4: The judge's cap and a truncation diag line in `structured()`

**Files:**
- Modify: `core/serving.py` (`TASKS["judge"]`)
- Modify: `core/structured.py:333-343` (`structured()`, the unparseable branch)
- Test: `tests/test_serving.py`

**Interfaces:**
- Consumes: `core.llms.was_truncated` (Task 2).
- Produces: `serving.num_predict("judge") == 1024`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serving.py`:

```python
def test_judge_cap_clears_a_long_rationale():
    """Run 16 (2026-09-02): the rectify judge's first attempt hit exactly 512 tokens writing its
    `reasoning` field and the JSON never closed; the retry happened to finish in 241. The cap is
    a breaker, not a budget — it must sit above a verbose-but-healthy verdict."""
    assert serving.num_predict("judge") >= 1024
    assert serving.num_predict("judge") <= serving.num_predict("plan")


def test_structured_logs_a_truncated_draw(monkeypatch):
    from langchain.messages import AIMessage
    import diag

    lines = []
    monkeypatch.setattr(diag, "log", lambda s: lines.append(s))
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)

    class M:
        def invoke(self, msgs, **kw):
            return AIMessage(content='{"reasoning":"cut', response_metadata={"done_reason": "length"})

    monkeypatch.setattr("core.llms.get_model", lambda role: M())
    out = structured.structured("judge", [], structured.RectifyBool, structured.RECTIFY_FORMAT,
                                structured.RECTIFY_SHAPE,
                                default=structured.RectifyBool(rectify=False, reasoning="d"))
    assert out.reasoning == "d"
    assert any("cut off at num_predict" in l for l in lines)
```

(`structured.structured` re-imports `get_model` from `core.llms` inside the function, which is why the monkeypatch targets `core.llms.get_model`. Check the existing `RectifyBool` field names with `grep -n "class RectifyBool" -A 4 core/structured.py` and adjust the default's kwargs if they differ.)

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_serving.py -q -k "judge_cap or logs_a_truncated"`
Expected: 2 failures (`512 >= 1024` is false; no diag line contains `cut off at num_predict`).

- [ ] **Step 3: Raise the judge cap**

In `core/serving.py` `TASKS`:

```python
    "judge": Task("judge", strict=True, num_predict=1024, think=False),
```

- [ ] **Step 4: Log truncation in `structured()`**

In `core/structured.py`, inside the `except (ValidationError, ValueError):` branch of `structured()`, before the existing `diag.log(...)`:

```python
            from core.llms import was_truncated

            if was_truncated(resp):
                diag.log(
                    f"structured[{role}/{schema.__name__}] attempt {i + 1} was cut off at "
                    f"num_predict — the {role} task's bound is below this verdict's length"
                )
```

- [ ] **Step 5: Run the tests and the whole suite**

Run: `python -m pytest tests/test_serving.py -q` → all pass (the existing `serving.num_predict("judge") <= serving.num_predict("plan")` assert at ~line 43 still holds: 1024 ≤ 1536).
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add core/serving.py core/structured.py tests/test_serving.py
git commit -m "serving: judge cap 1024, and structured() logs a draw cut at num_predict

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

### Task 5: Declare `numpy` — the knowledge-base search's missing dependency

**Files:**
- Modify: `pyproject.toml:32-58` (`dependencies`)
- Modify: `requirements.txt:7-10`
- Test: `tests/test_rag_sync.py` (append)

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_sync.py`:

```python
def test_in_memory_vector_store_can_run_a_similarity_search():
    """stores/rag.py's InMemoryVectorStore computes cosine similarity through numpy, which
    langchain-core stopped pulling in transitively. Without it every search_knowledge_base
    call failed at query time with 'cosine_similarity requires numpy' (run 19, 2026-09-02) —
    a path no other test reaches. Offline: a deterministic fake embedder, no daemon."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    store.add_texts(["welcome to saturn", "unrelated"])
    hits = store.similarity_search("welcome", k=1)
    assert len(hits) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_rag_sync.py -q -k similarity_search`
Expected: FAIL with `ImportError: cosine_similarity requires numpy to be installed` (or the langchain-core wording of the same).

- [ ] **Step 3: Declare and install the dependency**

`pyproject.toml`, inside `dependencies = [` after the `langchain-text-splitters` line (keep the section comments as they are):

```toml
    "numpy>=1.26",                 # InMemoryVectorStore similarity search (stores/rag.py); langchain-core no longer pulls it in
```

`requirements.txt`, after line 10 (`langchain-text-splitters...`):

```
numpy>=1.26                       # InMemoryVectorStore similarity search (stores/rag.py) - not pulled in transitively
```

Then: `pip install -e .[dev]` (from the activated `.venv`; confirm with `pip show numpy`).

- [ ] **Step 4: Run the test and the whole suite**

Run: `python -m pytest tests/test_rag_sync.py -q -k similarity_search` → 1 passed.
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt tests/test_rag_sync.py
git commit -m "deps: numpy — search_knowledge_base failed at query time without it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

### Task 6: Changelog, architecture note, and live verification

**Files:**
- Modify: `CHANGELOG.md` (under `## [Unreleased]`)
- Modify: `docs/ARCHITECTURE.md:98` (the `tool_args.py` row's neighbourhood — add one line about `serving.py`'s per-tool bound if the file lists `serving.py`; check with `grep -n serving docs/ARCHITECTURE.md`)

- [ ] **Step 1: Changelog**

Under `## [Unreleased]`, add a `### Fixed` section after the existing `### Added` block (before `### Removed`):

```markdown
### Fixed

- **Long writes actually happen.** `write_file` / `edit_file` calls now generate under a
  file-sized output bound; before, a story or a list longer than ~500 tokens was cut off inside
  the tool call, the step reported "no tool call emitted", and the engine retried the identical
  truncation three times. A call that still exceeds the bound is refused once with the limit
  named instead of looping.
- **A failed write is not "fabrication".** The semantic write gate arms only on a failed
  search/read upstream, not on a prior write attempt that errored — the redraft after a
  transient write failure is no longer skipped with a fabricated-value disclosure.
- **`search_knowledge_base` works from a clean install.** `numpy` is declared; without it every
  knowledge-base search failed at query time.
- The rectify judge's output bound is 1024 (a verbose verdict was being cut mid-JSON).
```

- [ ] **Step 2: Live verification against the daemon (Ollama running, tier `9b` pulled)**

Run each from the repo `.venv`; the workspace is `database/workspace`.

```bash
saturn -p "write me a short story and save it to a file named story.txt" --yolo
ls -la database/workspace/story.txt && wc -c database/workspace/story.txt
saturn -p "give me five ideas for a social function and write them to a text file called juan.txt" --yolo
wc -c database/workspace/juan.txt
tail -30 logging/diag.log | grep execute_node
```

Expected: both files exist and hold the content; every `execute_node` line for the write step ends `-> write_file` (no `no call:` line); the write step took one model call (check `sqlite3 database/db.sqlite "select node, output_tokens from llm_calls where run_id=(select max(run_id) from runs)"` — the execute row for the write is under 4096 and above 512 for the story).

Then the trust benchmark's write probes must still pass: `python benchmark.py` and confirm the `gate_probe` tasks report ok.

Finally clean the verification files out of the workspace: `rm database/workspace/story.txt database/workspace/juan.txt`.

- [ ] **Step 3: Full suite, then commit**

Run: `python -m pytest tests/ -q` → all pass.

```bash
git add CHANGELOG.md docs/ARCHITECTURE.md
git commit -m "docs: changelog + architecture note for the tool-call output cap fixes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NmoVQv5uy2kRm7stZHRmqe"
```

---

## Self-review

- Spec coverage: cap (Task 1), retry blindness (Task 2), gate arming (Task 3), judge cap + diagnosability (Task 4), numpy (Task 5), user-visible record + live proof (Task 6). The `_RESULT_CAP` observation is explicitly out of scope.
- Type consistency: `_task_for_tool(name) -> str` (Task 1) is what Task 2's `task = _task_for_tool(tool.name)` reads; `was_truncated(response) -> bool` (Task 2) is what Task 4 imports; `TRUNCATED_TEXT` is defined in Task 2 and asserted by Task 2's and Task 3's tests.
- Placeholder scan: every code step carries its code; the only "check and adjust" notes name the exact grep to run.

---

### Task 7 (added during execution): the no-call guard in rectify

**Why:** the user's execution instruction was "make sure the agent doesn't silently loop or
retry excessively when it should not". After Tasks 1–3 a write that still exceeds the 4096
bound fails once per step (Task 2), but `rectify`'s judge then sends the identical step to
`replan`, which redrafts it, which fails identically — up to `MAX_REPLANS = 5` cycles. Runs 10
and 15 show exactly this shape (three cycles, ~2 minutes, Ctrl-C).

**Files:**
- Modify: `nodes/rectify.py` — new constant `NO_CALL_LIMIT = 2`; new branch **1b** between the
  guarded-outcome branch (1) and the replan-budget check; module docstring lists 1b.
- Test: `tests/test_engine.py` — four tests under "the no-call guard (2026-09-02)".

**Rule (structural, no text sniffing):** the step just recorded has `status == "error"`, its
`intended_tool` appears in no `tool_events` entry this turn (it never actually executed), and
the plan now holds `NO_CALL_LIMIT` error-status steps for that tool → cancel the remaining
steps with `cancelled: the engine could not generate a valid <tool> call N times this turn, so
the run ended` and return `rectify=False` (routes to synthesize, which discloses every
incident). The first failure keeps its one redraft (a truncated write may be split); the
second ends the run. Tools that executed and then errored (a `tool_events` entry exists) are
tool failures, judged as before. Known limit: a tool that executed once earlier in the turn and
then fails to generate is not caught by 1b; the replan budget still bounds it.

- [x] Tests: `test_rectify_second_no_call_failure_for_a_tool_ends_the_run`,
  `test_rectify_first_no_call_failure_still_gets_its_redraft`,
  `test_rectify_no_call_guard_ignores_tools_that_actually_executed`,
  `test_rectify_no_call_guard_is_per_tool`.
- [x] Commit: `rectify: no-call guard — a tool that fails to generate a call twice ends the run`

---

## Follow-ups shipped on the same branch (the `/goal` robustness pass)

Each was found in the live traces after the plan above landed, fixed with a general rule, and
pinned by offline tests (`git log main..tool-call-output-cap` has one commit per item):

- **rectify 4 — one dead-end retry per turn**, read structurally off the plan (two dead-end
  results already recorded) rather than the replan counter.
- **synthesize — write-only turns do not arm the grounding gate** (`gate_applies` ignores
  `WRITE_TOOLS` observations; a read/search alongside still arms it).
- **startup — model warm-up** on a daemon thread after the health check
  (`app/startup.warm_model`, `start_warm_up`; `tests/test_warmup.py`).
- **files — `read_file` not-found refusal** names the workspace and, for an ingested document,
  `search_knowledge_base` (`tests/test_read_file_missing.py`).
- **rectify 1b — the no-call guard also counts empty reasoning steps** (key `None`).
- **planner rule** — a named file is a workspace file only when the context lists it there.
- **ground — workspace manifest reconciled with disk every turn**
  (`document_registry.sync_workspace_manifest`; `tests/test_workspace_sync.py`).
- **plan / replan — the namespace guard** (`plan_ops.retarget_knowledge_base_reads`): a
  read_file step naming an ingested document with no workspace file of that name becomes a
  search_knowledge_base step (`tests/test_plan_retarget.py`).
- **files — the navigation tools skip hidden entries** (`.manifest.md`, `.git`, `.DS_Store`),
  so an empty workspace reads as empty (`tests/test_hidden_entries.py`).
- **tests — a raising tool still records a `tool_event`** (the no-call guard's premise).

Observed but left alone (judgment calls, not engine defects): the 9b planner answers "How
many moons does Saturn have?" from priors and the judge does not send it to web_search
(identical on main); `core/plan_context._RESULT_CAP = 800` (noted above).

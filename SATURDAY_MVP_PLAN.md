# Saturday.ai — MVP Plan & Architecture

> Context document for future sessions. Captures the agreed MVP scope, target
> architecture, and phased roadmap. Last updated: 2026-06-03.

## Vision

Saturday.ai is a **local-first, transparent, flexible, general-purpose** agentic AI
platform — a personal copilot that can safely work with the user's files, documents,
tools, and (eventually) sensitive private data (email, calendar, notes, personal
knowledge). It is explicitly **not** a narrow single-purpose tool. Priorities, in order:
**transparency, configurability, safety, extensibility.**

---

## Current state (honest assessment)

The repo is a solid *skeleton* of a single-shot RAG+tool responder, but not yet an agent.

**Works today:** flat LangGraph pipeline (`context_builder → plan → {rag, tool} →
synthesize → END`), six tools, Tavily `deep_research`, markdown document-manifest system,
a latency/routing benchmark harness.

**Broken / stubbed / aspirational — drives the roadmap:**

1. **RAG is wired but inert.** `rag.build_ingest()` is never called in `agent.py` (only
   `build_retrieval()` is), so the `InMemoryVectorStore` is empty at query time and the
   `rag` branch returns nothing. It also re-embeds from scratch every run (no persistence).
2. **The "planner" is a one-shot router.** `plan_node` sets three booleans once, fans out,
   and goes straight to `synthesize`. There is **no agentic loop** — the model never sees a
   tool result and decides a next action. The `multi_tool` benchmark suite is therefore
   architecturally impossible today.
3. **No safety gate.** `approval_node` in `tool.py` is a commented-out stub; `write_file`
   and `deep_research` fire with no confirmation.
4. **No persistence.** No checkpointer, no session memory, no cross-run memory. `db.sqlite`
   exists but is unused. The `user_profile.md` / `agent_profile.md` templates in
   `MEMORIES.md` are not loaded anywhere.
5. **Benchmarks measure speed, not correctness.** Latency, routing flags, tools-called —
   never whether the answer was right or a file got the right contents.
6. **`CLAUDE.md` is out of date.** It describes a `backend/`/`frontend/` split and an
   Electron app; neither exists — the repo is flat at root with no frontend.
   `verifier`/`repair` are disabled. Model id `gemma4:e4b` is non-standard — verify it.

---

## MVP definition

**The minimum lovable Saturday.ai:** a local, **CLI-first** personal copilot that

- reliably answers questions and **chains multiple tools in a single turn**,
- **reads and writes local files safely**, with an **approval gate** on side effects,
- retrieves from a **persistent local knowledge base**,
- **remembers the user across sessions** (profile + working memory),
- is **fully observable** — every step, decision, and tool call is inspectable, and
- is **configurable** via a single config file (model, enabled tools, approval policy, paths).

**Deferred past MVP** (keeps it general but shippable): Electron frontend, email/calendar/
Drive, vision/multimodal, browser automation, proactive file-watching. The MVP should make
adding these *easy*, not include them.

---

## Target architecture

The core move: **from a static fan-out graph to a living-plan agentic loop.**

```
                ┌─────────────────── outer loop (verify/repair) ───────────────────┐
                │                                                                    │
START → context_builder → plan(draft) → agent(model) ──tool_calls?──▶ approval_gate ─▶ tools ─┐
            │                  │              │                            │                    │
       (load memory,      (initial      no tool_calls          (interrupt for side-      update_plan
        profiles,          plan in          │                   effecting tools;        (mark/revise) ┘
        manifests)         state)           ▼                   auto-pass read-only)
                                        verifier ──ok──▶ END
                                            └─fail─▶ repair ──▶ back to agent
```

### 1. Living-plan ReAct hybrid (the chosen middle ground)

Neither pure ReAct (flexible but opaque) nor plan-execute (inspectable but rigid). Instead,
the **plan is a first-class, mutable object in state**, drafted up front and revised *during*
the loop:

- New `AgentState` field: `plan: list[dict]`, each step
  `{step_id, label (human-readable), status: pending|active|done|skipped, intended_tool?}`
  (Pydantic `Plan`/`PlanStep` used only for the planner's structured output).
- `plan_node` (repurposed from the router) **drafts** the initial plan → immediate inspectability.
- The `agent` node runs a ReAct loop: reads plan + history + observations, picks the next
  action (tool call or finalize).
- After each action, `update_plan` (repurposed `reflect.py`) marks steps done / revises /
  inserts. **The plan is advisory, not a rigid DAG** — the model may deviate when reality
  disagrees, but each step it must reconcile (state which step it's on + update statuses).
- RAG retrieval becomes **just another tool** (`search_knowledge_base`), not a hard branch.
- `verifier` / `repair` become the **outer correctness loop**.

Guardrails: max iterations, step budget, no-progress detection (plan unchanged + same tool
repeated → halt).

**Open sub-decision (default chosen):** combine "pick next action" + "update plan" into one
structured-output call on capable (workstation) models; split into two steps on weak (laptop)
models. Architecture is identical; only the binding changes by model tier.

### 2. Persistence

Add LangGraph `SqliteSaver` over the existing `db.sqlite`. Gives resumable sessions and is
the mechanism that makes the approval-gate `interrupt` work (interrupt → human responds →
resume from checkpoint).

### 3. Three-layer memory

- **Session** — checkpointed message thread (per conversation).
- **Persistent** — `user_profile.md` + `agent_profile.md` loaded each turn by the grounding
  node (§8); agent can append learned facts (gated like any write).
- **Knowledge base** — the persistent RAG store.

### 4. Safety / approval gate

Classify every tool: `read_only` | `side_effecting` | `destructive`. Read-only runs freely;
side-effecting/destructive triggers a LangGraph `interrupt` showing the exact action + args;
execution resumes only on approval. Policy configurable (e.g. `auto_approve: read_only`).

### 5. Transparency / observability

A structured trace: every node entry/exit, routing decision, tool call + args + result,
latency → written to SQLite. This is the debugging tool now and the frontend's data source
later. Formalize the scattered `time.perf_counter()` calls into one trace writer.

### 6. Plan transparency — streamed event contract

The agent **emits the plan as a structured event stream**; surfaces **subscribe**. The loop
never knows whether a terminal panel or an Electron window is listening.

- Stable contract: `PlanEvent = {step_id, label, status, tool?}`, streamed via LangGraph
  streaming as the plan mutates (after draft, after each `update_plan`).
- Same stream is persisted to the trace store — "live plan panel" and "transparency log"
  are the same data, surfaced two ways.
- **CLI rendering (MVP):** a re-printed **Rich grid table** in a "blueprint terminal" style —
  thin cyan single-line grid, uppercase labels (`# | STATUS | STEP | TOOL`), text status
  labels (no emoji/glyphs), calm cyan accent. See `ui.py`. *(Textual split-pane / Electron
  window are the graduation path and require **no agent-code change** — only a different
  subscriber.)*

### 7. Configurability — role-based modular models

`llms.py` becomes a `get_model(role)` **factory**, not hard-coded globals. The agent
references **roles, never concrete models**, so swapping models per hardware never touches
graph logic.

- Roles: `planner`/`reasoner`, `tool_caller`, `synthesizer`, `utility`, `embedder`, `judge`.
- One `config.yaml` with **hardware-tier presets** the user picks one of:
  - `laptop` — all roles → one 7B-ish model; `utility` → 1–3B
  - `workstation` — `planner`/`synthesizer` → 14–32B; `utility` → small
  - `cloud-hybrid` — `planner` → Claude; rest local
- Use LangChain `init_chat_model` for the provider abstraction (Ollama/OpenAI/Anthropic all
  expose `BaseChatModel`); provider is just config.
- Each model carries a **capability descriptor**: `supports_tools`,
  `supports_structured_output`, `context_window`, `supports_vision`.
- **MVP requires native tool-calling + structured output** (documented constraint). The
  prompted-ReAct-protocol / JSON-repair fallback for weaker models is **deferred to post-MVP**.

Config also externalizes the hard-coded paths/model ids currently scattered across `llms.py`,
`rag.py`, `document_registry.py`.

### 8. Grounding node (re-scoped `context_builder`)

The existing `context_builder` assembles one big context string from tool inventory + chat
history + a (stubbed) doc inventory. Under the ReAct loop that is mostly **redundant noise**,
not latency (it makes no LLM call):

- **Tool inventory — drop it.** Tools are bound natively via `bind_tools`; the model already
  gets the schemas. A second hand-formatted text list makes the model see tools twice and
  degrades tool-calling on small local models.
- **Chat history — drop it.** `messages` (with `add_messages`) is already passed to the model;
  re-serializing the transcript into a string double-feeds the conversation and wastes context.
- **Document / workspace manifests + persistent memory — keep (this is the real value).** The
  planner needs to know *what docs/files/profile facts exist* to decide whether to retrieve.

So **re-scope it to a lean grounding/IO node** (suggested rename: `ground` / `load_context`)
whose only job is to load the things **not already in state** — user/agent profiles + document
& workspace manifests — and inject them as a *single* system/context message.

- Build it **once per turn** (manifests/profiles are static within a turn). Anything **dynamic**
  — tool results — flows through `messages`, never a frozen context blob, to avoid staleness
  in the loop.
- It remains the **sole writer** of that grounding message; downstream nodes read, never mutate.

This is also the natural home for loading the persistent layer of the three-layer memory.

---

## Essential tool suite (MVP)

The goal is **not many tools, but a small set of orthogonal, composable primitives** — power
comes from the loop chaining them, not from tool count. This also respects the hard local-
model constraint: small models choke on large, fuzzy tool schemas. The suite covers five
capability axes:

| Axis | Tool | Status | Risk tier |
|---|---|---|---|
| **Compute & code** | `run_python` — sandboxed code execution; the **force multiplier** that subsumes the long tail (math, data wrangling, parsing, conversion, automation) | **new** | destructive (gated + sandboxed) |
| | `calculate` — fast, deterministic arithmetic fast-path | keep | read-only |
| **Local files** | `read_file`, `list_directory` | keep (fix path bug, sandbox to workspace) | read-only |
| | `write_file` | keep; **gate behind approval** | side-effecting |
| | `search_files` — content grep + filename glob across the workspace | **new** | read-only |
| **Web** | `web_search` — discovery | keep | read-only |
| | `web_fetch` — fetch + extract a specific URL (promote the existing unregistered `web_extract.py`) | **new (exists, unwired)** | read-only |
| | `deep_research` — heavyweight multi-source research | keep, **optional** (search+fetch+loop covers most cases) | side-effecting (gated) |
| **Knowledge base** | `search_knowledge_base` — RAG retrieval as a callable tool | **new** | read-only |
| **Memory** | `remember` / `recall` — persist + retrieve durable facts across sessions | **new** | `remember` side-effecting (low risk) / `recall` read-only |

**Design principles for the suite:**
- **Orthogonal & composable** — each tool is one primitive on one axis; versatility comes
  from the loop combining them.
- **Code execution is the great multiplier** — `run_python` absorbs the infinite long tail,
  which is what keeps the suite small yet powerful. Highest-risk tool; needs sandbox +
  approval gate before it ships (see `database/documents/tool_sandbox_policy.txt`).
- **Short, distinct descriptions; ≤ ~10 tools exposed at once** — loop reliability on local
  models depends on this more than on prompt cleverness.
- **Capability-tiered** — the exposed tool set is filtered by model tier (lean on laptop
  models, full on workstation), reusing the `get_model(role)` modularity.
- **Every tool carries a risk tier** wired to the approval gate.

**Deferred past MVP:** `shell` exec (broader/more dangerous than `run_python`), browser
automation, clipboard/screenshot, email/calendar, vision.

---

## Canonical workflows (used to validate the MVP)

1. **Pure reasoning** — no tools (skip the loop).
2. **Single tool** — "17.5% of 3842" → calculate → answer.
3. **Multi-tool chain** — "search X and save a summary to notes.md" → web_search →
   write_file (with approval). *The headline capability that doesn't work today.*
4. **RAG Q&A** — "what does the handbook say about Y" → search_knowledge_base → answer w/ citation.
5. **Memory** — "remember I prefer terse answers" → profile updated; honored next session.
6. **Safe destructive** — "delete/overwrite X" → approval gate fires → user declines → no-op.

---

## Benchmarks (upgrade from latency-only to correctness)

The harness structure is good; it's missing the grading layer. Add:

- **Routing accuracy** — label each query with the tools it *should* use; score precision/recall.
- **Task success (checkable)** — filesystem/calculator/multi-tool: assert real outcomes
  (file exists with expected content; numeric answer within tolerance). No LLM needed.
- **Answer quality (LLM-as-judge)** — open-ended `llm`/`rag`/`deep_research`: grade 1–5 vs rubric.
- **Safety suite (new)** — destructive prompts that *must* trigger the gate; score gate-fire rate.
- **Overhead baseline** — direct single LLM call vs full pipeline, per query.

Keep latency as a secondary axis; add a run-to-run diff view to catch regressions.

---

## Implementation roadmap

**Phase 0 — Stabilize (fix what's silently broken). ◐ MOSTLY DONE.** `build_ingest` wired into
startup; model id confirmed (`gemma4:e4b` is a real local tag); the `__file__` path bug fixed
(workspace now `database/workspace/`); `SqliteSaver` checkpointer added. *Still open:* the
vector store re-embeds every run (no on-disk persistence yet). *Exit: RAG returns documents;
sessions resume — met.*

**Phase 1 — Living-plan agentic loop. ✅ DONE.** `plan`/`PlanStep` added to state; `plan_node`
drafts the plan; `reflect.py` is the `update_plan` step; static fan-out replaced with the
model↔tools ReAct loop (`tool.py:agent_node`/`tool_node`/`route_after_agent`); RAG exposed as
`search_knowledge_base`; `context_builder` re-scoped to the lean grounding node (§8). *Exit:
multi-tool chaining works (workflow #3) — met.*
  - *Deviations:* `update_plan` is **mechanical** (no LLM) because the local model can't emit
    JSON for it reliably; synthesize pairs each result with its call (`name(args) -> result`)
    so the model stops recomputing and contradicting tool output.

**Phase 2 — Safety & transparency. ✅ DONE.** Tool risk tiers (`registry.TOOL_RISK`/`risk_of`)
+ `interrupt` approval gate (`tool.py:approval_node`); file-tool sandbox hardened
(`is_relative_to`); structured trace → SQLite (`trace.py`); live plan panel via Rich
(`ui.py`). *Exit: workflow #6 passes; every run is inspectable; plan visible live — met.*
  - *Deviations:* the plan panel is a **re-printed Rich panel** (not `Live`) to avoid
    conflicting with the approval `input()` prompt — same event-stream design, swappable
    renderer. Checkpoints + trace both live in `database/db.sqlite`.
  - *Resolved follow-up:* the plan is now stored in state as plain dicts (Pydantic `Plan`/
    `PlanStep` only at the planner boundary via `state.steps_to_dicts`), so the `SqliteSaver`
    no longer warns about serializing a custom type.

**Phase 3 — Memory & config. ✅ DONE.** Persistent working-memory store (`memory_registry.py`)
+ `remember`/`recall` tools, loaded into context each turn by the grounding node; `config.yaml`
+ `config.py` accessor + `get_model(role)` factory (`llms.py`) with hardware-tier presets and
capability descriptors. Model ids and filesystem paths are no longer hard-coded — they live in
`config.yaml`; the agent references roles, never models. *Exit: workflow #5 passes; no
hard-coded models/paths; models swappable by tier — met.*
  - *Deviations:* durable memory is a dedicated append-only markdown store
    (`database/memory/memory.md`) rather than edits to `user_profile.md` — append-only is
    safer and the grounding node loads both. `remember` is gated like any side-effecting write
    (configurable via `runtime.auto_approve`). `get_model(role)` caches built models and
    exposes `reset_models()` so `/model` and `/config` can hot-swap tier/role bindings live
    (nodes call the factory at run time rather than importing fixed handles). Profile
    *auto-update* by the agent is left to a later pass; the `remember` tool covers the
    learned-fact path. `/model`, `/config` slash commands are now implemented.

**Phase 4 — Capability expansion. ✅ DONE.** Added the gated, sandboxed `run_python` tool
(`tool_registry/run_python.py`) — the "force multiplier" that subsumes computation, data
wrangling, parsing, and file generation. It runs the snippet in a separate `-I` interpreter
(stdin-fed, so no command-line escaping limits) with cwd pinned to the workspace, a clamped
wall-clock timeout (default 30s, max 120s), and size-capped stdout/stderr. Registered with risk
tier `destructive`, so the approval gate fires on every call. stdout/stderr/traceback are returned
verbatim, so a failing run flows back as a ToolMessage and the ReAct loop fixes and retries.
*Exit: agent can run and self-correct code — met.*
  - *Deviations:* `shell` is **deferred past MVP** (the tool suite marks it broader/more dangerous
    than `run_python`); Phase 4 ships code execution only. The sandbox is process + timeout + cwd
    isolation, **not** a security boundary — the snippet can still reach the network and the wider
    filesystem by absolute path. The approval gate (destructive tier) is the real control; true
    OS-level isolation (containers / seccomp) is post-MVP.

**Phase 5 — Eval upgrade. ✅ DONE.** The latency-only harness is now a correctness/safety
grader. Suites are structured `EvalCase`s (`eval_cases.py`) that carry their expected outcome;
`grading.py` scores each case across five orthogonal graders — **routing** (precision/recall/F1/
exact-set vs `expected_tools`), **task success** (deterministic substring / numeric-within-
tolerance / workspace-file checks), **answer quality** (LLM-as-judge 1–5 via the `judge` role),
**safety** (the approval gate MUST fire on destructive prompts; the harness declines and asserts
the side effect didn't happen), and an **overhead baseline** (direct model call vs full
pipeline). `benchmark.py` runs the suites, grades them, writes a scored JSON report, prints a
per-suite + overall summary, and **auto-diffs against the previous run** (also `--diff` standalone)
so one command tells you better/worse with a verdict. *Exit: one command reports if the agent got
better or worse — met.*
  - *Deviations:* the LLM-as-judge uses a strict `SCORE:`/`REASON:` text format parsed by regex
    (not structured output) because the local `judge` model is unreliable at JSON; it degrades to
    a null score (never raises) if the call fails, and is skippable with `--no-judge`. The judge
    prompt lives in `grading.py`, not `messages.py` — it's eval-time tooling, not a graph node.
    `deep_research` cases stay opt-in (`--run-deep-research`); the run-to-run diff compares the
    overall headline metrics only.

**Post-MVP:** frontend wiring → email/calendar → browser automation → vision → proactive watching.

Critical path is **Phases 0–2** (loop + safety + persistence are interdependent). 3–5 can
partly overlap.

---

## Key decisions log

| Decision | Choice | Notes |
|---|---|---|
| MVP surface | CLI-first | Frontend post-MVP; doesn't exist in repo yet |
| Agent architecture | Living-plan ReAct hybrid | Plan in state, revised in-loop, advisory not rigid |
| Action + plan-update | **Mechanical update_plan** (no LLM) for now | Local model can't emit JSON for revision reliably; LLM-based reviser is a post-MVP upgrade gated on a more capable model |
| Plan panel (CLI) | Rich **re-printed panel** (not `Live`) | Avoids conflict with the approval `input()` prompt; Textual/Electron = graduation path, no agent change |
| Model strategy | Role-based, modular, hardware-tier presets | Native tool-calling required for MVP |
| `context_builder` | Re-scope to lean grounding node (§8) | Profiles + manifests only; never duplicate tools (bound natively) or history (in `messages`) |
| Data scope | Files + KB only (proposed) | Email/calendar deferred until gate+memory proven — **to confirm** |

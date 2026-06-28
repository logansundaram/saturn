# Saturn — Feature Inventory

A complete map of what's in the codebase today, for the **keep / cut / re-derive** decision.
Sort each row's **Disposition** as `KEEP`, `CUT`, or `?`. Once it's filled in we can see whether
what's left is "Saturn minus N leaf modules" (delete in place) or "a different core" (rebuild earns it).

## Legend

**Layer** — what kind of thing it is, and the default disposition that implies:
- `Engine` — the control loop. Being *replaced* by the new small-model architecture → re-derive, don't port verbatim.
- `Trust` — approval / tracing / privacy / security. You said keep these → port onto the new engine.
- `Product` — the surface *around* the engine and trust stack. **This is the candidate bloat — the rows to actually sort.**
- `Dev` — tests, benchmarks, build/CLI plumbing. Keep what supports iteration.

**Coupling** — how hard it is to remove or move:
- `Leaf` — self-contained module; clean cut or clean port.
- `Seam` — leaf module, but wired into an engine seam; must be re-threaded when the engine changes.
- `Core` — woven through the loop; can't be removed without reshaping the graph.

---

## 1. Engine — the control loop  (`nodes/`, `core/`, `agent.py`)

> Being replaced by the new plan-time-routing / single-tool-execute / rectify architecture.
> Listed so you know exactly what the swap deletes or absorbs.

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| Living-plan ReAct graph (ground→plan→plan_gate→agent→approval→tools→update_plan→…→synthesize) | `agent.py` | Engine | Core | ___ |
| `ground` — context assembly (profiles, SATURDAY.md, memory, manifests, recap, attachments) | `nodes/ground.py` | Engine | Core | ___ |
| `plan` — structured Plan draft | `nodes/plan.py` | Engine | Core | ___ |
| `plan_gate` — plan-review checkpoint at every step boundary | `nodes/plan_gate.py` | Engine | Core | ___ |
| `agent` — all-tools-bound runtime tool-caller (the thing the new arch replaces) | `nodes/agent.py` | Engine | Core | ___ |
| `approval` — risk-tiered safety gate (engine seam for the trust stack) | `nodes/approval.py` | Engine/Trust | Seam | ___ |
| `tools` — execution + clamping + egress attribution + quarantine fencing | `nodes/tools.py` | Engine | Core | ___ |
| `update_plan` — mechanical positional/multiset step advance | `nodes/update_plan.py` | Engine | Core | ___ |
| `replan` — LLM judge: groundedness check, insert web_search | `nodes/replan.py` | Engine | Core | ___ |
| `synthesize` — streamed final answer + provenance + honesty note + orphan-cancel | `nodes/synthesize.py` | Engine | Core | ___ |
| Lockstep vs advisory mode (`runtime.lockstep`) | `nodes/agent.py`, `core/messages.py` | Engine | Core | ___ |
| Nudge floor (don't answer "no info" with a planned gather un-run) + nudge budget | `nodes/agent.py` | Engine | Core | ___ |
| Replan budget / web-only skip guard | `nodes/agent.py` | Engine | Core | ___ |
| Mid-turn steering (Esc-with-text → HumanMessage) | `core/plan_ops.py`, `nodes/plan_gate.py` | Engine | Seam | ___ |
| Plan-review pause (empty-Esc) + editor (add/edit/move/drop/go/abort) | `core/plan_ops.py` | Engine | Seam | ___ |
| Token budget ceiling (`runtime.token_budget`) | `core/budget.py` | Engine | Leaf | ___ |
| Auto-compaction + history collapse | `core/compaction.py` | Engine | Seam | ___ |
| `@file` mentions expansion | `core/mentions.py` | Engine | Leaf | ___ |
| AgentState shape + plan-as-dicts + accumulators | `core/state.py` | Engine | Core | ___ |
| All system prompts + per-call directives | `core/messages.py` | Engine | Core | ___ |
| Checkpoint pruning per turn (SqliteSaver) | `agent.py` | Engine | Core | ___ |

---

## 2. Trust stack  (`trust/`)  — you said keep these

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| Risk-tiered approval gate (read_only / side_effecting / destructive) | `trust/policy.py`, `nodes/approval.py` | Trust | Seam | ___ |
| Unified policy object — the 5 gate-relaxation views | `trust/policy.py` | Trust | Leaf | ___ |
| `/risk` overrides · `/allow` shell prefixes · `/autoapprove` · threshold · `--yolo` | `trust/policy.py` | Trust | Leaf | ___ |
| Policy profiles (export/import shareable YAML) | `trust/policy.py` | Trust | Leaf | ___ |
| Blast-radius readout (`/policy can`) | `trust/policy.py` | Trust | Leaf | ___ |
| Prompt-injection quarantine (scan + fence) | `trust/quarantine.py` | Trust | Seam | ___ |
| Quarantine gate escalation (first batch after injection gated) | `trust/quarantine.py`, `nodes/approval.py` | Trust | Seam | ___ |
| Data→action taint tracking (`taint_scan`) | `trust/quarantine.py` | Trust | Seam | ___ |
| The Glass Box — answer-level provenance | `trust/glassbox.py` | Trust | Leaf | ___ |
| Glass Box v2 — signed answer attestation | `trust/glassbox.py` | Trust | Leaf | ___ |
| Egress ledger + air-gap gate (the one network chokepoint) | `trust/egress.py` | Trust | Seam | ___ |
| Durable hash-chained egress log + chain verify | `trust/egress.py` | Trust | Leaf | ___ |
| Egress-chain anchoring (`log_tip` inside signed artifacts) | `trust/egress.py` | Trust | Leaf | ___ |
| ed25519 audit signing | `trust/signing.py` | Trust | Leaf | ___ |
| Signed trace exports | `trust/signing.py`, `commands/trace.py` | Trust | Leaf | ___ |
| Signed trust report | `trust/trust_report.py` | Trust | Leaf | ___ |
| Per-answer trust receipt + session posture line | `trust/receipt.py` | Trust | Seam | ___ |
| Secret redaction at the cloud boundary (off/warn/redact) | `trust/redaction.py` | Trust | Seam | ___ |
| Air-gap mode | `trust/egress.py`, `commands/privacy.py` | Trust | Seam | ___ |
| Dry-run mode (plan everything, execute nothing) | `nodes/*`, `commands/policy.py` | Trust | Seam | ___ |
| Standalone offline verifier + published spec | `utilities/saturn_verify.py`, `VERIFY_SPEC.md` | Trust | Leaf | ___ |
| Discovery hints + gate teaching preamble | `trust/receipt.py` | Trust | Leaf | ___ |

---

## 3. Tools  (`tools/`)

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| `calculate` (AST evaluator) + `current_time` | `tools/calculator.py` | Product | Leaf | ___ |
| `web_search` / `web_extract` / `http_request` (Tavily + DuckDuckGo + httpx) | `tools/web.py` | Product | Leaf | ___ |
| `read_file` / `write_file` / `edit_file` / `list_directory` / `search_files` / `find_files` | `tools/files.py` | Product | Leaf | ___ |
| `search_knowledge_base` (RAG) + `remember` / `recall` (memory) | `tools/knowledge.py` | Product | Leaf | ___ |
| `run_shell` (+ opt-in background jobs: `check_shell_job` / `stop_shell_job`) | `tools/shell.py` | Product | Leaf | ___ |
| MCP client — remote tools as `mcp_<server>_<tool>` | `tools/mcp_client.py` | Product | Seam | ___ |
| Tool registry + risk declaration + dynamic registration | `tools/toolspec.py`, `tools/registry.py` | Engine | Seam | ___ |

---

## 4. Stores / persistence  (`stores/`)

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| RAG engine (embed/sync; txt/md/pdf/html/csv/docx; ingest preprocessing; admission screening) | `stores/rag.py` | Product | Seam | ___ |
| Document + workspace manifests (LLM summaries, hash-cached) | `stores/document_registry.py` | Product | Seam | ___ |
| Durable memory (append-only memory.md) | `stores/memory_registry.py` | Product | Leaf | ___ |
| Snapshots / `/undo` (pre-write file snapshots) | `stores/snapshots.py` | Product | Seam | ___ |
| Trace DB (runs/events/llm_calls + LLMTraceHandler) | `stores/trace.py` | Trust | Seam | ___ |

---

## 5. Commands  (`commands/`)  — the big product surface; **sort this hardest**

| Command (aliases) | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| `/help` `/quit` `/clear` | `system.py`, `conversation.py` | Product | Leaf | ___ |
| `/docs` — RAG corpus management (add/remove/sync) | `knowledge.py` | Product | Leaf | ___ |
| `/models` — picker / bind / tier | `runtime.py` | Product | Leaf | ___ |
| `/context` — runtime readout + num_ctx | `runtime.py` | Product | Leaf | ___ |
| `/compact` — manual compaction | `conversation.py` | Product | Leaf | ___ |
| `/plan` — view + review/pause/lockstep modes | `plan.py` | Engine | Seam | ___ |
| `/trace` — observability hub (why/answer/invoke/calls/cost/state/export/verify/key/replay) | `trace.py` | Trust | Seam | ___ |
| `/glass` `/source` — provenance views | `trace.py` | Trust | Leaf | ___ |
| `/memory` — list/add/forget durable facts | `knowledge.py` | Product | Leaf | ___ |
| `/mcp` — server status + reload | `runtime.py` | Product | Seam | ___ |
| `/policy` (+ `/risk` `/allow` `/autoapprove`) — gate front door | `policy.py` | Trust | Leaf | ___ |
| `/undo` — revert file changes | `knowledge.py` | Product | Seam | ___ |
| `/rewind` — drop last exchange | `conversation.py` | Product | Seam | ___ |
| `/retry` — re-synthesize or full re-run | `conversation.py` | Product | Seam | ___ |
| `/init` — survey workspace, draft SATURDAY.md | `knowledge.py` | Product | Leaf | ___ |
| `/update` — self-update (git pull) | `system.py` | Product | Leaf | ___ |
| `/privacy` (+ egress / airgap / redact / report) | `privacy.py` | Trust | Seam | ___ |
| `/dryrun` — execution off switch | `policy.py` | Trust | Seam | ___ |
| `/resume` — sessions (save/restore/list/delete/rename/autosave) | `conversation.py`, `_session.py` | Product | Leaf | ___ |
| `/config` (+ setup/doctor, key) — YAML config front end | `config.py` | Dev | Seam | ___ |
| User-defined slash commands (`database/commands/*.md` → `/name`) | `user_commands.py` | Product | Leaf | ___ |
| Command grammar (REMOVE_VERBS / LIST_VERBS / `--save` / toggle-status) | `_framework.py`, `_utils.py` | Product | Core | ___ |
| Drag-and-drop file path handling (ingest/attach/text) | `core/mentions.py` | Product | Leaf | ___ |

---

## 6. TUI  (`tui/`)

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| 5-zone bottom-pinned status bar (posture/typeahead/progress/session/hardware) | `tui/ui/statusbar.py` | Product | Seam | ___ |
| Session posture line under banner | `tui/ui/statusbar.py`, `trust/receipt.py` | Trust | Seam | ___ |
| Plan rail rendering | `tui/ui/plan.py` | Product | Seam | ___ |
| Trace rail (node tree + egress leaves + gate-decision echo + judge verdict) | `tui/ui/trace.py` | Trust | Seam | ___ |
| Approval prompt UI (diff preview / full-width args / shell grant / secret warn) | `tui/ui/approval.py` | Trust | Seam | ___ |
| Response stream (token streaming + receipt + sources coloring + taint warning) | `tui/ui/response.py` | Trust | Seam | ___ |
| Glass Box renderer | `tui/ui/glass.py` | Trust | Leaf | ___ |
| Listing vocabulary (`section` / `table` / risk styles) | `tui/ui/listing.py` | Product | Leaf | ___ |
| Prompt (prompt_toolkit: `/cmd`+`@path` highlight, multiline, paste chips, rprompt posture) | `tui/ui/prompt.py` | Product | Seam | ___ |
| Type-ahead queue (queue while a turn runs) | `tui/typeahead.py` | Product | Seam | ___ |
| Saturn art / banner | `tui/ui/art.py` | Product | Leaf | ___ |
| System monitor (CPU/RAM/GPU/VRAM) | `tui/system_monitor.py` | Product | Leaf | ___ |
| Plain / UTF-8 fallback (no-rich path) | `tui/ui/_base.py` | Product | Leaf | ___ |

---

## 7. Models / providers  (`core/llms.py`, `config.yaml`)

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| Role-based model factory (planner/tool_caller/synthesizer/utility/judge) | `core/llms.py` | Engine | Core | ___ |
| Tiers (laptop / workstation / cloud-hybrid presets) | `config.yaml` | Engine | Seam | ___ |
| Providers: Ollama + Anthropic + OpenAI | `core/llms.py` | Product | Seam | ___ |
| Cloud-boundary model proxy (redaction + egress wrapping) | `core/llms.py` | Trust | Seam | ___ |
| Ollama locality detection (loopback vs remote) + embeddings boundary | `core/llms.py`, `trust/egress.py` | Trust | Seam | ___ |
| Model health check + hot-swap | `core/llms.py` | Dev | Leaf | ___ |

---

## 8. CLI / headless / config  (`agent.py`, `config.py`)

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| Interactive chat loop | `agent.py` | Engine | Core | ___ |
| Headless `-p` mode (`--json` / `--policy` / `--export`) | `agent.py` | Product | Seam | ___ |
| `verify` verb (offline artifact check) | `agent.py` | Trust | Leaf | ___ |
| `--replay` (render an export offline) | `agent.py` | Trust | Seam | ___ |
| Piped stdin attach | `agent.py` | Product | Leaf | ___ |
| `saturn.cmd` / `saturn.sh` launchers + strict argparse | `agent.py` | Dev | Leaf | ___ |
| YAML config (tiers/runtime/web/rag/mcp/shell/paths) + session-vs-persist | `config.py`, `config.yaml` | Dev | Core | ___ |
| Installed mode (SATURDAY_HOME / wheel seeding) | `config.py`, `pyproject.toml` | Dev | Seam | ___ |

---

## 9. Dev / benchmark / infra

| Feature | Module | Layer | Coupling | Disposition |
|---|---|---|---|---|
| Graded trust benchmark (grounding bait + gate probes) | `benchmark.py` | Dev | Leaf | ___ |
| Capability benchmark suites | `benchmark.py` | Dev | Leaf | ___ |
| Offline unit suite (~50 files) | `tests/` | Dev | Seam | ___ |
| CI (ubuntu + windows, wheel smoke test) | `.github/workflows/tests.yml` | Dev | Leaf | ___ |
| Diagnostic log | `diag.py` | Dev | Leaf | ___ |
| Graph printer | `utilities/print_graph.py` | Dev | Leaf | ___ |

---

## How to read the result

- **Mostly `Engine: Core`** rows are decided already — the new architecture re-derives them. Don't agonize over these.
- **`Trust`** rows are your stated keepers — the question for each is only *how cleanly it ports* (Leaf = drops on; Seam = needs the split-execute seam to attach to).
- **`Product: Leaf`** rows are the cheap cuts — removing any one touches nothing else. If most of your "bloat" lives here, **deletion-in-place wins and a rebuild is not justified.**
- **`Product/Engine: Core` or `Seam`** rows that you want gone are the real test — if there are *many*, that's the signal a clean rebuild earns its cost.

**Tally to compute once sorted:** count `CUT` rows by Coupling. Lots of `CUT + Leaf` → delete in place. Lots of `CUT + Core` → rebuild.

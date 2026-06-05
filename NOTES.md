# Saturday.ai — working notes

Scratch notes and the phase-by-phase build log, consolidated from the old `notes.txt`
and `progress.txt`.

---

## Scratchpad (feature ideas & design discussion)

Logan's scratchpad

Potential new features:

 Tier 1 — Do these first (highest signal/effort ratio)
  - Python REPL / shell exec — lets the agent run code, check results, and self-correct. Single biggest capability jump.
  - Persistent memory — a simple key/value or append-only markdown store the agent can write facts to and query. Your RAG is
  per-session; this survives restarts.
  - Clipboard / screen read — on a local machine, the user's actual context is on their screen, not in a prompt. Even basic
  screenshot-to-text unlocks a lot.

  Tier 2 — High value, moderate complexity
  - Browser automation (Playwright) — you have web_search but not click/fill/extract. Real browsing unlocks login-gated content, form
  submission, scraping.
  - Process / terminal control — run git, npm, docker commands and pipe back stdout/stderr. Combined with the REPL this makes it a
  coding assistant.
  - Structured task decomposition — a subtask planner that breaks a request into steps and tracks completion. Your current plan node
  is a router; this is a full planner.

  Tier 3 — Nice to have
  - Vision / multimodal — if you switch to a vision-capable local model (LLaVA, Gemma with vision), reading screenshots and diagrams
  becomes free.
  - Calendar / email — high value for personal productivity but adds auth complexity.
  - Proactive file watching — the agent monitors directories and surfaces changes without being asked.

  Local-specific concerns to design around:
  - Keep tool descriptions short and distinct — local models struggle with large tool schemas
  - Build an approval gate before destructive tools (shell exec, write) since you won't have a safety net
  - Add a "think step" before tool selection to reduce hallucinated tool calls (your plan node already does some of this)


Have a user config file to pass to the llm to have

Agent-wide and project-wide memory, descide what emmories are relvant to be shared to the entire agent, while what memories should remain for a certain execution

Config own tools and agent archietecture

Vision???

Need to work on rag, preprocess pdf's more effectively, append more descriptive medtadata so that the planner can make smarter choices, etc.


Change the splash image to a better looking saturn image but keep the animation

Make the TUI ui and expose system metrics 

Implement a system monitor like tui element


 The old architecture (what you had)

  The plan was a pure transparency surface — it decorated the trace but never touched control flow. The relevant edge:

  agent → route_after_agent → {approval, synthesize}

  route_after_agent made its decision off one signal only: did the AIMessage carry native tool_calls?

  - tool calls present → approval (→ tools → update_plan → back to agent)
  - no tool calls → synthesize

  So the plan's intended_tool: web_search annotation was advisory — nothing checked whether the planned tool actually
  ran. When gemma4:e4b answered "no information available" without firing the search, last.tool_calls was empty, routing
  fell to synthesize, and you got the non-answer. The plan said "search the web," the agent didn't, and no part of the
  loop noticed the gap.

  The new architecture

  Two things changed: the plan now feeds back into routing, and the agent node became self-correcting. The graph gained
  one edge — an agent → agent self-loop:

  agent → route_after_agent → {approval, synthesize, agent}
                                                        └── the nudge

  The new decision logic in route_after_agent (node_registry/agent.py):

  iteration ≥ max_iterations   → synthesize   (hard cap, unchanged)
  has tool calls               → approval
  no tool calls, BUT plan has an un-run gathering step
     AND agent_nudges < BUDGET → agent         ← NEW: loop back to act
  otherwise                    → synthesize

  The "un-run gathering step" check is the shared helper state.unrun_planned_tools(plan, tools_called): a step that's
  not terminal (not done/skipped) and has an intended_tool and that tool isn't in the tools actually called this turn.
  That's precisely the plan/execution gap the old design ignored.

  How the nudge actually rescues the turn

  Here's the subtle part. Routing back to agent on the same inputs would just reproduce the same refusal — the model
  already saw the plan and chose not to act. So the loop-back has to change what the model sees. That happens in
  agent_node, which now does plan-aware injection:

  - Every pass — a soft pointer at the first un-run step:
  "NEXT PLANNED ACTION — step 1: Search the web... The plan expects this step to call web_search. If that is the right
  next move, make the native tool call now."
  - On a loop-back specifically — agent_node detects it's being nudged (the last message is its own no-tool-call
  AIMessage while planned work remains — which can only happen via this new edge) and appends a pointed correction:
  "You returned without calling a tool, but the PLAN still has un-run step(s): ... You do NOT yet have this information,
  so you cannot answer fully. Call the indicated tool now. Do NOT claim information is unavailable while a search step
  is pending."

  That escalation is what breaks the refusal loop — the second attempt has strictly more pressure than the first.

  What stops it from spinning

  Two independent bounds, so a model that simply won't call the tool can't hang:

  1. NUDGE_BUDGET = 2 — a new per-turn agent_nudges counter (in AgentState, reset each turn, incremented by agent_node
  when it injects the correction). After 2 nudged retries, the guard stops.
  2. max_iterations — the existing hard cap still bounds every loop edge.

  When the budget runs out with the step still un-run, you fall through to synthesize — but synthesize_node now also
  checks unrun_planned_tools and, if any remain, injects an honesty instruction: say "I couldn't complete that lookup,"
  not "no information exists." So even the give-up path stops producing the misleading non-answer.

  The conceptual shift

  ┌──────────────────────┬─────────────────────────────────────────┬────────────────────────────────────────────────┐
  │                      │                 Before                  │                     After                      │
  ├──────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Plan's role          │ Advisory / display only                 │ Drives one routing edge (bounded)              │
  ├──────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Routing signal       │ tool_calls only                         │ tool_calls + plan/execution gap                │
  ├──────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Agent input          │ Static (sys + context + plan + history) │ + dynamic next-step pointer & escalating nudge │
  ├──────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Early-finish failure │ → misleading "no info" answer           │ → retry, then honest give-up                   │
  ├──────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ New state            │ —                                       │ agent_nudges counter                           │
  └──────────────────────┴─────────────────────────────────────────┴────────────────────────────────────────────────┘

  The honest framing: the plan went from descriptive (a record of intent) to partly prescriptive (it can pull the agent
  back to its own stated plan). It's deliberately a thin coupling — one edge, tightly bounded — not a rigid executor
  that forces tool calls. The model still decides; the loop just refuses to let it silently skip planned gathering work
  and then claim ignorance.

  Worth flagging: this reverses the "plan is ADVISORY" invariant still written in state.py and CLAUDE.md. And it's a
  safety net, not the primary fix — the gemma4:e4b → qwen3.5:9b swap fixes the base case (in your live run the nudge
  fired zero times because the 9B model called the tool on its own). The structure is there for the residual misses a
  more capable model still occasionally has.

✻ Churned for 28s

❯ is it better to have the plan as a suggestion than a strict follow?

● Short answer: for a general-purpose agent like this, suggestion is the right default — but the question is really
  "where on the spectrum," and the most defensible point isn't either pole. It's the soft-constraint middle you already
  landed on. Here's the reasoning.

  The core principle: who has more information?

  The plan is drafted in the plan node — before a single tool has run. It's a prediction made with the least information
  the loop will ever have. The agent, by contrast, decides each step with all accumulated observations (tool results
  sitting in messages).

  So "strict follow" means: trust the less-informed decision over the more-informed one. That's backwards for any task
  where the environment can surprise you. A search returns something unexpected, a file isn't where the planner assumed,
  a step turns out unnecessary — a strict executor faithfully marches through the now-wrong plan. Suggestion lets the
  better-informed agent pivot.

  Strictness doesn't remove the failure point — it relocates it from the agent loop to the planning step, which is the
  one place with the worst information. If the plan is wrong, strict execution executes the wrong thing perfectly.

  Two different failure modes

  - Suggestion protects against a bad plan — you can recover mid-loop.
  - Strict protects against a wandering agent — it can't drift or skip.

  Neither is free. With a weak model both risks are real (the wandering-agent risk is literally the bug we just fixed).
  The interesting thing about your nudge is that it buys some of strict's protection without paying suggestion's price:

  ▎ it stops the agent from silently skipping a planned gathering step, but still lets it deviate, add extra tool calls,
  ▎ or pivot on results.

  That's a floor, not a ceiling. The plan says "at minimum, gather this" — it doesn't say "do exactly this and nothing
  else." That asymmetry is the correct shape for a suggestion-with-teeth: it catches the omission that produces
  confidently-wrong answers, while leaving the agent free to do more than planned.

  When strict actually wins

  It's not never. Rigid execution is better when:

  - The task is well-specified and repeatable — a known pipeline (ETL, a deploy sequence). The plan isn't a guess there;
  it's a spec.
  - You need guarantees — safety/compliance ("these actions and no others"), cost control (no surprise tool calls), or
  reproducibility (deterministic evals).
  - The model is too weak to be trusted to adapt at all — sometimes a rigid rail beats a confused agent improvising.

  But note: even then, a strict plan can't insert steps it didn't foresee — and your own CLAUDE.md flags mid-loop step
  insertion as post-MVP. Reality routinely needs a step the planner couldn't predict. Suggestion accommodates that;
  strict can't.

  For this agent specifically

  Saturday's stated goal is a general-purpose local agent — open-ended, exploratory queries in an uncertain environment
  (web, files, RAG). That's the regime where suggestion dominates: you can't pre-plan correctly because you don't know
  what you'll find. Going strict would trade away the one thing ReAct is for — adapting to observations.

  So my recommendation: keep it as suggestion, and you've already added the right amount of teeth. The graded design —
  advisory plan, a bounded nudge that enforces only the gathering floor, an honest give-up when even that fails — is
  well-matched to a general-purpose agent on a fallible local model. Don't tighten it into a strict executor.

  If anything, the next refinement isn't more strictness — it's making the plan itself adaptive (let update_plan
  insert/revise steps as observations come in, the LLM-based reviser the docs gate on a capable model). Now that you're
  on qwen3.5:9b, that's newly viable, and it pushes in the opposite direction from strict: a plan that learns during the
  turn rather than one that's obeyed regardless. That's the more interesting frontier than tightening the leash.

✻ Baked for 42s

---

## Build log (phase-by-phase progress)

 1. Planning & architecture (discussion → locked decisions)

  We explored the existing repo and designed the MVP for Saturday.ai (local-first, transparent, general-purpose agent). Key decisions
  reached:

  - CLI-first MVP; Electron frontend deferred.
  - Living-plan ReAct hybrid — the middle ground between pure ReAct and plan-execute: a mutable plan lives in state, drafted up front
  and revised in-loop, advisory not rigid. It doubles as the transparency surface.
  - Role-based modular models (get_model(role), hardware-tier presets) — deferred to Phase 3.
  - Streamed plan-event → subscriber rendering, so the panel renderer is swappable.
  - context_builder re-scoped to a lean grounding node (drop tool inventory + chat history as redundant).
  - Defined the essential tool suite (run_python, search_files, web_fetch, search_knowledge_base, remember/recall as the key
  additions).
  - Honest competitiveness assessment (not yet competitive as code; the direction is competitive in the local-agent niche).

  2. Docs & memory

  - Created SATURDAY_MVP_PLAN.md — full plan: vision, current-state assessment, target architecture (§1–8), tool suite, workflows,
  benchmarks, phased roadmap, decisions log.
  - Rewrote CLAUDE.md twice — first to match reality (flat repo, no frontend, the path bug), then again after Phases 1+2 to the
  living-plan loop architecture.
  - Saved project memory (saturday-mvp-architecture.md + index), kept updated as decisions landed.

  3. Phase 1 — living-plan ReAct loop (implemented)

  - state.py — added PlanStep/Plan, plan/iteration/verified/verifier_feedback fields, accumulator reducers; dropped the old router
  booleans.
  - Built the loop: ground → plan → agent ⇄ tools → update_plan → … → synthesize, replacing the static fan-out. New/repurposed:
  tool.py (agent_node/tool_node/route_after_agent), plan_node (drafts a Plan), reflect.py (update_plan), grounding context_builder.
  - Added search_knowledge_base (RAG as a tool), wired build_ingest at startup, fixed the __file__ path bug (workspace →
  database/workspace/).
  - Verified: multi-tool chaining works (workflow #3).

  4. Bug fixes from your benchmark run

  - 'str' has no attribute page_content → fixed synthesize's doc handling.
  - Invalid json output (local model can't emit plan JSON) → plan_node fallback + made update_plan mechanical (no LLM).
  - UnicodeEncodeError → forced UTF-8 console output.
  - Synthesis faithfulness (calculator returned wrong numbers) → fixed by pairing each tool result with its call
  (calculate(expression=…) -> 260621), so the model stops recomputing.

  5. Phase 2 — safety & transparency (implemented)

  - Approval gate — risk tiers (registry.TOOL_RISK/risk_of) + interrupt-based approval_node; read-only passes through, side-effecting
  prompts. Verified workflow #6 (decline = no-op, approve = acts, read-only never prompts).
  - SQLite trace (trace.py) — runs + events per turn, every run inspectable.
  - Live plan panel (ui.py) via Rich.
  - SqliteSaver checkpointer + unified run_turn (stream → trace + UI + interrupt/resume); per-turn thread_id.
  - Sandbox hardened (is_relative_to); installed rich + langgraph-checkpoint-sqlite.

  6. PlanStep-as-dict cleanup

  - Moved the plan to plain dicts in state (Pydantic only at the planner boundary via steps_to_dicts), killing the SqliteSaver
  deserialization warning. Updated all 7 call sites; verified the warning is gone and the loop still works.

  7. Blueprint-terminal TUI

  - Redesigned ui.py: thin cyan grid tables, uppercase labels, text status (PENDING/ACTIVE/DONE/SKIPPED), calm cyan accent, emoji
  removed, ASCII fallback. Synced the doc reference.

  ---
  Where things stand

  - Phases 1 & 2 are complete and verified. Phase 0 is mostly done (one leftover: the vector store re-embeds every run — no on-disk
  cache).
  - The agent is now a real agentic loop with safe tool use, a live plan view, a full trace, and resumable sessions.

  Open / next:
  - Phase 0 leftover: persist embeddings to disk.
  - Phase 3 — persistent memory + config.yaml + get_model(role) (also the fix for small-model reliability, the biggest remaining
  quality limiter).
  - Minor: write_file append default, calculate raw floats, the gemma4:e4b occasional fake-tool-call-as-text.

  One bookkeeping note: the Phase 1/2 task lists (#1–17) are all marked complete, so the tracker is clean for whatever you pick up
  next.

  for me to verify what claude has given

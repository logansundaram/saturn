# The quick path — design (2026-09-08)

The spec `PLAN.md` → "The common-case contract" promised. Shipped the same day; this records
the design as built, the measurements that justified it, and what was deliberately left out.

## The problem, in the traffic's own numbers

Of the 46 traced turns since 2026-09-02, 41 plans were single-step and 17 were a lone reasoning
step with no tool. Every one of them paid the plan call (5–19 s on the 9b — thinking, which
must stay ON for the planner or it drafts a lone `ask_user` stub), an execute pass, and the
rectify judge (3–5 s) before synthesize wrote the answer. The judge said `rectify=true` 13 times
in 53 and never once on a clean, fully-done plan. The engine was built for the hardest five
percent and the other ninety-five paid for it in seconds.

## The design

One engine, plus a cheap check that routes around it.

**1. The complexity check — `core/complexity.py`.** A regex over the user's own words, zero
tokens, deterministic, reusing the readings `core/request_intent.py` already trusts. Plan mode
is forced by a state change (`wants_state_change`: write/edit/delete/remember/the calendar,
notes, mail and reminder effects), a change verb the effect vocabulary leaves out ("change X to
Y", "replace", "rename", "fix"), a computed figure (an aggregation word or an arithmetic
expression), a reference hop, a request to be asked, more than one workspace path, or a
multi-clause request ("… then …", "for each …", "…; …", ", and read …"). Everything else is
simple. One exemption: "write/draft/compose me a story" is prose, not a file, unless a file, a
save, or a path is also named. The check errs toward the engine: a request planned
unnecessarily is merely slower; a request the quick path cannot finish is handed over.

**2. The quick node — `nodes/quick.py`.** One structured call through the existing hardened
path (`core/structured.structured`, the `tool_caller` role, the `tool_args` task: think off,
512-token bound) under a grammar whose tool enum is every registered tool plus `answer`. The
prompt (`core/messages.quick_sys_msg`) describes the read-only set with argument shapes IN the
system message — never a native bind, which renders a schema block into the chat template and
re-prefills the prompt whole (8k tokens / 20 s, the measurement behind the execute node's
grammar) — and names the other tools so the model can hand over by choosing one. The outcomes:

- `answer` → synthesize, unchanged: it streams, cites, grades confidence, discloses incidents.
- a `QUICK_TOOLS` tool (`core/plan_context.py`: web, files, search, knowledge base, calculator,
  clock, recall, the Apple readers; filtered at prompt time by the LIVE risk tier so a
  `/policy risk` raise drops a tool out, never in) → a pending step appended to the plan — THE
  data bus, unchanged — and a tool-calling AIMessage, so approval, tools, egress attribution,
  quarantine and update_plan run exactly as for the execute node. `update_plan` routes back to
  quick by `state["route"]`, and the next decision sees each result as its own message, capped
  at 3000 chars for the router's eyes (the full clamped observation still rides the step and
  `tool_results` into synthesize). Prompt-cache order is preserved: `[system][stable
  grounding][request + per-turn grounding][results…]` — the first two are a primed lineage
  (`core/prime.py`, first in the list because it is the first call of most turns), so a turn's
  first call prefills only the request and each later call extends the previous one.
- any other tool, nothing parseable, an error observation, or a further call past the
  three-call budget (the model is told when the budget is spent and may still answer) → the
  turn is HANDED OVER: `route = "plan"` with a `reasoning` line; `plan_node` drafts fresh when
  nothing ran, `replan` keeps the gathered steps and drafts the rest.
- a guarded outcome (a gate decline, an air-gap refusal) → synthesize, disclosed, never retried
  — rectify's first-branch rule.

**3. Overrides.** `/quick <request>` and `/plan <request>` in the REPL (a command hands the
turn back to the loop with its route forced; the removed `/plan save|recipes|run|lockstep`
verbs still error rather than becoming requests), `--quick` / `--plan` headless, and
`runtime.quick_path: false` to plan every turn. A `/draft`-seeded plan is the engine's by
definition.

**4. What the quick path gives up, by design and only on turns the check judged simple.**
Plan-level review (no plan_gate visit — Esc still steers nothing on a quick turn, since there
is no remaining plan to redraft), the rectify judge, and the semantic write gate, which has
nothing to gate because no writing tool is reachable.

## Measurements before shipping

Regex over the 31 distinct traced requests: 24 agree with the plan that actually ran; 5 sent to
the quick path that the engine had handled (a week's news digest that needed four extracts,
follow-ups to a refused appointment, a typo'd "appointmenet" the effect vocabulary missed) —
all of which the quick node hands over; 2 sent to the engine that a quick answer would have
served ("write me a story" without the prose exemption, since added).

Live router prototype on the 9b, one primed lineage, 26 requests
(`docs/OPTIMIZATIONS.md` §8 has the table): the first decision took 0.5–0.8 s on every request
(prompt ~1370 tokens, ~50 prefilled); the second, with a search or a document result, 1.4–2.4 s.
Every grounding bait chose `web_search` on the first call; both injection probes chose
`search_knowledge_base`; the calendar and reminder requests named `create_calendar_event` and
`schedule_notification` — the hand-over — on the first call; "review the emails" listed the
inbox and read three, then hit the budget. The same requests' traced engine cost
(plan + execute + rectify + replan, synthesize excluded) was 7.5–115 s.

## Not done, and why

- **No judge on the quick path.** The groundedness rule ("current facts never looked up") is
  carried by the router prompt instead; measured 6/6 baits searched up front. If a tier ever
  answers a bait from memory, the fix is the prompt or a deterministic request-side reading,
  not an eleventh rectify branch.
- **No union grammar with per-tool argument schemas (`oneOf`).** A free `arguments` object plus
  `core/tool_args.coerce_args` matched every read-only tool in the prototype; a schema that does
  not fit hands the turn over rather than retrying.
- **`ask_user` is out of the set.** A question to the user is the engine's seam (the ask gate,
  the dangling-ask redraft); a quick turn that needs one hands over.

# Saturn documents

The three files a repository needs stay at the root: `README.md` (the front door),
`CHANGELOG.md` (user-visible history, Keep a Changelog) and `CLAUDE.md` (the working notes for
Claude Code). Everything else is here.

| File | What it is | Read it when |
|---|---|---|
| `ARCHITECTURE.md` | The guided code map and reading order. | Before touching `nodes/` or `trust/`. |
| `pivot.md` | The product goal since 2026-09-27 and the ranked work that closes the distance to it: features, cuts, the loop items, the two knives for deciding. | Choosing what to build next. |
| `engine.md` | The loop's shape today — nodes, prompt order, the guards, the observation path — and the ranked engine improvements with the benchmark evidence behind them. | Changing the loop itself. |
| `dogfood.md` | The prompts a real user would try, written against the direction, not what is built. | Dogfooding; turning failures into pivot items. |
| `advantages.md` | Why the pivot items matter: what a local, terminal-native agent can promise that a cloud one cannot. | Deciding whether an item is worth its day. |
| `OPTIMIZATIONS.md` | Latency techniques for the one loop — the prefix cache and the prime, thinking, warm-up — shipped, next and closed, with the numbers. | Anything touching prompt order, the prefix cache or the model call. |
| `research.md` | Everything still open in `pivot.md` / `engine.md` / `advantages.md`, ranked, with the plan written for each of the top items; then an outside survey (the agent literature, what the vendors and open-source agents ship) turned into further ideas, with sources. | Picking the next item, or checking whether an idea has already been weighed. |
| `superpowers/specs/` | Design specs, one per feature, dated; each opens with its status. All shipped except `2026-10-02-macos-app-identity.md` (Saturn holding its own macOS permissions instead of the terminal: undecided, nothing built). `2026-10-04-adaptive-thinking-design.md` is built and measured (`auto` thinks before acting); its sampling experiment is not run. `2026-10-04-know-the-user-design.md` is the one design above the auto-memory, first-run-interview and launch-brief plans, with what a 2026-10-04 survey changes in each: its auto-memory part is built (see its "As built"), unmeasured on a model; the interview is built too (2026-10-05, its second "As built"); incognito, the memory receipt and `/memory import` are not. | Understanding why a mechanism is shaped the way it is. |
| `superpowers/plans/` | Implementation plans. Of the ones dated 2026-10-01 `terminal-escape-sanitising`, Phase 1 of `skills` (its Phase 2, the model loading a skill by itself, is open) and `auto-memory-from-user-statements` (Tasks 1–10; Task 11, the measurement on a model, is not run) are built, and so is `first-run-interview` (Tasks 1–6 with the know-the-user spec's I1–I3; Task 7, the manual dogfood, is not run); the rest are open (`research.md` ranks them; each carries its own design section); `2026-10-03-create-skill.md` (the agent's `create_skill` tool) is built; `2026-10-04-adaptive-thinking.md` is executed; the two from 2026-09-27 and 2026-09-29 are the plans their specs were built from; `2026-09-02-tool-call-output-cap.md` is a plan for the deleted plan/execute engine, kept as history. | Building one of the open items; otherwise history. |

Conventions: a new document goes here, dated in its first line; a document that describes a
deleted mechanism is deleted with it (the 2026-09-27 cut removed five); user-visible changes go
in `CHANGELOG.md` under `[Unreleased]`, not here.

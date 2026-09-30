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
| `OPTIMIZATIONS.md` | Latency techniques: shipped, next, to measure, with the numbers. | Anything touching prompt order, the prefix cache or the model call. |
| `superpowers/specs/` | Design specs from past feature work, one per feature, dated. | Understanding why a mechanism is shaped the way it is. |
| `superpowers/plans/` | The implementation plans those specs were built from. | Rarely — history. |
| `superpowers/TWEAKS.md` | A parking lot for small changes worth weighing before doing. | Looking for an afternoon's improvement. |

Conventions: a new document goes here, dated in its first line; a document that describes a
deleted mechanism is deleted with it (the 2026-09-27 cut removed five); user-visible changes go
in `CHANGELOG.md` under `[Unreleased]`, not here.

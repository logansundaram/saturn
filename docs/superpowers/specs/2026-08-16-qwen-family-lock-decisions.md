# SDD ledger — plan: docs/superpowers/plans/2026-08-16-qwen-family-lock.md

Spec: docs/superpowers/specs/2026-08-16-qwen-family-lock-design.md (read — binding authority)
Branch: transplant/isolates-2026-08 (feature branch, not main)
Workspace: .superpowers/sdd/2026-08-16-qwen-family-lock

## Pre-flight scan

### Cross-task rows (shared file or interface)

| A → B | produces → consumes | finding |
|---|---|---|
| T1 → T2 | `in_family`/`migrate`/`tag_for` → `Config._enforce_family` | clean; T2 composes `tag_for(migrate(id))` per spec |
| T1 → T3 | `SIZE_LADDER`/`classes()` → template invariant tests | clean |
| T1 → T5 | `SIZE_LADDER`/`in_family` → `_bind` gate, `_tier_rows` | clean |
| T1 → T10 | `FAMILY_PREFIXES` → chat_template agreement test | clean |
| T2 → T4 | `migrated_bindings()` → `_migration_problems()` | clean; both read the module-global ledger |
| T2 → T5 | `Capability.max_context_window` → `_tier_rows` ctx columns | clean |
| T3 → T5 | config.yaml capabilities entries → `_tier_rows` test asserts ctx 32768 / max 262144 | **DEPENDENCY**: `_tier_rows` reads the LIVE config, so T5's test only passes after T3 edits `config.yaml` (not just the template). Order T3 before T5 — the plan already does. No change needed. |
| T5 → T7 | `confidence.calibration_for` → `_tier_rows` `calibrated` flag | clean; returns a truthy record either side of T7's overlay change, and the test only asserts `isinstance(bool)` |
| T6 → T7 | `confidence_store.entry_for` → `calibration_for` | clean |
| T6 → T9 | `write_entry`/`clear_entry`/`entry_for` | clean; signatures match the command's call sites |
| T8 → T9 | `calibration.measure(tag, prompts, on_progress)`/`PROMPTS` | clean; the test spy mirrors the signature exactly |
| T9 → T10 | `_GROUPS` registration → `test_help` registry cross-check | clean |
| T12 → T7 | regenerated CALIBRATION → overlay-order tests | clean; T7's tests read the shipped value dynamically, never a literal |
| T12 → T5 | regenerated CALIBRATION → `calibrated` flag | clean; flag flips to all-true, test asserts type not value |
| T1,T2,T3,T4,T10 | all append classes to `tests/test_model_family.py` | clean; strictly sequential appends, no overlapping edits |

### Per-task self-consistency rows

| Task | own tests vs own code | finding |
|---|---|---|
| T1 | migrate table + size parse vs assertions | **CONFLICT — see Ruling 1** |
| T2 | `Config({...})` fixtures vs `capability_of`/`num_ctx_for` | clean; `num_ctx_override` reads `runtime.num_ctx`, fixture supplies it |
| T2 | `config.py` importing `core.model_family` at module scope | clean — `core/__init__.py` is docstring-only and `model_family` imports nothing, so no cycle; config's leaf property holds |
| T3 | `str(template)` retired-id scan | clean; YAML comments are not parsed, so the explanatory comments naming gemma4 cannot trip it (they were rewritten anyway) |
| T4 | `_migration_problems` imports config inside the fn | clean; avoids the llms↔config import cycle |
| T5 | `_bind` gate placement vs the embedder exemption | clean; `target != "embedder"` guard precedes the early return |
| T6 | atomic write + mtime cache vs the fail-soft tests | clean |
| T7 | overlay-wins tests vs `exit_threshold`'s existing table read | clean; `exit_threshold` already reads through `calibration_for`, so no second edit needed |
| T8 | `summarize` sort-independence vs `quantile` sorting internally | clean |
| T9 | tests mutate the real config singleton | **CONFLICT — see Ruling 2** |
| T10 | template deletion vs the three named test lines | clean; verified by grep that no other test binds a retired id |
| T12 | live-daemon task inside an otherwise offline plan | clean; flagged as such in the plan and gated on `ollama list` |

### Rulings

**Ruling 1: T1's `mystery:33b` assertion is wrong; the rule is right.**
The plan's Step 1 test asserts `migrate("mystery:33b") == "27b"`, but the nearest-class rule over
`_CLASS_PARAMS` gives |33−27.3| = 5.7 vs |33−36.0| = 3.0 → `35b`. The migration rule matches the
spec ("the parameter count parsed out of the tag mapped to the nearest class"); the assertion does
not. Fix the ASSERTION to `== "35b"`, keep the rule.
*Cost if wrong:* a legacy 33b-class binding lands on 35b instead of 27b — a larger model than
intended, caught immediately by the startup warning naming the substitution.

**Ruling 2: T9's tests must restore `runtime.confidence` on the shared config singleton.**
`get_config()` is a process-wide singleton; the toggle tests leave it `False`, which would leak
into every later test file that reads `confidence.enabled()`. Add an autouse fixture to
`tests/test_confidence_command.py` saving and restoring `runtime.confidence` around each test.
*Cost if wrong:* cross-file test pollution producing failures unrelated to the change under test.

## Progress

Task 1: dispatched (BASE 6ae7187, sonnet, brief task-1-brief.md, Ruling 1 carried in dispatch)
Briefs for Tasks 2-12 pre-extracted.
Task 1: complete (commits 6ae7187..dcdb5db, review clean — spec OK, quality approved, 27 tests)
Task 1: minor (deferred): dead `except ValueError` branch in migrate() — _SIZE_RE can only capture float-parseable digits
Task 1: minor (deferred): model_family docstring claims a chat_template prefix-agreement test that Task 10 adds — self-resolves at T10; verify then

**Ruling 3 (corrects a preflight-scan error): test_onboarding's synthetic tier ids must become real in-family tags.**
My preflight row said test_onboarding.py's "tiny"/"mid" placeholders were synthetic fixtures "the gate
never sees". That was WRONG: commands/config._tier_honesty_line:352 calls cfg.model_for_role("tool_caller"),
so the seam substitutes them and test_tier_honesty_fires_on_the_first_declared_preset's `assert "tiny" in
line` fails. The PRODUCTION behavior is correct and in fact improved — the doctor's honesty line should
name the model that will actually run, not the one config.yaml claims. Only the fixture is stale.
Decision: swap the placeholders to in-family tags ("tiny"->"qwen3.5:2b", "mid"->"qwen3.5:9b") with matching
_CAPS keys, and update the one literal assertion. Do NOT weaken the seam to keep a placeholder working.
Owner: Task 2 (fix round 1) — not Task 10; the breakage is Task 2's to clear.
*Cost if wrong:* a doctor test asserts against tags that differ from the shipped ladder — cosmetic, caught
by the next test run.
Task 2: fix round 1/5 (1 addressed, 0 open — onboarding fixture -> in-family tags; commits f8cee69..1d28427)
Task 2: complete (commits dcdb5db..1d28427, full suite 992 passed 0 failed)

**Ruling 4: compress the remaining orchestration.**
User challenged the process weight. Assessed: the code substance is real (a model id is read across
config/llms/chat_template/commands/installers/tests) but 9 more per-task subagent+reviewer pairs is more
scaffolding than the work earns, and the plan already contains the code to transcribe. Decision: collapse
Tasks 3-11 into batched dispatches by area and run ONE whole-branch review at the end instead of per-task
reviews. Task 12 stays separate (live GPU). Proceeding without waiting on an answer because the produced
code is identical under either process — only orchestration differs, so no assumption can spoil the work.
*Cost if wrong:* defects that a per-task review would have caught surface later, at the final review,
where the fix diff is larger. Mitigated by the full suite being green at every batch boundary.
Batches: A=T3+T4 (tiers/capabilities/installers + startup reporting), B=T5 (/models), C=T6+T7+T8
(overlay store, threshold resolution, calibration move), D=T9 (/confidence), E=T10+T11 (template + docs).
Authorization: user confirmed the compressed plan and pre-authorized Task 12's live calibration pass
("trigger the live calibration yourself when ready") — no further check-in needed before running it.
Batch A: dispatched (BASE 1d28427, sonnet, briefs task-3 + task-4)
Batch A: commits dfdfc7c (tiers) + bcfeb5b (startup reporting); suite 1003 passed 0 failed

**Ruling 5 (my design error): rename the size class `0.8b` -> `800m`.**
`config.get/set/persist` parse dotted paths, so a tier KEY containing a literal "." is split into two
segments. commands/runtime.py builds `f"tiers.{active_tier}.roles.{role}"` at 4 sites and
commands/config._persist_key routes `tiers.*` through the same grammar — with active_tier="0.8b" a
`/models synthesizer <id>` would write to tiers->0->8b->roles instead of the real tier, silently
corrupting config. The implementer already fixed the read path (dict membership at runtime.py:209);
the write/persist paths remain broken. Options weighed: (a) rename the key, (b) make the dotted grammar
dot-safe across get/set/persist. Chose (a): (b) touches _set_yaml_scalar, the comment-preserving persist
seam, which is load-bearing and out of scope for a naming problem I introduced. "800m" keeps the metric
naming the user asked for and is dot-free. The MODEL tag `qwen3.5:0.8B` is unchanged — that one is
Ollama's, not ours. Adding a guard test: no size-class key may contain "." (a future "1.5b" would
reintroduce this).
*Cost if wrong:* the smallest tier is spelled 800m rather than 0.8b in `/models tier` — cosmetic, and
the metrics column still shows the true 873M parameter count.

**Ruling 6: installers' SATURDAY_TIER default must move off the deleted `laptop` tier.**
install.sh:20 and install.ps1:18 default to `laptop`, which Task 3 removed; a fresh install with no
override would fail at "Setting active tier". Out of the brief's literal line-scope but a direct
consequence of it. Set both to `4b`, matching config.default.yaml's active_tier.
*Cost if wrong:* a fresh install lands on the wrong tier — visible on first run, fixed with /models tier.
Batch A: fix round 1 (rulings 5+6 applied; commit d9248db); suite 1004 passed 0 failed
Batch A: complete (commits 1d28427..d9248db)
Batch B: dispatched (BASE d9248db, sonnet, brief task-5-brief.md, 800m rename carried)
Batch B: complete (commit 2b07b0e; suite 1010 passed 0 failed; also repaired 6 stale non-family
placeholder ids in tests/test_command_grammar.py — correct call, same class as Ruling 3)
Batch C: dispatched (BASE 2b07b0e, sonnet, briefs task-6 + task-7 + task-8)
Batch C: commits 05c0c86 + c2217c2 + 5268be7; suite 1032 passed 0 failed

**Ruling 7 (plan defect I introduced): restore the calibration guards the brief's replacement code dropped.**
The implementer flagged that my verbatim CLI replacement lost the banner, the <50-scored-token skip guard,
and the p50 diagnostic. Reviewing the pre-move code (5268be7~1) I found a FOURTH, unflagged loss that
matters more: the old CLI computed `exit_ = round(max(enter, _quantile(ps, 0.10)), 4)`; my `summarize()`
does `round(quantile(ps, 0.10), 4)` with NO max clamp. If p05 == p10 (a flat distribution or rounding
collision) the recorded exit can equal or sit below enter, and confidence.exit_threshold's
`if th <= v <= 1.0` guard then silently discards it and falls back to the 1.5x derivation — a calibrated
model quietly behaving as an uncalibrated one. Decision: put BOTH invariants in core/calibration.py so the
CLI and /confidence tune inherit them — the `max(enter, ...)` clamp inside summarize(), and a MIN_TOKENS=50
constant both callers check. Restore the banner and p50 print for the live run's legibility.
Carry into Batch D: Task 9's "refuse a zero-token measurement" becomes "refuse tokens < MIN_TOKENS".
*Cost if wrong:* a model with a degenerate probability distribution records an exit == enter; hysteresis
collapses to a single threshold for that model. Detectable in the Task 12 verification step.
Batch C: fix round 1 (ruling 7 applied; commit b8b5419); suite 1033 passed 0 failed
Batch C: complete (commits 2b07b0e..b8b5419)
Batch D: dispatched (BASE b8b5419, sonnet, brief task-9-brief.md; carries Ruling 2 autouse fixture
+ Ruling 7 MIN_TOKENS)
Batch D: complete (commit 9a88770; suite 1059 passed 0 failed)
Batch E: dispatched (BASE 9a88770, sonnet, briefs task-10 + task-11)
Batch E: complete (commits 5d4a5bf + e8c989c; suite 1061 passed 0 failed)
CLAUDE.md is gitignored in this repo (.gitignore:71) — edited on disk, deliberately NOT force-added.
Calibration pass run: 5/6 measured cleanly, monotonic by size (0.0261/0.0546/0.1766/0.2538/-/0.3859).
qwen3.8:27b returned 11 scored tokens — Ollama 0.32.13 emits logprobs for qwen3.8 on the FIRST CHUNK
ONLY. Ruling 7's MIN_TOKENS guard caught it on its first real run and recorded nothing.
Dropped the two now-unbindable rows (gemma4:e4b, qwen3-coder:30b); kept qwen3.6:27b (in-family,
bindable by hand).

**USER DECISION (not a ruling): qwen3.8:27b stays bound at the 27b tier with ESTIMATED thresholds.**
User: "Give best guess estimated values for qwen3.8 as it is the most advanced model and I want it to
be supported." Implementation constraint I am adding to honor it without breaking the transparency
premise: the estimate must be LABELED as an estimate at every surface (/models tier, /confidence status,
CHANGELOG, README) — never rendered as "calibrated". Value derived by log-linear interpolation on
parameter count between the two same-run measured neighbours (9.7B->0.2538, 36.0B->0.3859; ln-fraction
0.789) giving enter 0.358, exit 0.606, with the derivation recorded in the row.

Whole-branch review returned 2 Critical, 7 Important, ~18 Minor. Dispatching ONE fix wave.
Fix wave: complete (commits 06a1754, 8e116a5, 28463c8, 7f5acd1; suite 1112 passed 0 failed)
Scoped re-review: ALL findings ADDRESSED (2 Critical, 7 Important, 10 Minor, 6 test-quality); both
owner revisions verified in the final state; the five measured rows byte-exact; family gate tightened
not weakened; embedder exemption survives at both doors; no new exception reaches a streaming path.

**Ruling 8: fix the --inherit contract drift rather than defer it.**
Re-review found one NEW Important: utilities/confidence_calibrate.py's --inherit still writes the old
{inherited_from, no source} shape, while the wave introduced source:'estimated'/estimated_from AND two
committed contract tests enforcing it. So regenerating the table with the documented flag yields a table
that FAILS the suite, and the hand-authored qwen3.8:27b row can't be reproduced by its own generator —
a one-producer/one-parser violation in the module whose docstring says "do not edit by hand". The skill's
default is to surface a residual rather than open a second wave, but this is load-bearing (it breaks the
documented regeneration path) and ~5 lines. Dispatched a scoped fix.
*Cost if wrong:* a dev utility writes a slightly different provenance string than intended — visible
immediately in the generated file.

**Correction to two agents' deferred observation (verified false).** Both the whole-branch reviewer and
the re-reviewer claimed CLAUDE.md and documentation.md still describe gemma4 tiers and laptop/workstation
presets and need a branch-wide pass. Verified directly: CLAUDE.md carries the 19-command list including
/confidence, the six size-class tiers, max_context_window, and a line stating laptop/workstation/bench-coder
are GONE; documentation.md has 27 family-lock references. Both were updated in Batch E. The reviewers
pattern-matched on the tier-name strings without reading the surrounding clause. Also: CLAUDE.md is
gitignored, not "tracked" as the re-review stated. NO doc pass needed.
--inherit contract fix: complete (commit 948cc06; suite 1117 passed 0 failed)

**Ruling 9 (CRITICAL — found by my own end-to-end smoke test, missed by every review):**
`core/structured.py:311` builds `payload = list(messages) + [SystemMessage(content=shape)]` — a
TRAILING system message. Ollama 0.32.13 rejects that for qwen3.8 with `system message must be at the
beginning (status code: 500)`; qwen3.6 tolerated it. Because the 27b tier now binds qwen3.8:27b, EVERY
structured call on the default tier fails all 3 attempts and returns its safe default: plan drafting
(-> PLAN_PARSE_ERROR incident every turn), the rectify verdict, the resolution check, and the write gate.
The turn still answers, but the whole plan/execute engine degrades to a disclosed incident path.
Measured directly: qwen3.8 trailing-SystemMessage RAISES, trailing-HumanMessage validates; qwen3.5:9b
validates either way. Decision: move the shape hint to a trailing HumanMessage — universally compatible,
one line, and the hint is a formatting instruction that reads naturally as the last user turn.
Not caught by the suite because every structured test stubs the model; not caught by review because it
is a runtime template incompatibility, invisible in a diff.
*Cost if wrong:* the shape hint carries user-turn rather than system-turn weight for the four judgment
calls; plan quality is the thing to watch, and the contract/benchmark suites are how you'd see it.
Ruling 9 fix: complete (commit 510a7d2; suite 1118 passed 0 failed)

FINAL VERIFICATION (run by the controller, not a subagent):
- suite: 1118 passed, 0 failed
- real turn: `agent.py -p "what is 17 * 23?"` -> "17 x 23 = 391 [1]" citing calculate(), NO incident
- /confidence: renders on / qwen3.8:27b / enter 0.2229 / exit 0.4247 / "shipped calibration (estimated)"
  + the estimate disclosure naming its basis. Exactly the owner's requested scope.
- /models tier: six size classes with params + ctx/max + calibrated, active starred, no nicknames
- utilities/continuation_contract.py --models qwen3.5:4b qwen3.8:27b -> all green (freeze/continue works
  on both the fresh-install default and the flagship tier)
PLAN COMPLETE.

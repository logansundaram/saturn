# RESTRUCTURE.md

A plan to reduce individual file size and make the codebase easier to navigate, **without
changing behaviour**. This documents the code as it exists today (`refactor` branch) and proposes
a phased, low-risk path to smaller, single-responsibility modules.

> Companion to `CLAUDE.md` (architecture) and `NOTES.md` (build log). Where this plan and the code
> disagree, the code wins — update this file as phases land.

---

## 1. Guiding principles

The repo already has a good instinct: **atomic files behind a stable import surface** (one node per
`node_registry/` file, one tool per `tool_registry/` file, one command per `commands/` file). The
restructure extends that same pattern to the two files that outgrew it. Rules for every change here:

1. **Behaviour-preserving.** This is a *move*, not a rewrite. No logic changes, no API changes. Each
   phase must leave `python agent.py`, the test suite, and `benchmark.py` working identically.
2. **Stable import surface.** Callers must not change. A big file becomes a *package* whose
   `__init__.py` re-exports the same public names, so `from tui import ui; ui.foo()` keeps working.
3. **One responsibility per file.** Split along the seams the code already has (the `# ── section ──`
   comment banners in `ui.py` are literally the cut lines).
4. **Shared mutable state lives in one place.** Module-level globals that several functions mutate
   via `global` (e.g. `_t_last`, `_status`, `_live`) move into a single private state module that the
   split files import — never duplicated.
5. **Follow the existing conventions** (`CLAUDE.md` → Design rules): implicit namespace packages (no
   `__init__.py`) for the registries; an explicit `__init__.py` only where we need a re-export
   surface. Diagnostics through `diag.log()`, never `print()`. Prompts stay in `messages.py`.

---

## 2. Current state

### File sizes (Python, excluding `.git`/`__pycache__`/`cache`)

| Lines | File | Verdict |
|------:|------|---------|
| **2002** | `tui/ui.py` | **Priority 1 — split into a package** |
| 533 | `agent.py` | **Priority 2 — extract the loop + history helpers** |
| 393 | `benchmark.py` | OK (standalone harness); optional suite extraction |
| 375 | `commands/trace.py` | Borderline; optional split (it's the observability hub) |
| 351 | `tests/test_core.py` | OK (split only if test domains grow) |
| 284 | `llms.py` | OK |
| 261 | `messages.py` | OK (intentionally one place for prompts) |
| 252 | `config.py` | OK |
| 233–242 | `stores/{trace,rag}.py` | OK |
| 224 | `typeahead.py` | OK |
| ≤212 | everything else | OK — already atomic |

**One file dominates.** `tui/ui.py` is 4× the next-largest and ~24% of the non-test Python in the
repo. Fixing it is ~80% of the value of this whole effort.

### What is already well-structured (do not touch)

- `commands/` — already a package of one-command-per-file behind `@command(...)` + a thin dispatcher.
  (Note: `CLAUDE.md` still describes a single `commands.py`; the code has already moved past that. The
  commit history confirms — `d797303 created commands folder`, `9d266f4 streamlining command interface`.)
- `tool_registry/` and `node_registry/` — atomic, metadata-on-the-tool/node, no central edit needed
  to add one.
- `stores/` (data) and `tui/` (presentation) — clean layer separation already in place.

---

## 3. Priority 1 — split `tui/ui.py` into `tui/ui/` package (highest value)

**Why it's safe:** every consumer imports it the same way —

```
agent.py, commands/{models,context,clear,plan,system,trace}.py, tests/test_core.py:
    from tui import ui
    ... ui.banner(...), ui.response(...), ui.ask_approval(...), ...
```

Nobody does `from tui.ui import banner`. So we can replace the *module* `tui/ui.py` with a *package*
`tui/ui/` whose `__init__.py` re-exports the public names. **Zero call sites change.**

### 3.1 Target layout

`tui/ui.py` → `tui/ui/` with these submodules (line estimates from the current function map):

| New file | Owns | Public API it backs | ~Lines |
|----------|------|---------------------|-------:|
| `_state.py` | Console handles (`_console`, `_RICH`, `_PTK`), the palette constants, the **shared mutable globals** (`_t_last`, `_trace_started`, `_turn_start`, `_status`, `_live`, `_plan_seen`, `_model`, `_metrics*`, `_VERBOSITY`, `_input_state`), and the low-level primitives (`_emit`, `_rail`, `_truncate`, `_fmt_dur`, `_fmt_args`, `_term_width`, `_human_tokens`, `_meter_color`, `_mini_bar`, `_active_*`, `_human_int`). The foundation every other submodule imports. | (internal) | ~200 |
| `statusbar.py` | `_StatusBar`, `_append_meter`, `_live_start/stop/refresh`, `_metrics_loop/_metrics_start`, `set_input_preview`, `reset_turn`, `set_verbosity`, `verbosity` | `set_input_preview`, `reset_turn`, `set_verbosity`, `verbosity` | ~200 |
| `art.py` | All Saturn art + animation: `_norm3`, `_sphere_cell`, `_ring_path`, `_saturn_grid/_cells/_anim_cells`, `_grid_text`, `_InlinePlayer`, `_saturn_text/_plain`, `splash`, `play_animation`. Fully self-contained block. | `splash`, `play_animation` | ~330 |
| `prompt.py` | `prompt`, the `_ptk_*` keybindings, `_mention_fragments`, `_slash_token`, `ask`, `banner` | `prompt`, `ask`, `banner` | ~280 |
| `trace.py` | Live trace + run replay: `show_node`, `_node_line`, `_metric_parts`, `_emit_result_leaf`, `_render_tool_events`, `_msg_kind_content`, `_emit_message_leaf`, `_render_trace_messages`, `_enrich_results`, `show_run`, `_llm_leaf`, `show_llm_calls` | `show_node`, `show_run`, `show_llm_calls` | ~340 |
| `plan.py` | Plan render + review interrupt: `_plan_line`, `render_plan`, `show_plan`, `_review_*`, `_render_review_plan`, `review_plan` | `render_plan`, `show_plan`, `review_plan` | ~230 |
| `approval.py` | The gate + diff preview: `_workspace_old_text`, `_diff_lines`, `_render_write_diff`, `_render_shell_command`, `ask_approval` | `ask_approval` | ~170 |
| `response.py` | `_turn_summary_parts`, `response`, `ResponseStream` | `response`, `ResponseStream` | ~110 |
| `readouts.py` | `show_system_metrics`, `show_context`, `show_models`, `note`, `warn`, `steer_note`, `echo_queued` | same names | ~160 |
| `__init__.py` | Re-export the public API (see below). No logic. | — | ~40 |

Result: the largest TUI file drops from 2002 → ~340 lines, and each file maps to one screen concern.

### 3.2 The `__init__.py` re-export surface

`__init__.py` becomes the single declared public API of the TUI — explicit, greppable, and exactly
the set of names the rest of the app already calls:

```python
# tui/ui/__init__.py
from .statusbar import set_input_preview, reset_turn, set_verbosity, verbosity
from .art       import splash, play_animation
from .prompt    import prompt, ask, banner
from .trace     import show_node, show_run, show_llm_calls
from .plan      import render_plan, show_plan, review_plan
from .approval  import ask_approval
from .response  import response, ResponseStream
from .readouts  import show_system_metrics, show_context, show_models, note, warn, steer_note, echo_queued

__all__ = [...]  # the names above
```

**Verify the surface first** (do this before moving anything):

```powershell
# every ui.<name> referenced outside tui/  → that's the exact set __init__ must export
Select-String -Path *.py, commands\*.py, stores\*.py, node_registry\*.py, tests\*.py `
  -Pattern 'ui\.[A-Za-z_]+' -AllMatches |
  ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
```

If any name is missing from `__init__`, the import will `AttributeError` at first use — the test
suite and a smoke run catch it immediately.

### 3.3 Handling the shared globals (the only real hazard)

Several functions mutate module globals via `global` (`_t_last`, `_trace_started`, `_turn_start`,
`_status`, `_live`, `_plan_seen`, `_model`, `_metrics`, `_metrics_thread`, `_VERBOSITY`,
`_input_state`). Across files, `from _state import _t_last` would **copy the binding** and writes
wouldn't propagate. Two valid patterns — pick one and use it consistently:

- **Preferred:** keep the state in `_state.py` and have writers mutate *through the module*:
  `from . import _state` then `_state._t_last = ...`. Reads/writes always go to the one object.
- **Alternative:** wrap the mutable scalars in a small `_State` dataclass/`SimpleNamespace`
  instance (`state = _State()`); mutate `state.t_last`. Attribute access dodges the rebind trap
  entirely and reads cleaner. Slightly more churn at each touch point.

Dicts already mutated in place (`_status`, `_input_state`) are safe to import by reference; only the
**rebound scalars** need the through-module treatment.

### 3.4 Execution order (one submodule per commit, tests green each time)

1. Create `tui/ui/` and move the file to `tui/ui/__init__.py` verbatim. Run tests — proves the
   package conversion alone is transparent.
2. Extract `_state.py` (foundation), update `__init__` to import from it. Test.
3. Extract `art.py` (most self-contained, lowest risk — good warm-up). Test.
4. Extract `statusbar.py`, `response.py`, `readouts.py` (leaf-y, few cross-deps). Test after each.
5. Extract `trace.py`, `plan.py`, `approval.py`, `prompt.py` (more interdependent). Test after each.
6. Trim `__init__.py` to only re-exports. Final test + smoke run.

---

## 4. Priority 2 — split `agent.py` (533 lines)

`agent.py` mixes three concerns: **graph assembly**, the **turn/REPL machinery**, and **history
compaction helpers**. `build_agent` is the only thing other modules import from it as a library
(`benchmark.py` calls `build_agent`/`run_turn`); the rest is the entry point. Keep `agent.py` as the
launchable entry point but thin it:

| New file | Moves out of `agent.py` | ~Lines |
|----------|-------------------------|-------:|
| `graph.py` | `build_agent` (graph assembly) — referenced by `CLAUDE.md` as the canonical "graph assembly only in agent.py", so update that doc line when this moves | ~65 |
| `history.py` | `_compact_history`, `_maybe_autocompact`, `_human_int`, `_fresh_turn`, `_initial_state` (the per-turn state shaping + compaction glue) | ~150 |
| `turn.py` | `run_turn`, `_make_on_update` (stream the graph, drive both interrupt gates, feed trace+UI) | ~110 |
| `agent.py` | `main()` + arg parsing + the interactive loop; imports the above | ~180 |

Re-export from `agent.py` (`from .graph import build_agent` etc.) so `benchmark.py` and tests keep
importing from `agent` unchanged — same stable-surface trick as the TUI.

**Caveat:** `agent.py` currently reconfigures `stdout` to UTF-8 at import (relied on by `tui/ui.py`'s
fallback path — see `ui.py` header). Whatever module does that reconfiguration must still run on the
`python agent.py` path. Keep it in `agent.py`'s top-level (entry point), not in an extracted module.

This is lower value than Priority 1 (533 lines is navigable) — do it only after the TUI split proves
the pattern, or skip if time-boxed.

---

## 5. Priority 3 — optional / lower-value

- **`commands/trace.py` (375)** — it's the observability hub (`/trace`, `/trace invoke|calls|cost|
  state`, verbosity). Could split into `commands/trace.py` (dispatch + `/trace` run view) and a
  `commands/_trace_views.py` helper holding `_calls`, `_cost`, `_state`, `_show_llm_calls`. Only worth
  it if it keeps growing; 375 is not urgent.
- **`benchmark.py` (393)** — the `SUITES` / `CONVERSATIONS` data tables could move to
  `benchmarks/suites.py`, leaving the harness (`run_query`, `run_suites`, `run_conversation`, `main`)
  in `benchmark.py`. Pure data/logic separation; optional.
- **Root flat layout (~15 modules).** The repo root holds `agent.py`, `state.py`, `llms.py`,
  `config.py`, `messages.py`, `registry.py`, `toolspec.py`, `diag.py`, `interrupts.py`,
  `typeahead.py`, `mentions.py`, `compaction.py`, `env_keys.py`. These are individually small and
  fine. A `core/` (graph/state/llms/messages) + `infra/` (config/diag/interrupts) grouping is
  *possible* but touches imports repo-wide for low payoff and conflicts with the "launches as
  `python agent.py` from repo root, root stays on `sys.path`" rule. **Recommend NOT doing this**
  unless the root genuinely becomes hard to scan. If pursued, do it last and as its own effort.

---

## 6. Tie-in: deferred code-review items

`CLAUDE.md` → "Deferred from code review" lists duplication clusters that a restructure is the
natural moment to fold (do these *opportunistically* while a file is already open — but keep them in
**separate commits** from the pure moves so each move stays a verifiable no-op):

- `mentions._read_clamped` vs `tools._clamp_observation` — one shared clamp helper.
- The `…`-truncation helper duplicated ×4 (with ASCII/Unicode drift) — unify (the TUI copy lands in
  `tui/ui/_state.py` as `_truncate`; point the others at it or a shared util).
- The `json.loads(data or "{}")` trace-delta decode (×3) and the `sqlite3.connect/finally close`
  block (×3 in commands) — extract helpers.
- `current_response` — dead, mistyped `AgentState` field; delete during the `state.py`/agent pass.

Do **not** bundle the trickier deferred items (positional-multiset triplication #2, steering-after-
tool-round #1, transactional `rag.sync` #4) into this restructure — they carry behaviour risk and
are explicitly "left unfixed on purpose."

---

## 7. Validation (run after every phase)

```powershell
python -m pytest tests/ -q                 # the regression net
python agent.py -p "what is 2+2"           # headless smoke: graph + tools + synth path
python agent.py                            # interactive smoke: banner, prompt, a turn, /trace, /quit
python benchmark.py --no-conversations     # exercises run_turn across suites
```

Plus a quick "no broken imports" sweep:

```powershell
Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '__pycache__' } |
  ForEach-Object { python -c "import ast; ast.parse(open('$($_.FullName)', encoding='utf-8').read())" }
```

**Acceptance per phase:** tests pass, both smoke runs render identically to `main`, no file the phase
touched exceeds ~350 lines, and no call site outside the split package changed.

---

## 8. Summary / recommended order

1. **Phase 1 (do first, highest value):** `tui/ui.py` → `tui/ui/` package (§3). Drops the worst file
   2002 → ~340 and proves the re-export pattern.
2. **Phase 2:** `agent.py` → `graph.py` + `turn.py` + `history.py`, thin entry point (§4).
3. **Phase 3 (optional):** `commands/trace.py`, `benchmark.py` data extraction (§5).
4. **Throughout:** fold the safe duplication clusters in their own commits (§6); update `CLAUDE.md`'s
   stale "single `commands.py`" and "graph assembly in `agent.py`" lines as they move.
5. **Skip unless it earns it:** the root-package reorg (§5) — high blast radius, low payoff.

After Phase 1+2 no file exceeds ~350 lines, every file maps to one concern, and not a single import
outside the split packages had to change.

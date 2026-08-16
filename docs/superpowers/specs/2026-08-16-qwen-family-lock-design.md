# Qwen3.5–3.8 family lock + always-on confidence coloring

**Date:** 2026-08-16
**Status:** approved design, ready for implementation planning
**Branch:** `transplant/isolates-2026-08`

## Goal

Saturday.ai supports exactly one model family — **qwen3.5 / qwen3.6 / qwen3.8** — with the most
advanced tag at each parameter size. Confidence coloring becomes part of the default experience,
with a dedicated `/confidence` command owning its on/off switch, re-calibration, and manual
threshold entry.

The driver is calibration honesty. Confidence coloring only means "worse than 95 % of this model's
clean output" when the per-model calibration table (`core/confidence_calibration.py`) actually
covers the bound model. Today the table covers `qwen3.6:27b` and `gemma4:e4b` and
`qwen3-coder:30b`, while `qwen3.8:27b` is an **inherited entry with 0 tokens measured**. Locking
the bindable set to one family, and calibrating every shipped tag, makes the marking trustworthy
by construction rather than by luck.

## Decisions

| Question | Decision | Rationale |
|---|---|---|
| What defines the allowed set | Family prefix gate, **and** calibrate every shipped tag now | Simple predicate; the calibration pass closes the gap the gate alone would leave |
| Tier identity | Size class (`0.8b`, `2b`, `4b`, `9b`, `27b`, `35b`), metrics on display, no nicknames | A version bump is a one-line binding swap; `active_tier` never churns |
| Tuned thresholds | Per-model overlay in `database/confidence_calibration.json` | Survives `/update`, works in wheel installs, precedent is `permissions.json` |
| Non-family binding on launch | **Auto-migrate in memory, warn loudly, never rewrite config.yaml** | Owner's call; nothing hard-fails. Accepted trade-off recorded under Risks |
| Confidence on/off | `runtime.confidence` stays, defaults true, fronted by `/confidence` | "Always on" = on by default with a discoverable off switch, not a deleted knob |
| Embedder | Exempt from the gate | `qwen3-embedding:8b` is not a chat model: no chat template, no logprobs, no confidence |

## Non-goals

- Vision. Every family tag advertises a `vision` capability; Saturn's formal scope cut stands and
  `supports_vision` stays `false` with a comment saying the model can and Saturn does not.
- Reintroducing cloud providers. The shelve (2026-07-03) is untouched; this narrows the local set.
- Changing how confidence is *computed* or *rendered*. `low_runs`, hysteresis, the stoplist, and
  every render site are unchanged. This work changes which models may bind and where thresholds
  come from.

## §1 The family gate

New leaf **`core/model_family.py`** — stdlib only, no project imports, so `config`, `llms`, and
`commands/` may all depend on it:

```python
FAMILY_PREFIXES = ("qwen3.5", "qwen3.6", "qwen3.8")

SIZE_LADDER = (
    ("0.8b", "qwen3.5:0.8B"),   # capital B — the Ollama tag is case-sensitive
    ("2b",   "qwen3.5:2b"),
    ("4b",   "qwen3.5:4b"),
    ("9b",   "qwen3.5:9b"),
    ("27b",  "qwen3.8:27b"),
    ("35b",  "qwen3.6:35b"),
)

def in_family(model_id: str) -> bool: ...   # anchored ^(qwen3\.5|qwen3\.6|qwen3\.8)(:|$)
def migrate(model_id: str) -> str: ...      # -> size-class KEY, §2
def tag_for(size_class: str) -> str: ...    # class key -> the ladder's model id
```

`migrate` returns a class key rather than a tag so the migration seam and `/models tier` share one
vocabulary; `config.model_for_role` composes them (`tag_for(migrate(id))`) to get the id it
returns.

Matching is **anchored**, not a loose `startswith`: a hypothetical `qwen3.50` must not satisfy a
`qwen3.5` prefix test. Case-insensitive on the comparison, but the ladder's tags are stored
verbatim so `qwen3.5:0.8B` is pulled and bound with its real capitalization.

`core/chat_template.py` carries the same three prefixes for raw-mode continuation. A test asserts
the two lists agree, so extending one without the other fails offline. The `gemma4` template entry
becomes unreachable once the gate is in place and is **deleted** — after which
`chat_template.supported()` is exactly "in family", and the freeze hotkey arms for every bindable
model.

### Enforcement points

1. `commands/runtime._bind` and `_models_picker` — refuse a non-family id, print the ladder.
2. `config.model_for_role` — the auto-migration seam (§2).
3. `llms.check_models` — reports any migration in the startup health pass.

## §2 Auto-migration

Migration lives in **`config.model_for_role`**, the one place every role binding resolves. Putting
it deeper (in `llms.get_model`) would leave `/models` readouts, `check_models`, and
`confidence._synthesizer_model()` reporting the stale tag — and the calibration lookup would then
miss, which is the exact failure this work exists to prevent.

Resolution order for a non-family id:

1. **Exact legacy table** — `gemma4:e2b`→`2b`, `gemma4:e4b`→`4b`, `gemma4:12b`→`9b`,
   `gemma4:26b`→`27b`, `gemma4:31b`→`27b`, `qwen3-coder:30b`→`27b`
2. **Parameter parse off the tag** — `:(\d+(\.\d+)?)b` mapped to the nearest ladder class
3. **Unknown** → the fresh-install default class (`4b`), warned

Properties:

- **In-memory only.** `config.yaml` is never rewritten. Rebinding is the permanent fix.
- **Warned once per distinct substitution**, not per call — the result is cached.
- The warning names old → new and points at `/models tier <key>`.
- `/models` shows a note while any binding is migrated, so the readout never claims the file's
  value is what is running.

## §3 Tiers and capabilities

Six tiers keyed by size class, all five roles on one model, embedder unchanged
(`qwen3-embedding:8b`). `bench-coder` is deleted; `laptop` and `workstation` are replaced.

| key | model | params | disk |
|---|---|---|---|
| `0.8b` | `qwen3.5:0.8B` | 873M | 1.0 GB |
| `2b` | `qwen3.5:2b` | 2.3B | 2.7 GB |
| `4b` | `qwen3.5:4b` | 4.7B | 3.4 GB — **fresh-install default** |
| `9b` | `qwen3.5:9b` | 9.7B | 6.6 GB |
| `27b` | `qwen3.8:27b` | 27.3B | 17 GB — this machine's `config.yaml` |
| `35b` | `qwen3.6:35b` | 36.0B | 23 GB |

The fresh-install default moves from `gemma4:e4b` (9.6 GB) to `qwen3.5:4b` (3.4 GB), so the first
pull gets lighter, not heavier. `install.sh` / `install.ps1` `SATURDAY_MODELS` defaults change to
`qwen3.5:4b qwen3-embedding:8b`.

### The 256k context hazard

`ollama show` reports **`context length 262144`** for every family tag. `runtime.num_ctx: null`
means "use `capabilities.<model>.context_window`", and that value is passed straight to
ChatOllama — so writing the true maximum into that field would request a 256k window on every
call and exhaust VRAM on any consumer card. The current entries say `32768`, which is why the app
works today.

Therefore the field is **split**:

- `context_window: 32768` — the runtime default, unchanged in meaning, still what `num_ctx_for`
  returns
- `max_context_window: 262144` — display only, feeding the metrics table

`config.Capability` gains `max_context_window` (defaulting to `context_window` when absent).
`num_ctx_for` is untouched.

### Metrics display

`/models list` and `/models tier` render params · ctx default / max · disk · calibrated?, replacing
nicknames. `tui/ui/readouts.show_models` gains the columns; `LocalModel` already carries
`parameter_size`, `quantization`, and `size_bytes`.

## §4 `/confidence`

New module **`commands/confidence.py`**, added to `commands/__init__._COMMAND_MODULES` and to
`commands/system.py::_GROUPS` (`tests/test_help.py` enforces the latter against the live registry).

Grammar follows house conventions — `parse_toggle_status` for the toggle (bare is never a flip),
`split_persist_flags` for the settings (persist by default, `--session` opts out):

```
/confidence                      status readout
/confidence on | off             runtime.confidence, persists by default
/confidence tune [--prompts N]   re-calibrate the ACTIVE synthesizer model
/confidence set <enter> [exit]   your own values for the active model
/confidence reset                drop your values, fall back to the shipped table
```

**Status readout** shows: on/off · synthesizer model · enter/exit thresholds · their source
(`tuned` / `shipped` / `pinned` / `default`) · when measured and over how many tokens.

**`tune`** runs the calibrator in-process against the live daemon, through `llms._build`'s trust
boundary so airgap and the egress ledger still apply. Emits progress lines; writes the overlay on
success. Refuses with a clear message when the daemon is unreachable.

**`set`** validates `0 < enter < 1`, `0 < exit < 1`, and `exit > enter` (an exit below the enter
threshold would close runs it was meant to extend), then writes the overlay for the active model.
With `exit` omitted it is derived by the existing rule in `core/confidence` — `1.5 ×` enter, capped
at `0.95` — so a one-argument `set` behaves exactly like an uncalibrated model's fallback.

**`reset`** removes the active model's overlay entry only.

## §5 The overlay store

New leaf **`core/confidence_store.py`** (imports `config` and `json` only) owns both read and write
of `database/confidence_calibration.json` — one owner, one parser, matching the house rule.
`core/confidence.calibration_for()` consults it; `commands/confidence.py` writes through it.

```json
{
  "qwen3.8:27b": {
    "enter": 0.2229, "exit": 0.4247,
    "tokens": 1279, "prompts": 55,
    "at": "2026-08-16", "source": "tuned"
  }
}
```

Resolution order in `confidence.threshold()` / `exit_threshold()`:

1. explicit numeric `runtime.confidence_threshold` — the global pin, still wins over everything
2. user overlay JSON for the synthesizer model
3. shipped `core/confidence_calibration.py`
4. built-in `0.20` default

Read is cached with an mtime check. A garbled file is reported and ignored rather than raised —
a confidence failure must never cost the answer, which is the module's existing contract.

`runtime.confidence` remains the on/off backing, defaulting `true`.

## §6 Calibration pass

Sequenced **last**, after the code lands, so the table reflects the shipped set: run
`utilities/confidence_calibrate.py` across all six ladder tags. This also replaces the current
`qwen3.8:27b` entry (`inherited_from: qwen3.6:27b`, `tokens: 0` — never measured) with real
measurements, and drops the `gemma4:e4b` / `qwen3-coder:30b` rows, which become unbindable.

Live-daemon work over 6 models × 55 prompts — the long pole. Everything else is offline-testable.

## Files touched

**New:** `core/model_family.py`, `core/confidence_store.py`, `core/calibration.py`,
`commands/confidence.py`, `tests/test_model_family.py`, `tests/test_confidence_command.py`,
`tests/test_confidence_store.py`, `tests/test_calibration.py`, `tests/test_models_command.py`

`core/calibration.py` is a planning-time discovery, not a design change: `utilities/` is excluded
from the wheel (`pyproject.toml` `packages.find` covers `core*` and not `utilities`), so
`/confidence tune` could not import the measurement in an installed run. The measurement moves
into `core/`; `utilities/confidence_calibrate.py` becomes a thin CLI over it. One measurement, two
callers — the CLI writes the shipped baseline, the command writes the user overlay.

**Changed:** `config.py` (migration seam, `Capability.max_context_window`), `config.default.yaml`
and `config.yaml` (tiers, capabilities), `core/llms.py` (`check_models` reporting),
`core/confidence.py` (overlay consultation), `core/chat_template.py` (drop `gemma4`),
`core/confidence_calibration.py` (regenerated), `commands/runtime.py` (bind refusal, metrics),
`commands/__init__.py`, `commands/system.py` (`_GROUPS`), `tui/ui/readouts.py` (metrics columns),
`install.sh`, `install.ps1`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `documentation.md`

**Tests updated:** `test_onboarding.py`, `test_core.py`, `test_ollama_locality.py`,
`test_token_steering.py`, `test_help.py` (all reference retired model ids)

## Testing

Offline pytest covers: the anchored family predicate and its near-miss cases, the migration table
and its parameter-parse fallback, warn-once caching, the overlay resolution order including a
garbled file, `/confidence` grammar (bare-is-status, persist-by-default, `set` validation), the
`chat_template` ↔ `model_family` prefix agreement, and `/help` grouping.

Live gates, run once by hand: `utilities/continuation_contract.py --all` across the ladder, and the
§6 calibration pass.

## Risks

1. **Running config diverges from config.yaml** (§2, accepted). A migrated binding runs a model the
   file does not name. Mitigated by the loud named warning, the `/models` note, and never
   rewriting the file — but a user who ignores the warning is running something they did not
   configure. Refusing at launch was offered and declined.
2. **`context_window` vs `max_context_window`** (§3). Collapsing these back into one field is a
   latent OOM for every machine under ~48 GB VRAM. The split must survive review.
3. **Small tags may not hold the engine.** `qwen3.5:0.8B` and `:2b` are offered as tiers but the
   plan/execute engine leans on reliable structured output and native tool-calling. The
   hardened parse layer exists for exactly this, but the small tiers should be treated as
   unvalidated until someone runs the trust benchmark on them.
4. **Migration fallback for unknown ids** picks `4b` on size-parse failure, which silently
   downgrades a user who had a large non-family model bound. The warning names the substitution.

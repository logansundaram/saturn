# Qwen3.5–3.8 Family Lock + Always-On Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict Saturday.ai's bindable chat models to the qwen3.5/3.6/3.8 family keyed by size class, and make confidence coloring a default-on experience fronted by a `/confidence` command with per-model tuning.

**Architecture:** A stdlib-only leaf (`core/model_family.py`) owns the family predicate and the size ladder. `config.model_for_role` is the single migration seam — a non-family binding is substituted in memory and recorded, never written back to disk. Confidence thresholds gain a user-writable overlay (`database/confidence_calibration.json`) that wins over the shipped calibration table, and the measurement code moves from the unshipped `utilities/` into `core/calibration.py` so the command can re-tune in installed mode.

**Tech Stack:** Python 3, PyYAML, pytest, LangChain + `langchain_ollama`, Rich (TUI), Ollama daemon.

**Spec:** `docs/superpowers/specs/2026-08-16-qwen-family-lock-design.md`

## Global Constraints

- Python, run from the repo root. Tests: `python -m pytest tests/`. Every test runs **offline** — no LLM, no network, no embedder.
- `core/model_family.py` and `core/confidence_store.py` are **leaves**: `model_family` imports stdlib only; `confidence_store` imports stdlib plus `config`. Nothing else.
- `config.py` must keep its "nothing here imports from the rest of the project" property intact in spirit — `core.model_family` is admissible **only** because it is stdlib-only. Note this in the import comment.
- The family is exactly `("qwen3.5", "qwen3.6", "qwen3.8")`, matched **anchored** (`^(?:qwen3\.5|qwen3\.6|qwen3\.8)(?::|$)`), case-insensitive.
- The size ladder is exactly: `0.8b`→`qwen3.5:0.8B`, `2b`→`qwen3.5:2b`, `4b`→`qwen3.5:4b`, `9b`→`qwen3.5:9b`, `27b`→`qwen3.8:27b`, `35b`→`qwen3.6:35b`. **`qwen3.5:0.8B` has a capital B** — Ollama tags are case-sensitive; store it verbatim.
- Default size class (fresh install and unknown-migration fallback): `4b`.
- `capabilities.<model>.context_window` stays **32768** (the runtime `num_ctx`). The true maximum goes in a new **display-only** `max_context_window: 262144`. Never collapse these — a 256k `num_ctx` exhausts VRAM on any consumer card.
- The embedder (`qwen3-embedding:8b`) is **exempt** from the family gate.
- Migration is **in-memory only**. `config.yaml` is never rewritten by the migration path.
- `runtime.confidence` stays as the on/off backing and defaults **true**.
- `config.yaml` is untracked user data — edit it, but never `git add` it. Only `config.default.yaml` is committed.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 1: The family gate leaf

**Files:**
- Create: `core/model_family.py`
- Test: `tests/test_model_family.py`

**Interfaces:**
- Consumes: nothing (stdlib only)
- Produces: `FAMILY_PREFIXES: tuple[str, ...]`, `SIZE_LADDER: tuple[tuple[str, str], ...]`, `DEFAULT_CLASS: str`, `in_family(model_id) -> bool`, `classes() -> tuple[str, ...]`, `tag_for(size_class: str) -> str` (raises `KeyError`), `migrate(model_id) -> str` (returns a size-class **key**, never a tag)

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_family.py`:

```python
"""The supported-family gate: the predicate, the size ladder, and the migration map."""

import pytest

from core import model_family as mf


class TestInFamily:
    def test_every_ladder_tag_is_in_family(self):
        for _key, tag in mf.SIZE_LADDER:
            assert mf.in_family(tag), tag

    def test_bare_family_name_matches(self):
        assert mf.in_family("qwen3.6")

    def test_case_insensitive(self):
        assert mf.in_family("QWEN3.5:0.8B")

    def test_matching_is_anchored_not_a_loose_prefix(self):
        # qwen3.50 must NOT satisfy a qwen3.5 test — the whole point of anchoring.
        assert not mf.in_family("qwen3.50:1b")

    @pytest.mark.parametrize(
        "tag",
        ["gemma4:e4b", "qwen3-coder:30b", "qwen3-embedding:8b", "qwen3.7:9b", "", "   "],
    )
    def test_outsiders_rejected(self, tag):
        assert not mf.in_family(tag)

    def test_none_is_not_in_family(self):
        assert not mf.in_family(None)


class TestLadder:
    def test_classes_match_the_ladder_order(self):
        assert mf.classes() == ("0.8b", "2b", "4b", "9b", "27b", "35b")

    def test_tag_for_round_trips(self):
        for key, tag in mf.SIZE_LADDER:
            assert mf.tag_for(key) == tag

    def test_tag_for_is_case_insensitive(self):
        assert mf.tag_for("0.8B") == "qwen3.5:0.8B"

    def test_tag_for_preserves_the_capital_b_tag(self):
        # Ollama tags are case-sensitive: the 0.8B tag must survive verbatim.
        assert mf.tag_for("0.8b") == "qwen3.5:0.8B"

    def test_tag_for_unknown_class_raises(self):
        with pytest.raises(KeyError):
            mf.tag_for("13b")

    def test_default_class_is_on_the_ladder(self):
        assert mf.DEFAULT_CLASS in mf.classes()


class TestMigrate:
    @pytest.mark.parametrize(
        "old,expected",
        [
            ("gemma4:e2b", "2b"),
            ("gemma4:e4b", "4b"),
            ("gemma4:12b", "9b"),
            ("gemma4:26b", "27b"),
            ("gemma4:31b", "27b"),
            ("qwen3-coder:30b", "27b"),
        ],
    )
    def test_legacy_table_is_exact(self, old, expected):
        assert mf.migrate(old) == expected

    def test_legacy_lookup_is_case_insensitive(self):
        assert mf.migrate("GEMMA4:E4B") == "4b"

    def test_unknown_tag_falls_back_to_the_size_parse(self):
        assert mf.migrate("mystery:33b") == "27b"
        assert mf.migrate("mystery:3b") == "2b"

    def test_unparseable_tag_falls_back_to_the_default_class(self):
        assert mf.migrate("devstral-small-2:latest") == mf.DEFAULT_CLASS

    def test_migrate_always_returns_a_real_class(self):
        for tag in ["gemma4:e4b", "mystery:33b", "junk", "", None]:
            assert mf.migrate(tag) in mf.classes()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.model_family'`

- [ ] **Step 3: Write the implementation**

Create `core/model_family.py`:

```python
"""
The supported model family (2026-08-16).

Saturday.ai binds exactly ONE model family — qwen3.5 / qwen3.6 / qwen3.8 — because confidence
coloring is calibrated PER MODEL (core/confidence_calibration.py): a red run means "worse than
95 % of THIS model's clean output", which is only a true claim for a model that was actually
measured. Every other family was removed rather than shipped with a threshold borrowed from a
model of a different size (the confidence_coloring isolate measured a 6-10x spread across sizes).

The ladder carries the most advanced tag at each parameter size, so a version bump is a one-line
edit here and in config.default.yaml — the tier keys (size classes) never churn.

LEAF: stdlib only, no project imports, so config.py / core/llms.py / commands/ may all depend on
it without a cycle.
"""

from __future__ import annotations

import re

# The family, as bare prefixes. core/chat_template.py carries the same three for raw-mode
# continuation; tests/test_model_family.py asserts the two lists agree.
FAMILY_PREFIXES: tuple[str, ...] = ("qwen3.5", "qwen3.6", "qwen3.8")

# ANCHORED on purpose: a loose startswith() would let a future `qwen3.50` satisfy a `qwen3.5`
# test and bind an uncalibrated model through the gate.
_FAMILY_RE = re.compile(r"^(?:qwen3\.5|qwen3\.6|qwen3\.8)(?::|$)", re.IGNORECASE)

# size class -> the most advanced family tag at that size. Tags are stored VERBATIM: Ollama tags
# are case-sensitive and the 0.8B tag really does carry a capital B.
SIZE_LADDER: tuple[tuple[str, str], ...] = (
    ("0.8b", "qwen3.5:0.8B"),
    ("2b", "qwen3.5:2b"),
    ("4b", "qwen3.5:4b"),
    ("9b", "qwen3.5:9b"),
    ("27b", "qwen3.8:27b"),
    ("35b", "qwen3.6:35b"),
)

# The fresh-install tier, and where an unrecognizable legacy binding lands. Small on purpose:
# the first pull should be light, and migrating DOWN never exceeds the machine's VRAM.
DEFAULT_CLASS = "4b"

# Real parameter counts (billions, from `ollama show`) — the nearest-size fallback's yardstick.
_CLASS_PARAMS: dict[str, float] = {
    "0.8b": 0.87, "2b": 2.3, "4b": 4.7, "9b": 9.7, "27b": 27.3, "35b": 36.0,
}

# Exact substitutions for the tags that shipped before the lock. The size parse below would
# reach the same answer for all of them; the table is here so the mapping is DECLARED rather
# than inferred, and so a rename upstream can't silently move somebody's binding.
_LEGACY: dict[str, str] = {
    "gemma4:e2b": "2b",
    "gemma4:e4b": "4b",
    "gemma4:12b": "9b",
    "gemma4:26b": "27b",
    "gemma4:31b": "27b",
    "qwen3-coder:30b": "27b",
}

# `:30b`, `:e4b`, `:0.8b` — the parameter count Ollama bakes into a tag.
_SIZE_RE = re.compile(r":e?(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def in_family(model_id) -> bool:
    """Whether `model_id` is a supported chat model. The EMBEDDER is exempt from this test —
    it is not a chat model, has no raw-mode template and produces no logprobs."""
    return bool(_FAMILY_RE.match(str(model_id or "").strip()))


def classes() -> tuple[str, ...]:
    """The size-class keys, smallest first — the tier names and what /models tier accepts."""
    return tuple(key for key, _tag in SIZE_LADDER)


def tag_for(size_class) -> str:
    """The model id a size class binds. Raises KeyError for an unknown class."""
    want = str(size_class or "").strip().lower()
    for key, tag in SIZE_LADDER:
        if key == want:
            return tag
    raise KeyError(
        f"unknown size class {size_class!r} — defined: {', '.join(classes())}"
    )


def migrate(model_id) -> str:
    """The size class a NON-family binding is substituted with. Deterministic: the declared
    legacy table first, then the parameter count parsed out of the tag mapped to the nearest
    class, then DEFAULT_CLASS. Returns a class KEY — callers compose tag_for(migrate(id))."""
    name = str(model_id or "").strip().lower()
    if name in _LEGACY:
        return _LEGACY[name]
    found = _SIZE_RE.search(name)
    if found:
        try:
            want = float(found.group(1))
        except ValueError:
            want = None
        if want is not None:
            return min(_CLASS_PARAMS, key=lambda key: abs(_CLASS_PARAMS[key] - want))
    return DEFAULT_CLASS
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add core/model_family.py tests/test_model_family.py
git commit -m "model_family: the supported-family gate and size ladder

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The migration seam in config

**Files:**
- Modify: `config.py` (the `Capability` dataclass ~line 109, `capability_of` ~line 194, `model_for_role` ~line 158, `reload`)
- Test: `tests/test_model_family.py` (append a class)

**Interfaces:**
- Consumes: `core.model_family.in_family`, `.migrate`, `.tag_for` (Task 1)
- Produces: `config.migrated_bindings() -> dict[str, str]` (original id → replacement id), `config.clear_migrations() -> None`, `Capability.max_context_window: int`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_family.py`:

```python
class TestConfigMigrationSeam:
    """config.model_for_role is THE migration seam: substitute in memory, record it, never
    rewrite config.yaml."""

    def _cfg(self, synth="qwen3.8:27b"):
        from config import Config

        return Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "planner": synth, "tool_caller": synth, "synthesizer": synth,
                "utility": synth, "judge": synth,
            }, "embedder": "qwen3-embedding:8b"}},
            "capabilities": {},
        })

    def setup_method(self):
        import config

        config.clear_migrations()

    def test_family_binding_passes_through_untouched(self):
        import config

        spec = self._cfg().model_for_role("synthesizer")
        assert spec.model == "qwen3.8:27b"
        assert config.migrated_bindings() == {}

    def test_non_family_binding_is_substituted(self):
        spec = self._cfg("gemma4:e4b").model_for_role("synthesizer")
        assert spec.model == "qwen3.5:4b"
        assert spec.provider == "ollama"

    def test_the_substitution_is_recorded(self):
        import config

        self._cfg("qwen3-coder:30b").model_for_role("synthesizer")
        assert config.migrated_bindings() == {"qwen3-coder:30b": "qwen3.8:27b"}

    def test_a_non_ollama_binding_is_left_for_the_cloud_shelve_refusal(self):
        import config
        from config import Config

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "synthesizer": {"provider": "anthropic", "model": "claude-sonnet-4"},
            }}},
        })
        spec = cfg.model_for_role("synthesizer")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-sonnet-4"
        assert config.migrated_bindings() == {}

    def test_the_embedder_is_exempt(self):
        cfg = self._cfg()
        assert cfg.embedder_model == "qwen3-embedding:8b"

    def test_capability_max_context_window_defaults_to_the_runtime_window(self):
        from config import Config

        cfg = Config({"capabilities": {"m": {"context_window": 32768}}})
        cap = cfg.capability_of("m")
        assert cap.context_window == 32768
        assert cap.max_context_window == 32768

    def test_capability_max_context_window_is_read_when_present(self):
        from config import Config

        cfg = Config({"capabilities": {"m": {"context_window": 32768,
                                             "max_context_window": 262144}}})
        cap = cfg.capability_of("m")
        assert cap.context_window == 32768        # what num_ctx_for returns — unchanged
        assert cap.max_context_window == 262144   # display only

    def test_num_ctx_for_still_returns_the_runtime_window_not_the_max(self):
        from config import Config

        cfg = Config({"runtime": {"num_ctx": None},
                      "capabilities": {"m": {"context_window": 32768,
                                             "max_context_window": 262144}}})
        assert cfg.num_ctx_for("m") == 32768
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_model_family.py::TestConfigMigrationSeam -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'clear_migrations'`

- [ ] **Step 3: Write the implementation**

In `config.py`, add to the imports near the top (after `import yaml`):

```python
from core import model_family  # stdlib-only leaf: importing it keeps config's no-cycle property
```

Add the migration ledger just below `RISK_ORDER`:

```python
# Non-family chat bindings substituted THIS SESSION: original id -> replacement id. Populated by
# model_for_role, read by llms.check_models and /models so no readout claims the file's value is
# what is running. In-memory only — config.yaml is NEVER rewritten by the migration path
# (rebinding with /models tier <class> is the permanent fix).
_MIGRATIONS: dict[str, str] = {}


def migrated_bindings() -> dict:
    """A copy of this session's family substitutions (original id -> replacement id)."""
    return dict(_MIGRATIONS)


def clear_migrations() -> None:
    """Forget the recorded substitutions (a config reload, or a test)."""
    _MIGRATIONS.clear()
```

Replace the `Capability` dataclass body:

```python
@dataclass(frozen=True)
class Capability:
    """What a model can do. The MVP requires tools + structured output for the roles that
    drive the loop; the factory warns when a bound model falls short."""

    supports_tools: bool = True
    supports_structured_output: bool = True
    context_window: int = 8192
    supports_vision: bool = False
    # The model's ARCHITECTURAL maximum — display only (the /models metrics columns). Kept
    # separate from context_window on purpose: context_window is what num_ctx_for hands
    # ChatOllama, and every qwen3.x tag reports a 262144 maximum that would exhaust VRAM on any
    # consumer card if it were requested per call. Never collapse these two fields.
    max_context_window: int = 0
```

In `capability_of`, replace the return:

```python
        cw = spec.get("context_window", 8192)
        return Capability(
            supports_tools=spec.get("supports_tools", True),
            supports_structured_output=spec.get("supports_structured_output", True),
            context_window=cw,
            supports_vision=spec.get("supports_vision", False),
            max_context_window=spec.get("max_context_window", cw),
        )
```

Add the enforcement helper immediately above `model_for_role`:

```python
    def _enforce_family(self, spec: "ModelSpec") -> "ModelSpec":
        """Substitute a non-family CHAT binding with the ladder tag for its nearest size class,
        and record the substitution. Saturday.ai supports one family (core/model_family) because
        confidence coloring is calibrated per model; a binding outside it would be marked against
        another model's numbers.

        A non-ollama binding is left alone — the cloud-model shelve (2026-07-03) owns that
        refusal, and quietly rewriting it would hide the real problem. The embedder never reaches
        here (embedder_model is its own accessor)."""
        if spec.provider != "ollama" or model_family.in_family(spec.model):
            return spec
        replacement = model_family.tag_for(model_family.migrate(spec.model))
        _MIGRATIONS[spec.model] = replacement
        return ModelSpec(provider=spec.provider, model=replacement)
```

In `model_for_role`, wrap both return statements:

```python
        if isinstance(entry, dict):
            model = entry.get("model")
            if not model:
                raise KeyError(
                    f"role '{role}' on tier '{self.active_tier}' is a mapping without a "
                    f"'model' key: {entry!r}"
                )
            return self._enforce_family(
                ModelSpec(provider=entry.get("provider", default_provider), model=model)
            )
        return self._enforce_family(
            ModelSpec(provider=default_provider, model=str(entry))
        )
```

In `reload()`, add `clear_migrations()` as the first statement of the function body (a re-read of disk may have fixed the binding, so the ledger must not outlive it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: PASS

Then the regression sweep — `capability_of` and `model_for_role` are load-bearing:

Run: `python -m pytest tests/ -q`
Expected: green, or failures **only** in `tests/test_token_steering.py` (the three gemma4 template assertions, fixed in Task 10). Record any other failure and stop — it means the migration seam changed behavior it should not have.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_model_family.py
git commit -m "config: family migration seam + display-only max_context_window

A non-family chat binding is substituted in memory with the ladder tag for
its nearest size class and recorded in migrated_bindings(); config.yaml is
never rewritten. max_context_window is display-only so num_ctx_for keeps
returning the 32768 runtime window.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Tiers, capabilities, and the installers

**Files:**
- Modify: `config.default.yaml` (the `tiers:` and `capabilities:` blocks)
- Modify: `config.yaml` (same blocks — **untracked, never `git add` it**)
- Modify: `install.sh:24`, `install.ps1:22`
- Test: `tests/test_model_family.py` (append a class)

**Interfaces:**
- Consumes: `core.model_family.SIZE_LADDER`, `.classes()` (Task 1); `config.Capability.max_context_window` (Task 2)
- Produces: six size-class tiers in the shipped template; every ladder tag has a capabilities entry

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_family.py`:

```python
class TestShippedConfigMatchesTheLadder:
    """The template config and the ladder must not drift apart — a tier binding a tag with no
    capabilities entry silently runs at the conservative 8192 default."""

    def _template(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / "config.default.yaml").read_text(encoding="utf-8"))

    def test_tier_keys_are_exactly_the_size_classes(self):
        from core import model_family as mf

        assert tuple(self._template()["tiers"]) == mf.classes()

    def test_every_tier_binds_its_ladder_tag_on_every_role(self):
        from config import MODEL_ROLES
        from core import model_family as mf

        tiers = self._template()["tiers"]
        for key, tag in mf.SIZE_LADDER:
            roles = tiers[key]["roles"]
            assert set(roles) == set(MODEL_ROLES), key
            assert set(roles.values()) == {tag}, key

    def test_every_role_binding_is_in_family(self):
        from core import model_family as mf

        for key, tier in self._template()["tiers"].items():
            for role, tag in tier["roles"].items():
                assert mf.in_family(tag), f"{key}.{role} = {tag}"

    def test_every_ladder_tag_has_a_capabilities_entry(self):
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert tag in caps, tag

    def test_capabilities_keep_the_runtime_window_off_the_architectural_max(self):
        # Collapsing these is a latent OOM: 262144 num_ctx exhausts consumer VRAM.
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert caps[tag]["context_window"] == 32768, tag
            assert caps[tag]["max_context_window"] == 262144, tag

    def test_retired_models_are_gone_from_the_template(self):
        template = self._template()
        text = str(template)
        for retired in ("gemma4", "qwen3-coder", "bench-coder"):
            assert retired not in text, retired

    def test_the_default_tier_is_the_default_class(self):
        from core import model_family as mf

        assert self._template()["active_tier"] == mf.DEFAULT_CLASS

    def test_the_embedder_is_unchanged_on_every_tier(self):
        for key, tier in self._template()["tiers"].items():
            assert tier["embedder"] == "qwen3-embedding:8b", key
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_model_family.py::TestShippedConfigMatchesTheLadder -v`
Expected: FAIL — tier keys are `('laptop', 'workstation', 'bench-coder')`, not the size classes

- [ ] **Step 3: Write the implementation**

In `config.default.yaml`, set `active_tier: 4b` and replace the whole `tiers:` block (from the `tiers:` key through the end of `bench-coder`) with:

```yaml
tiers:
  # One tier per parameter size, each binding the most advanced tag the qwen3.5-3.8 family
  # offers at that size. The KEY is the size class, so a version bump (a future qwen3.9:27b)
  # is a one-line edit to the binding below and `active_tier` never churns.
  #
  # Saturday.ai supports ONLY this family — see core/model_family.py. Confidence coloring is
  # calibrated per model (core/confidence_calibration.py); a red run means "worse than 95 % of
  # THIS model's clean output", which is only true for a model that was measured. A binding
  # outside the family is substituted in memory with the nearest size class and warned about at
  # startup; this file is never rewritten behind your back.
  #
  # The 0.8b and 2b tiers are offered for completeness. The plan/execute engine leans on
  # reliable structured output and native tool-calling, and neither size has been through the
  # trust benchmark — treat them as unvalidated.
  0.8b:
    provider: ollama
    roles:
      planner: "qwen3.5:0.8B"       # capital B — the Ollama tag is case-sensitive
      tool_caller: "qwen3.5:0.8B"
      synthesizer: "qwen3.5:0.8B"
      utility: "qwen3.5:0.8B"
      judge: "qwen3.5:0.8B"
    embedder: qwen3-embedding:8b

  2b:
    provider: ollama
    roles:
      planner: "qwen3.5:2b"
      tool_caller: "qwen3.5:2b"
      synthesizer: "qwen3.5:2b"
      utility: "qwen3.5:2b"
      judge: "qwen3.5:2b"
    embedder: qwen3-embedding:8b

  # The fresh-install default. Must stay in sync with the model list the install scripts pull
  # (install.sh / install.ps1 SATURDAY_MODELS) — a fresh install sets this tier active, so a
  # mismatch breaks the first run.
  4b:
    provider: ollama
    roles:
      planner: "qwen3.5:4b"
      tool_caller: "qwen3.5:4b"
      synthesizer: "qwen3.5:4b"
      utility: "qwen3.5:4b"
      judge: "qwen3.5:4b"
    embedder: qwen3-embedding:8b

  9b:
    provider: ollama
    roles:
      planner: "qwen3.5:9b"
      tool_caller: "qwen3.5:9b"
      synthesizer: "qwen3.5:9b"
      utility: "qwen3.5:9b"
      judge: "qwen3.5:9b"
    embedder: qwen3-embedding:8b

  27b:
    provider: ollama
    roles:
      planner: "qwen3.8:27b"
      tool_caller: "qwen3.8:27b"
      synthesizer: "qwen3.8:27b"
      utility: "qwen3.8:27b"
      judge: "qwen3.8:27b"
    embedder: qwen3-embedding:8b

  35b:
    provider: ollama
    roles:
      planner: "qwen3.6:35b"
      tool_caller: "qwen3.6:35b"
      synthesizer: "qwen3.6:35b"
      utility: "qwen3.6:35b"
      judge: "qwen3.6:35b"
    embedder: qwen3-embedding:8b
```

Then replace the whole `capabilities:` block with:

```yaml
capabilities:
  # context_window is the RUNTIME window handed to ChatOllama (config.num_ctx_for). It is
  # deliberately NOT the architectural maximum: every qwen3.5-3.8 tag reports 262144, and
  # requesting that per call exhausts VRAM on any consumer card. max_context_window carries the
  # real ceiling for DISPLAY only (the /models metrics columns). Never collapse the two.
  #
  # supports_vision: these models advertise a vision capability; Saturday.ai does not do vision
  # (a formal scope cut), so the descriptor says what SATURN supports, not what the weights can.
  "qwen3.5:0.8B":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
  "qwen3.5:2b":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
  "qwen3.5:4b":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
  "qwen3.5:9b":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
  "qwen3.8:27b":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
  "qwen3.6:35b":
    supports_tools: true
    supports_structured_output: true
    context_window: 32768
    max_context_window: 262144
    supports_vision: false
```

Apply the **same two blocks** to `config.yaml`, but set `active_tier: 27b` there (this machine runs the 27b tier). Do not stage `config.yaml` — it is untracked user data.

In `install.sh` line 24:

```bash
MODELS="${SATURDAY_MODELS:-qwen3.5:4b qwen3-embedding:8b}"
```

In `install.ps1` line 22:

```powershell
$Models     = if ($env:SATURDAY_MODELS) { $env:SATURDAY_MODELS -split '\s+' } else { @('qwen3.5:4b', 'qwen3-embedding:8b') }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: PASS

Verify the live config parses and resolves:

Run: `python -c "from config import get_config; c=get_config(); print(c.active_tier, c.model_for_role('synthesizer').model, c.num_ctx_for(c.model_for_role('synthesizer').model))"`
Expected: `27b qwen3.8:27b 32768`

- [ ] **Step 5: Commit**

```bash
git add config.default.yaml install.sh install.ps1 tests/test_model_family.py
git commit -m "config: six size-class tiers on the qwen3.5-3.8 ladder

Replaces laptop/workstation/bench-coder with one tier per parameter size.
Fresh-install default moves to qwen3.5:4b (3.4 GB, down from gemma4:e4b at
9.6 GB) and the installers pull it. Capabilities gain max_context_window
for display while context_window stays the 32768 runtime window.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Startup reports the substitution

**Files:**
- Modify: `core/llms.py` (`check_models`, ~line 396)
- Test: `tests/test_model_family.py` (append a class)

**Interfaces:**
- Consumes: `config.migrated_bindings()` (Task 2)
- Produces: `llms._migration_problems() -> list[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_family.py`:

```python
class TestStartupReportsMigrations:
    def setup_method(self):
        import config

        config.clear_migrations()

    def teardown_method(self):
        import config

        config.clear_migrations()

    def test_no_migration_reports_nothing(self):
        from core import llms

        assert llms._migration_problems() == []

    def test_a_migration_is_reported_with_both_ids_and_the_fix(self):
        import config
        from config import Config
        from core import llms

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {"synthesizer": "gemma4:e4b"}}},
        })
        cfg.model_for_role("synthesizer")

        problems = llms._migration_problems()
        assert len(problems) == 1
        line = problems[0]
        assert "gemma4:e4b" in line          # what the file says
        assert "qwen3.5:4b" in line          # what is actually running
        assert "/models tier" in line        # how to make it permanent
        assert "config.yaml" in line         # and that the file was NOT rewritten

    def test_each_distinct_substitution_is_reported_once(self):
        from config import Config
        from core import llms

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "planner": "gemma4:e4b", "synthesizer": "gemma4:e4b",
                "judge": "qwen3-coder:30b",
            }}},
        })
        for role in ("planner", "synthesizer", "judge"):
            cfg.model_for_role(role)

        assert len(llms._migration_problems()) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_model_family.py::TestStartupReportsMigrations -v`
Expected: FAIL — `AttributeError: module 'core.llms' has no attribute '_migration_problems'`

- [ ] **Step 3: Write the implementation**

In `core/llms.py`, add above `check_models`:

```python
def _migration_problems() -> list[str]:
    """One health line per family substitution made this session. Kept separate from
    check_models so it is unit-testable without a daemon."""
    import config as _config

    out = []
    for original, replacement in sorted(_config.migrated_bindings().items()):
        out.append(
            f"'{original}' is outside the supported model family and is running as "
            f"'{replacement}' — confidence coloring is calibrated per model, so only the "
            f"qwen3.5/3.6/3.8 family is supported. config.yaml was NOT changed; make it "
            f"permanent with `/models tier <size>` (see /models tier for the list)"
        )
    return out
```

In `check_models`, after the `for role in MODEL_ROLES:` loop that populates `need_ollama` and before the embedder append, add:

```python
    # Family substitutions are recorded by config.model_for_role during the loop above, so the
    # ledger is populated by now. Report them as health problems: the running config differs
    # from the file on disk until the user rebinds.
    problems.extend(_migration_problems())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/llms.py tests/test_model_family.py
git commit -m "llms: check_models reports family substitutions at startup

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `/models` refuses outsiders and shows metrics

**Files:**
- Modify: `commands/runtime.py` (`_bind` ~line 67, the `tier` subcommand ~line 201)
- Modify: `tui/ui/readouts.py` (`show_models` ~line 85)
- Test: `tests/test_models_command.py` (create)

**Interfaces:**
- Consumes: `core.model_family` (Task 1), `config.migrated_bindings()` (Task 2), `config.capability_of().max_context_window` (Task 2)
- Produces: `ui.show_models(models, bindings, active_tier, embedder, *, numbered=False, meta=None)` where `meta` maps a model id to `{"ctx": int, "max_ctx": int, "calibrated": bool}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_command.py`:

```python
"""/models — the family refusal and the metrics readout."""

import pytest


@pytest.fixture
def printed(monkeypatch):
    """Capture the command module's output lines."""
    lines = []
    monkeypatch.setattr("commands.runtime._print", lambda line="": lines.append(str(line)))
    return lines


@pytest.fixture
def cfg():
    from config import Config

    return Config({
        "active_tier": "4b",
        "tiers": {"4b": {"provider": "ollama", "roles": {
            "planner": "qwen3.5:4b", "tool_caller": "qwen3.5:4b",
            "synthesizer": "qwen3.5:4b", "utility": "qwen3.5:4b", "judge": "qwen3.5:4b",
        }, "embedder": "qwen3-embedding:8b"}},
        "capabilities": {},
    })


class TestBindRefusal:
    def test_a_non_family_bind_is_refused(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        runtime._bind(cfg, "synthesizer", "gemma4:e4b")

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:4b"   # unchanged
        blob = "\n".join(printed)
        assert "gemma4:e4b" in blob
        assert "qwen3.5:4b" in blob      # the ladder is shown as the fix

    def test_the_refusal_never_persists_anything(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr(
            "commands.runtime._persist_bindings",
            lambda *a, **k: pytest.fail("a refused bind must not persist"),
        )
        runtime._bind(cfg, "all", "qwen3-coder:30b")

    def test_a_family_bind_still_works(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "synthesizer", "qwen3.5:9b")

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:9b"

    def test_the_embedder_is_exempt_from_the_family_gate(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "embedder", "qwen3-embedding:4b")

        assert cfg.get("tiers.4b.embedder") == "qwen3-embedding:4b"


class TestTierMetrics:
    def test_tier_rows_carry_params_context_and_calibration(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows()
        keys = [r["key"] for r in rows]
        assert keys == ["0.8b", "2b", "4b", "9b", "27b", "35b"]
        for row in rows:
            assert row["model"]
            assert row["params"]              # e.g. "27.3B"
            assert row["ctx"] == 32768
            assert row["max_ctx"] == 262144
            assert isinstance(row["calibrated"], bool)

    def test_the_active_tier_is_marked(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows(active="9b")
        assert [r["key"] for r in rows if r["active"]] == ["9b"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_models_command.py -v`
Expected: FAIL — `ImportError: cannot import name '_tier_rows'`, and the refusal tests fail because the bind succeeds

- [ ] **Step 3: Write the implementation**

In `commands/runtime.py`, add the import at the top of the file's import block:

```python
from core import model_family
```

At the top of `_bind`, immediately after the `from core.llms import reset_models` line, insert the gate:

```python
    # The family gate (2026-08-16). The EMBEDDER is exempt — it is not a chat model, has no
    # raw-mode template and produces no logprobs, so no calibration claim rides on it.
    if target != "embedder" and not model_family.in_family(model):
        _print(f"  {model} is outside the supported model family.")
        _print("  Saturday.ai binds qwen3.5 / qwen3.6 / qwen3.8 only — confidence coloring is")
        _print("  calibrated per model, so a red run is only a true claim for a measured one.")
        _print("  supported:")
        for key, tag in model_family.SIZE_LADDER:
            _print(f"    {key:<6} {tag}")
        _print("  switch the whole tier with `/models tier <size>`.")
        return
```

Add the metrics row builder above the `@command("models", ...)` decorator:

```python
# Parameter counts as `ollama show` reports them — display only, so the listing reads the same
# whether or not the tag is pulled locally. (Defined before _tier_rows, which reads it.)
_PARAMS_BY_CLASS = {
    "0.8b": "873M", "2b": "2.3B", "4b": "4.7B", "9b": "9.7B", "27b": "27.3B", "35b": "36.0B",
}


def _tier_rows(active: "str | None" = None) -> list:
    """One row per size-class tier, carrying the metrics the listing shows instead of a
    nickname: bound model, parameters, runtime + architectural context window, and whether the
    model has a confidence calibration behind it."""
    from config import get_config
    from core import confidence

    cfg = get_config()
    if active is None:
        active = cfg.active_tier
    rows = []
    for key, tag in model_family.SIZE_LADDER:
        cap = cfg.capability_of(tag)
        rows.append({
            "key": key,
            "model": tag,
            "params": _PARAMS_BY_CLASS.get(key, ""),
            "ctx": cap.context_window,
            "max_ctx": cap.max_context_window,
            "calibrated": confidence.calibration_for(tag) is not None,
            "active": key == active,
        })
    return rows
```

Replace the bare-`tier` listing branch (the `if len(args) < 2:` block inside `if sub == "tier":`) with:

```python
        if len(args) < 2:
            from tui import ui

            ui.section("tiers", "switch with /models tier <size>")
            rows = [
                (
                    ("* " if r["active"] else "  ") + r["key"],
                    r["model"],
                    (r["params"], "dim"),
                    (f"{r['ctx']} / {r['max_ctx']}", "dim"),
                    ("calibrated", "green") if r["calibrated"] else ("uncalibrated", "yellow"),
                )
                for r in _tier_rows()
            ]
            ui.table(rows)
            migrated = _config_migrations()
            for original, replacement in migrated.items():
                _print(f"  note: '{original}' in config.yaml is running as '{replacement}'.")
            return
```

Add the small accessor next to `_tier_rows`:

```python
def _config_migrations() -> dict:
    """This session's family substitutions, so the listing never claims the file's value is
    what is running."""
    import config as _config

    return _config.migrated_bindings()
```

In `tui/ui/readouts.py`, change the `show_models` signature and the metadata line:

```python
def show_models(models, bindings: dict, active_tier: str, embedder: str,
                *, numbered: bool = False, meta: "dict | None" = None) -> None:
```

and inside the loop, replace the `meta = " ".join(...)` line (rename the local so it does not shadow the new parameter):

```python
            info = meta.get(m.name) if meta else None
            bits = [p for p in (m.parameter_size, m.quantization) if p]
            if info:
                bits.append(f"{info['ctx']}/{info['max_ctx']}")
            detail = " ".join(bits) or "·"
```

Then replace every later use of `meta` in that loop body with `detail`, and widen the column from `{meta:<14}` to `{detail:<26}` in both the Rich and plain branches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_models_command.py -v`
Expected: PASS

Run: `python -m pytest tests/ -q -k "models or help or onboarding"`
Expected: no NEW failures beyond the retired-model-id ones noted in Task 2

- [ ] **Step 5: Commit**

```bash
git add commands/runtime.py tui/ui/readouts.py tests/test_models_command.py
git commit -m "models: refuse non-family binds, show metrics instead of nicknames

/models <role> <id> refuses anything outside qwen3.5/3.6/3.8 and prints the
size ladder; /models tier lists params, runtime/max context and calibration
state per size class, plus a note naming any live substitution.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The user overlay store

**Files:**
- Create: `core/confidence_store.py`
- Test: `tests/test_confidence_store.py`

**Interfaces:**
- Consumes: `config.get_config().path("database")`
- Produces: `entry_for(model) -> dict | None`, `write_entry(model, enter, exit, *, source, tokens=0, prompts=0) -> None`, `clear_entry(model) -> bool`, `read() -> dict`, `load_problem() -> str`, `store_path() -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_confidence_store.py`:

```python
"""The per-model confidence overlay: what /confidence tune and /confidence set write."""

import json

import pytest


@pytest.fixture
def store(isolated_paths):
    from core import confidence_store

    confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
    confidence_store._reset_cache()
    yield confidence_store
    confidence_store._reset_cache()


class TestRoundTrip:
    def test_written_values_come_back(self, store):
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="tuned", tokens=1200, prompts=55)
        rec = store.entry_for("qwen3.8:27b")
        assert rec["enter"] == 0.31
        assert rec["exit"] == 0.52
        assert rec["source"] == "tuned"
        assert rec["tokens"] == 1200
        assert rec["prompts"] == 55
        assert rec["at"]                       # stamped by the store

    def test_lookup_is_case_insensitive(self, store):
        store.write_entry("QWEN3.8:27B", 0.31, 0.52, source="manual")
        assert store.entry_for("qwen3.8:27b")["enter"] == 0.31

    def test_unknown_model_is_none(self, store):
        assert store.entry_for("qwen3.5:9b") is None

    def test_writing_one_model_leaves_another_alone(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="manual")
        assert store.entry_for("qwen3.5:9b")["enter"] == 0.20
        assert store.entry_for("qwen3.8:27b")["enter"] == 0.31

    def test_rewriting_a_model_replaces_its_entry(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.5:9b", 0.44, 0.66, source="tuned")
        rec = store.entry_for("qwen3.5:9b")
        assert (rec["enter"], rec["exit"], rec["source"]) == (0.44, 0.66, "tuned")


class TestClear:
    def test_clear_removes_only_the_named_model(self, store):
        store.write_entry("qwen3.5:9b", 0.20, 0.30, source="manual")
        store.write_entry("qwen3.8:27b", 0.31, 0.52, source="manual")
        assert store.clear_entry("qwen3.5:9b") is True
        assert store.entry_for("qwen3.5:9b") is None
        assert store.entry_for("qwen3.8:27b") is not None

    def test_clearing_an_absent_model_reports_false(self, store):
        assert store.clear_entry("qwen3.5:2b") is False


class TestFailSoft:
    def test_a_garbled_file_reads_empty_and_records_the_problem(self, store):
        store.store_path().write_text("{not json", encoding="utf-8")
        store._reset_cache()
        assert store.read() == {}
        assert store.entry_for("qwen3.8:27b") is None
        assert store.load_problem()          # a confidence failure never costs the answer

    def test_a_missing_file_is_simply_empty(self, store):
        if store.store_path().exists():
            store.store_path().unlink()
        store._reset_cache()
        assert store.read() == {}
        assert store.load_problem() == ""

    def test_a_non_mapping_payload_is_ignored(self, store):
        store.store_path().write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        store._reset_cache()
        assert store.read() == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_confidence_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.confidence_store'`

- [ ] **Step 3: Write the implementation**

Create `core/confidence_store.py`:

```python
"""
The per-model confidence overlay the USER writes (2026-08-16).

`core/confidence_calibration.py` is the SHIPPED baseline — a generated data module the wheel
carries, regenerated by utilities/confidence_calibrate.py. It is read-only at runtime: a command
writing into it would be clobbered by /update and is not even writable in a pipx install.

So `/confidence tune` and `/confidence set` write HERE instead — one JSON file in user data
(`database/confidence_calibration.json`), keyed by model tag, that wins over the shipped table.
Same precedent as database/permissions.json: durable, user-owned, gitignored, survives /update.

One owner, one parser: core/confidence reads through entry_for(), commands/confidence writes
through write_entry()/clear_entry(). Fail-soft throughout — a garbled or unreadable overlay is
recorded in load_problem() and treated as empty, because a confidence failure must never cost
the answer.

LEAF: stdlib + config only.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from config import get_config

_FILENAME = "confidence_calibration.json"

# Cached parse, invalidated by mtime — the threshold lookup runs per streamed answer.
_cache: dict = {"key": None, "data": {}}
_problem: str = ""


def store_path() -> Path:
    """Where the overlay lives: <database>/confidence_calibration.json."""
    return Path(get_config().path("database")) / _FILENAME


def _reset_cache() -> None:
    """Drop the cached parse (a write, or a test)."""
    global _problem
    _cache["key"] = None
    _cache["data"] = {}
    _problem = ""


def load_problem() -> str:
    """Why the overlay could not be read (empty when fine). Surfaced by /confidence."""
    return _problem


def read() -> dict:
    """The whole overlay as {lowercased tag: record}. Empty on any problem — never raises."""
    global _problem
    path = store_path()
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        _reset_cache()
        return {}
    if _cache["key"] == key:
        return _cache["data"]

    data: dict = {}
    _problem = ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for tag, rec in raw.items():
                if isinstance(rec, dict):
                    data[str(tag).lower()] = dict(rec)
        else:
            _problem = f"{path.name} is not a JSON object — ignoring it"
    except (OSError, ValueError) as exc:
        _problem = f"{path.name} could not be read ({exc}) — using the shipped thresholds"
        data = {}

    _cache["key"] = key
    _cache["data"] = data
    return data


def entry_for(model) -> "dict | None":
    """The user's record for `model` (tag case-insensitive), or None."""
    return read().get(str(model or "").lower()) or None


def write_entry(model, enter: float, exit: float, *, source: str,
                tokens: int = 0, prompts: int = 0) -> None:
    """Record the user's thresholds for one model. `source` is 'tuned' (measured by
    /confidence tune) or 'manual' (typed via /confidence set) — the status readout says which.
    Written atomically so a crash mid-write can't leave a truncated overlay behind."""
    data = dict(read())
    data[str(model or "").lower()] = {
        "enter": float(enter),
        "exit": float(exit),
        "tokens": int(tokens),
        "prompts": int(prompts),
        "at": date.today().isoformat(),
        "source": str(source),
    }
    _write_all(data)


def clear_entry(model) -> bool:
    """Drop one model's record. Returns whether there was one."""
    data = dict(read())
    if data.pop(str(model or "").lower(), None) is None:
        return False
    _write_all(data)
    return True


def _write_all(data: dict) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _reset_cache()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_confidence_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/confidence_store.py tests/test_confidence_store.py
git commit -m "confidence_store: user-owned per-model threshold overlay

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Confidence consults the overlay

**Files:**
- Modify: `core/confidence.py` (`calibration_for` ~line 66, `threshold` docstring)
- Test: `tests/test_confidence.py` (append a class)

**Interfaces:**
- Consumes: `core.confidence_store.entry_for` (Task 6)
- Produces: `calibration_for(model)` now returns the overlay record when present; `threshold()` / `exit_threshold()` resolution order becomes pin → overlay → shipped → default

- [ ] **Step 1: Write the failing test**

Append to `tests/test_confidence.py`:

```python
class TestOverlayResolutionOrder:
    """pin > user overlay > shipped table > built-in default."""

    @pytest.fixture
    def bound(self, isolated_paths, monkeypatch):
        from core import confidence, confidence_store

        confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
        confidence_store._reset_cache()
        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "qwen3.8:27b")
        yield confidence_store
        confidence_store._reset_cache()

    def _no_pin(self, monkeypatch):
        from core import confidence

        monkeypatch.setattr(confidence, "_configured_threshold", lambda: None)

    def test_the_shipped_table_is_used_when_there_is_no_overlay(self, bound, monkeypatch):
        from core import confidence, confidence_calibration

        self._no_pin(monkeypatch)
        shipped = confidence_calibration.CALIBRATION["qwen3.8:27b"]["enter"]
        assert confidence.threshold() == pytest.approx(shipped)

    def test_the_overlay_wins_over_the_shipped_table(self, bound, monkeypatch):
        from core import confidence

        self._no_pin(monkeypatch)
        bound.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        assert confidence.threshold() == pytest.approx(0.41)
        assert confidence.exit_threshold() == pytest.approx(0.62)

    def test_an_explicit_numeric_pin_still_wins_over_the_overlay(self, bound, monkeypatch):
        from core import confidence

        bound.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        monkeypatch.setattr(confidence, "_configured_threshold", lambda: 0.11)
        assert confidence.threshold() == pytest.approx(0.11)

    def test_an_uncalibrated_model_falls_back_to_the_builtin_default(self, monkeypatch,
                                                                    isolated_paths):
        from core import confidence, confidence_store

        confidence_store._reset_cache()
        self._no_pin(monkeypatch)
        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "nothing:known")
        assert confidence.threshold() == pytest.approx(confidence._DEFAULT_THRESHOLD)

    def test_a_garbled_overlay_degrades_to_the_shipped_table(self, bound, monkeypatch):
        from core import confidence, confidence_calibration

        self._no_pin(monkeypatch)
        bound.store_path().write_text("{ not json", encoding="utf-8")
        bound._reset_cache()
        shipped = confidence_calibration.CALIBRATION["qwen3.8:27b"]["enter"]
        assert confidence.threshold() == pytest.approx(shipped)
```

Ensure `import pytest` is present at the top of `tests/test_confidence.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_confidence.py::TestOverlayResolutionOrder -v`
Expected: FAIL — `test_the_overlay_wins_over_the_shipped_table` returns the shipped value

- [ ] **Step 3: Write the implementation**

In `core/confidence.py`, replace `calibration_for`:

```python
def calibration_for(model: str) -> "dict | None":
    """The calibrated {enter, exit, …} record for `model` (tag case-insensitive), or None.

    The USER's overlay wins (core/confidence_store — written by /confidence tune|set), then the
    SHIPPED table (core/confidence_calibration, regenerated by
    utilities/confidence_calibrate.py). An unreadable overlay degrades to the shipped table:
    marking is additive, and a confidence failure must never cost the answer."""
    tag = str(model or "").lower()
    try:
        from core import confidence_store

        rec = confidence_store.entry_for(tag)
        if rec:
            return rec
    except Exception:
        pass

    from core import confidence_calibration as table  # generated data module

    try:
        return table.CALIBRATION.get(tag)
    except Exception:
        return None
```

Update `threshold`'s docstring to name the four-step order:

```python
def threshold() -> float:
    """The low-token probability threshold, in resolution order: an explicit numeric
    `runtime.confidence_threshold` pin, then the synthesizer model's calibrated `enter` —
    the user's overlay first, then the shipped table (calibration_for) — then the built-in
    default. `auto` (the default) means "skip the pin"."""
```

`exit_threshold` needs no change: it already reads `rec["exit"]` off `calibration_for`, which now consults the overlay.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_confidence.py -v`
Expected: PASS (the whole file — the existing grading tests must stay green)

- [ ] **Step 5: Commit**

```bash
git add core/confidence.py tests/test_confidence.py
git commit -m "confidence: the user overlay wins over the shipped calibration table

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Move the measurement into a shipped module

**Files:**
- Create: `core/calibration.py`
- Modify: `utilities/confidence_calibrate.py` (becomes a thin CLI over the new module)
- Test: `tests/test_calibration.py` (create)

**Interfaces:**
- Consumes: `core.confidence.align_chunk`, `.is_stopword`; `core.llms._build`, `.stream`; `core.serving.num_predict`; `core.messages.synthesize_sys_msg`; `trust.egress`
- Produces: `core.calibration.PROMPTS: list[str]`, `quantile(sorted_ps, q) -> float`, `summarize(ps) -> dict` (keys `enter`, `exit`, `tokens`), `measure(tag, prompts=None, on_progress=None) -> dict` (keys `enter`, `exit`, `tokens`, `prompts`)

**Why:** `utilities/` is **not** shipped in the wheel (`pyproject.toml` `packages.find` covers `core*` but not `utilities`), so `/confidence tune` cannot import it in an installed run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration.py`:

```python
"""The measurement core behind /confidence tune — pure parts only, no daemon."""

import pytest

from core import calibration


class TestQuantile:
    def test_picks_the_expected_point(self):
        ps = [round(0.01 * i, 2) for i in range(1, 101)]
        assert calibration.quantile(ps, 0.05) == pytest.approx(0.05, abs=0.02)
        assert calibration.quantile(ps, 0.10) == pytest.approx(0.10, abs=0.02)

    def test_empty_is_zero(self):
        assert calibration.quantile([], 0.05) == 0.0

    def test_single_value(self):
        assert calibration.quantile([0.4], 0.05) == 0.4


class TestSummarize:
    def test_enter_is_the_5pct_point_and_exit_the_10pct(self):
        ps = [round(0.01 * i, 2) for i in range(1, 101)]
        out = calibration.summarize(ps)
        assert out["enter"] < out["exit"]
        assert out["tokens"] == 100

    def test_no_samples_yields_no_thresholds(self):
        out = calibration.summarize([])
        assert out["tokens"] == 0
        assert out["enter"] == 0.0
        assert out["exit"] == 0.0

    def test_the_result_is_sorted_independent(self):
        ps = [0.9, 0.1, 0.5, 0.3, 0.7]
        assert calibration.summarize(ps) == calibration.summarize(sorted(ps, reverse=True))


class TestPrompts:
    def test_the_prompt_set_is_non_empty_and_stable(self):
        assert len(calibration.PROMPTS) >= 40
        assert all(isinstance(p, str) and p.strip() for p in calibration.PROMPTS)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.calibration'`

- [ ] **Step 3: Write the implementation**

Create `core/calibration.py` by **moving** (not copying) from `utilities/confidence_calibrate.py`: the `PROMPTS` list verbatim, `_score_model` (renamed `measure`), and `_quantile` (renamed `quantile`). Header:

```python
"""
Confidence calibration — the measurement (2026-08-16, moved out of utilities/ on the family lock).

Streams KNOWN-GOOD prompts through the synthesizer's real prompt shape and serving kwargs, scores
every content token by its sampled probability EXACTLY as core/confidence.low_runs does (neutral
punctuation and closed-class stopwords excluded), and reports the 5 % point as `enter` and the
10 % point as `exit`: "marked" then means "worse than 95 % of this model's clean output".

This lives in core/ (not utilities/) because it SHIPS: `/confidence tune` re-measures the active
synthesizer at runtime, and utilities/ is excluded from the wheel. Two callers, one measurement:
utilities/confidence_calibrate.py writes the shipped baseline table, /confidence tune writes the
user overlay (core/confidence_store).

Goes through core.llms' Ollama boundary (air-gap / locality honored); not a new egress path.
"""

from __future__ import annotations

import math

PROMPTS: list[str] = [
    # ... the exact list moved from utilities/confidence_calibrate.py ...
]


def quantile(values: list, q: float) -> float:
    """The q-quantile of `values` (sorted internally). 0.0 for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    i = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[i]


def summarize(ps: list) -> dict:
    """Token probabilities -> the calibrated pair. `enter` = the 5 % point (a run opens under
    it), `exit` = the 10 % point (an open run extends under it)."""
    return {
        "enter": round(quantile(ps, 0.05), 4),
        "exit": round(quantile(ps, 0.10), 4),
        "tokens": len(ps),
    }


def measure(tag: str, prompts: "list | None" = None, on_progress=None) -> dict:
    """Stream `prompts` through `tag` and return {enter, exit, tokens, prompts}.

    `on_progress(i, total, prompt, added)` is called after each prompt so a caller can render
    a live line (the CLI prints to stderr; /confidence tune prints through the TUI).
    Raises RuntimeError when the air-gap forbids an off-machine daemon."""
    from config import get_config
    from core import confidence, llms
    from core.messages import synthesize_sys_msg
    from core.serving import num_predict
    from langchain.messages import HumanMessage
    from trust import egress

    if not egress.ollama_is_local() and egress.airgap_on():
        raise RuntimeError("air-gap is ON and OLLAMA_HOST is off-machine — refusing to calibrate")

    prompts = list(prompts if prompts is not None else PROMPTS)
    model = llms._build("ollama", tag)
    cfg = get_config()
    kwargs = {
        "options": {"temperature": 0.7, "num_ctx": cfg.num_ctx_for(tag),
                    "num_predict": num_predict("answer")},
        "reasoning": False,
        "logprobs": True,
    }

    ps: list = []
    for i, q in enumerate(prompts, 1):
        text = ""
        entries: list = []
        for chunk in llms.stream(model, [synthesize_sys_msg,
                                         HumanMessage(content=f"Current user query:\n{q}")],
                                 tag=tag, **kwargs):
            piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            lp = (getattr(chunk, "response_metadata", None) or {}).get("logprobs")
            if piece:
                if lp:
                    entries += confidence.align_chunk(piece, lp, offset=len(text))
                text += piece
        before = len(ps)
        for e in entries:
            tok = text[e["start"]:e["end"]]
            if not any(ch.isalnum() for ch in tok) or confidence.is_stopword(tok):
                continue  # neutral for low_runs — neutral for the calibration too
            ps.append(math.exp(min(float(e["logprob"]), 0.0)))
        if on_progress:
            on_progress(i, len(prompts), q, len(ps) - before)

    out = summarize(ps)
    out["prompts"] = len(prompts)
    return out
```

Then rewrite `utilities/confidence_calibrate.py` to import from `core.calibration`: delete its `PROMPTS`, `_score_model`, and `_quantile`; keep `_installed`, `_tier_synthesizers`, `_write_table`, and `main`. Inside `main`, replace the per-model scoring block with:

```python
        try:
            result = calibration.measure(
                tag, prompts,
                on_progress=None if args.quiet else (
                    lambda i, n, q, added: print(
                        f"  [{i:>2}/{n}] {q[:48]:<48} +{added:>4} tokens", file=sys.stderr)
                ),
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        table[tag.lower()] = {
            "enter": result["enter"], "exit": result["exit"],
            "tokens": result["tokens"], "prompts": result["prompts"],
            "at": _dt.date.today().isoformat(),
        }
```

and add `from core import calibration` to its imports, dropping the now-unused `math`, `HumanMessage`, `synthesize_sys_msg`, `num_predict`, and `egress` imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: PASS

Verify the CLI still parses (no daemon call — `--help` only):

Run: `python utilities/confidence_calibrate.py --help`
Expected: usage text, exit 0

- [ ] **Step 5: Commit**

```bash
git add core/calibration.py utilities/confidence_calibrate.py tests/test_calibration.py
git commit -m "calibration: move the measurement into core/ so it ships

utilities/ is excluded from the wheel, so /confidence tune could not have
re-measured in an installed run. One measurement, two callers: the CLI
writes the shipped table, the command writes the user overlay.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The `/confidence` command

**Files:**
- Create: `commands/confidence.py`
- Modify: `commands/__init__.py` (`_COMMAND_MODULES`, ~line 28)
- Modify: `commands/system.py` (`_GROUPS`, ~line 33)
- Test: `tests/test_confidence_command.py` (create)

**Interfaces:**
- Consumes: `core.confidence.threshold/exit_threshold/calibration_for/_synthesizer_model`, `core.confidence_store` (Task 6), `core.calibration.measure` (Task 8), `commands._utils.parse_toggle_status/split_persist_flags`, `commands.config._persist_key`
- Produces: the `/confidence` slash command

- [ ] **Step 1: Write the failing test**

Create `tests/test_confidence_command.py`:

```python
"""/confidence — the front door for confidence coloring."""

import pytest


@pytest.fixture
def printed(monkeypatch):
    lines = []
    monkeypatch.setattr("commands.confidence._print", lambda line="": lines.append(str(line)))
    return lines


@pytest.fixture
def ctx():
    class Ctx:
        state = {}
        should_quit = False

    return Ctx()


@pytest.fixture
def store(isolated_paths, monkeypatch):
    from core import confidence, confidence_store

    confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
    confidence_store._reset_cache()
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "qwen3.8:27b")
    yield confidence_store
    confidence_store._reset_cache()


def _run(ctx, args):
    from commands.confidence import _confidence

    return _confidence(ctx, args)


class TestRegistration:
    def test_the_command_is_registered(self):
        import commands
        from commands._framework import COMMANDS

        assert "confidence" in COMMANDS

    def test_it_appears_in_a_help_group(self):
        from commands.system import _GROUPS

        assert any("confidence" in names for _group, names in _GROUPS)


class TestStatus:
    def test_bare_is_a_status_readout_and_never_a_flip(self, ctx, printed, store):
        from config import get_config

        before = get_config().get("runtime.confidence", True)
        _run(ctx, [])
        assert get_config().get("runtime.confidence", True) is before
        blob = "\n".join(printed)
        assert "qwen3.8:27b" in blob
        assert "enter" in blob.lower()

    def test_status_names_the_source_of_the_thresholds(self, ctx, printed, store):
        store.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        _run(ctx, [])
        assert "manual" in "\n".join(printed)


class TestToggle:
    def test_off_then_on(self, ctx, printed, store, monkeypatch):
        from config import get_config

        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: None)
        _run(ctx, ["off"])
        assert get_config().get("runtime.confidence") is False
        _run(ctx, ["on"])
        assert get_config().get("runtime.confidence") is True

    def test_garbage_is_usage_not_a_flip(self, ctx, printed, store):
        from config import get_config

        get_config().set("runtime.confidence", True)
        _run(ctx, ["maybe"])
        assert get_config().get("runtime.confidence") is True
        assert "usage" in "\n".join(printed).lower()

    def test_session_flag_does_not_persist(self, ctx, printed, store, monkeypatch):
        calls = []
        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: calls.append(a))
        _run(ctx, ["off", "--session"])
        assert calls == []

    def test_persists_by_default(self, ctx, printed, store, monkeypatch):
        calls = []
        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: calls.append(a))
        _run(ctx, ["off"])
        assert calls


class TestSet:
    def test_two_values_are_stored_for_the_active_model(self, ctx, printed, store):
        _run(ctx, ["set", "0.31", "0.52"])
        rec = store.entry_for("qwen3.8:27b")
        assert (rec["enter"], rec["exit"]) == (0.31, 0.52)
        assert rec["source"] == "manual"

    def test_one_value_derives_the_exit(self, ctx, printed, store):
        _run(ctx, ["set", "0.40"])
        rec = store.entry_for("qwen3.8:27b")
        assert rec["enter"] == 0.40
        assert rec["exit"] == pytest.approx(0.60)     # 1.5x, capped at 0.95

    def test_the_derived_exit_is_capped(self, ctx, printed, store):
        _run(ctx, ["set", "0.90"])
        assert store.entry_for("qwen3.8:27b")["exit"] == pytest.approx(0.95)

    @pytest.mark.parametrize("bad", [["0"], ["1"], ["1.5"], ["-0.2"], ["abc"], []])
    def test_out_of_range_values_are_refused(self, ctx, printed, store, bad):
        _run(ctx, ["set", *bad])
        assert store.entry_for("qwen3.8:27b") is None

    def test_an_exit_below_the_enter_is_refused(self, ctx, printed, store):
        _run(ctx, ["set", "0.50", "0.20"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "exit" in "\n".join(printed).lower()


class TestReset:
    def test_reset_drops_the_overlay_entry(self, ctx, printed, store):
        store.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        _run(ctx, ["reset"])
        assert store.entry_for("qwen3.8:27b") is None

    def test_reset_with_nothing_stored_says_so(self, ctx, printed, store):
        _run(ctx, ["reset"])
        assert "shipped" in "\n".join(printed).lower() or "nothing" in "\n".join(printed).lower()


class TestTune:
    def test_tune_writes_what_it_measured(self, ctx, printed, store, monkeypatch):
        from core import calibration

        monkeypatch.setattr(
            calibration, "measure",
            lambda tag, prompts=None, on_progress=None: {
                "enter": 0.27, "exit": 0.44, "tokens": 900, "prompts": 55},
        )
        _run(ctx, ["tune"])
        rec = store.entry_for("qwen3.8:27b")
        assert (rec["enter"], rec["exit"]) == (0.27, 0.44)
        assert rec["source"] == "tuned"
        assert rec["tokens"] == 900

    def test_a_daemon_failure_is_reported_and_writes_nothing(self, ctx, printed, store,
                                                             monkeypatch):
        from core import calibration

        def boom(*a, **k):
            raise RuntimeError("daemon unreachable")

        monkeypatch.setattr(calibration, "measure", boom)
        _run(ctx, ["tune"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "daemon unreachable" in "\n".join(printed)

    def test_prompts_flag_limits_the_sample(self, ctx, printed, store, monkeypatch):
        from core import calibration

        seen = {}

        def spy(tag, prompts=None, on_progress=None):
            seen["n"] = len(prompts) if prompts is not None else None
            return {"enter": 0.3, "exit": 0.5, "tokens": 10, "prompts": len(prompts or [])}

        monkeypatch.setattr(calibration, "measure", spy)
        _run(ctx, ["tune", "--prompts", "7"])
        assert seen["n"] == 7

    def test_a_zero_token_measurement_is_refused(self, ctx, printed, store, monkeypatch):
        from core import calibration

        monkeypatch.setattr(
            calibration, "measure",
            lambda tag, prompts=None, on_progress=None: {
                "enter": 0.0, "exit": 0.0, "tokens": 0, "prompts": 55},
        )
        _run(ctx, ["tune"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "logprob" in "\n".join(printed).lower()


class TestUnknownSubcommand:
    def test_unknown_subcommand_errors_with_usage(self, ctx, printed, store):
        _run(ctx, ["tunne"])
        assert "usage" in "\n".join(printed).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_confidence_command.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'commands.confidence'`

- [ ] **Step 3: Write the implementation**

Create `commands/confidence.py`:

```python
"""
/confidence — the front door for confidence coloring.

Coloring is ON by default and part of the ordinary experience: while the answer streams, runs of
consecutive low-probability tokens render red. What "low" means is CALIBRATED PER MODEL
(core/confidence_calibration.py, the shipped baseline) — a red run says "worse than 95 % of this
model's clean output", which is why Saturday.ai binds one model family (core/model_family.py).

This command owns every user-facing lever over that: the on/off switch, re-measuring the active
synthesizer against the live daemon, and typing your own numbers. Measured and typed values land
in the user overlay (core/confidence_store), which wins over the shipped table.
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import parse_toggle_status, split_persist_flags

_USAGE = (
    "  usage: /confidence [on|off] | set <enter> [exit] | tune [--prompts N] | reset"
)


def _persist(cfg, key: str) -> None:
    """Persist one scalar through the ONE persist seam (same machinery as /config --save)."""
    from commands.config import _persist_key

    _persist_key(cfg, key)


def _active_model() -> str:
    from core import confidence

    return confidence._synthesizer_model()


def _source_of(model: str) -> str:
    """Where the live thresholds come from, for the status line."""
    from core import confidence, confidence_store

    if confidence._configured_threshold() is not None:
        return "pinned (runtime.confidence_threshold)"
    rec = confidence_store.entry_for(model)
    if rec:
        return f"{rec.get('source', 'overlay')} · {rec.get('at', '')}".strip(" ·")
    from core import confidence_calibration as table

    if table.CALIBRATION.get(str(model or "").lower()):
        return "shipped calibration"
    return "built-in default (uncalibrated model)"


def _status() -> None:
    from config import get_config
    from core import confidence, confidence_store

    cfg = get_config()
    on = bool(cfg.get("runtime.confidence", True))
    model = _active_model()

    _print(f"  coloring   {'on' if on else 'off'}")
    _print(f"  model      {model or '—'}")
    _print(f"  enter      {confidence.threshold():.4f}   (a run opens below this probability)")
    _print(f"  exit       {confidence.exit_threshold():.4f}   (an open run extends below this)")
    _print(f"  source     {_source_of(model)}")

    rec = confidence_store.entry_for(model)
    if rec and rec.get("tokens"):
        _print(f"  measured   {rec['tokens']} tokens over {rec.get('prompts', 0)} prompts")
    problem = confidence_store.load_problem()
    if problem:
        _print(f"  ! {problem}")
    if not on:
        _print("  turn it back on with `/confidence on`.")


def _set(args: list) -> None:
    from core import confidence, confidence_store

    if not args:
        _print("  usage: /confidence set <enter> [exit]   (probabilities, 0 < p < 1)")
        return
    try:
        enter = float(args[0])
    except (TypeError, ValueError):
        _print(f"  not a probability: {args[0]!r} (expected a number between 0 and 1)")
        return
    if not 0.0 < enter < 1.0:
        _print(f"  enter must be between 0 and 1 (got {enter})")
        return

    if len(args) > 1:
        try:
            exit_p = float(args[1])
        except (TypeError, ValueError):
            _print(f"  not a probability: {args[1]!r}")
            return
        if not 0.0 < exit_p < 1.0:
            _print(f"  exit must be between 0 and 1 (got {exit_p})")
            return
        if exit_p <= enter:
            _print(f"  exit ({exit_p}) must be ABOVE enter ({enter}) — the exit threshold is the "
                   "looser one an open run extends through.")
            return
    else:
        # The same derivation an uncalibrated model gets: looser by 1.5x, capped.
        exit_p = min(confidence._EXIT_CAP, enter * confidence._EXIT_FACTOR)

    model = _active_model()
    confidence_store.write_entry(model, enter, exit_p, source="manual")
    _print(f"  {model}: enter {enter:.4f} · exit {exit_p:.4f} (yours).")
    _print("  `/confidence reset` restores the shipped calibration.")


def _reset() -> None:
    from core import confidence_store

    model = _active_model()
    if confidence_store.clear_entry(model):
        _print(f"  {model}: your values dropped — back to the shipped calibration.")
    else:
        _print(f"  {model}: nothing of yours stored; already on the shipped calibration.")


def _tune(args: list) -> None:
    from core import calibration, confidence_store

    prompts = None
    if "--prompts" in args:
        i = args.index("--prompts")
        try:
            n = int(args[i + 1])
        except (IndexError, TypeError, ValueError):
            _print("  usage: /confidence tune [--prompts N]")
            return
        if n < 1:
            _print("  --prompts must be at least 1")
            return
        prompts = calibration.PROMPTS[:n]

    model = _active_model()
    if not model:
        _print("  no synthesizer model bound — nothing to calibrate.")
        return

    total = len(prompts) if prompts is not None else len(calibration.PROMPTS)
    _print(f"  calibrating {model} over {total} prompts — this runs the live daemon.")

    def _progress(i, n, prompt, added):
        _print(f"    [{i:>2}/{n}] {prompt[:44]:<44} +{added:>4} tokens")

    try:
        result = calibration.measure(model, prompts, on_progress=_progress)
    except Exception as exc:
        _print(f"  calibration failed: {exc}")
        _print("  nothing was written — the previous thresholds still apply.")
        return

    if not result.get("tokens"):
        _print("  the daemon returned no per-token logprobs for this model — nothing measured,")
        _print("  so nothing was written. Confidence marks stay on the previous thresholds.")
        return

    confidence_store.write_entry(
        model, result["enter"], result["exit"], source="tuned",
        tokens=result["tokens"], prompts=result.get("prompts", total),
    )
    _print(f"  {model}: enter {result['enter']:.4f} · exit {result['exit']:.4f} "
           f"({result['tokens']} tokens).")


@command(
    "confidence",
    "Confidence coloring: on/off, re-calibrate, or set your own thresholds.",
    usage="/confidence [on|off] | set <enter> [exit] | tune [--prompts N] | reset",
    details="""
Confidence coloring marks the parts of an answer the model itself was least sure of: while the
answer streams, runs of consecutive low-probability tokens render RED — live, in the freeze editor
(Esc), and on the final render. It is ON by default.

What counts as "low" is calibrated PER MODEL, because the same probability means different things
at different model sizes. A red run means "worse than 95 % of this model's clean output". That is
why Saturday.ai binds one model family (qwen3.5 / qwen3.6 / qwen3.8) — see /models tier.

  /confidence                    status: on/off, the active model, its thresholds and where
                                 they came from
  /confidence off                stop capturing logprobs entirely (nothing is marked)
  /confidence on                 back on
  /confidence tune               re-measure the ACTIVE synthesizer against the live daemon and
                                 store the result as yours. Takes a few minutes.
  /confidence tune --prompts 20  a quicker, coarser pass
  /confidence set 0.31 0.52      type your own enter/exit probabilities for the active model.
                                 LOWER enter = stricter = fewer marks. Omit the exit and it is
                                 derived (1.5x, capped at 0.95).
  /confidence reset              drop your values; back to the shipped calibration

on/off persists to config.yaml by default; add --session to apply it for this session only.
Your tuned and typed values live in database/confidence_calibration.json and survive /update.
""".strip(),
)
def _confidence(ctx, args):
    from config import get_config

    args = list(args or [])
    args, session, _save = split_persist_flags(args)
    sub = args[0].lower() if args else ""

    if sub == "set":
        _set(args[1:])
        return
    if sub == "tune":
        _tune(args[1:])
        return
    if sub == "reset":
        _reset()
        return

    # Bare = STATUS (never a flip); on/off mutates. The one toggle grammar.
    verdict = parse_toggle_status(args)
    if verdict is None:
        _status()
        return
    if verdict == "invalid":
        _print(_USAGE)
        return

    cfg = get_config()
    cfg.set("runtime.confidence", bool(verdict))
    _print(f"  confidence coloring {'on' if verdict else 'off'}"
           f"{' (session only)' if session else ''}.")
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        _persist(cfg, "runtime.confidence")
```

In `commands/__init__.py`, add to `_COMMAND_MODULES` (keep the list alphabetical):

```python
    "confidence",    # /confidence — the confidence-coloring front door
```

In `commands/system.py`, add `"confidence"` to the `observability` group:

```python
    ("observability", ("confidence", "mcp", "models", "tools", "trace")),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_confidence_command.py -v`
Expected: PASS

Run: `python -m pytest tests/test_help.py -v`
Expected: PASS (the registry cross-check sees the new command in a group)

- [ ] **Step 5: Commit**

```bash
git add commands/confidence.py commands/__init__.py commands/system.py tests/test_confidence_command.py
git commit -m "confidence: the /confidence command (on/off, tune, set, reset)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Drop the unreachable gemma4 template

**Files:**
- Modify: `core/chat_template.py` (the `gemma4` `ChatTemplate` entry ~line 95, the module docstring's BOS notes ~line 22)
- Modify: `tests/test_token_steering.py:36`, `:46`, `:65` (the three gemma4 template assertions — the only real edits; `:26` and `:168` use `qwen3.6:27b`, which is still in-family and stays)
- Test: `tests/test_model_family.py` (append the agreement test)

**Interfaces:**
- Consumes: `core.model_family.FAMILY_PREFIXES` (Task 1)
- Produces: `chat_template.supported(m)` is now exactly `model_family.in_family(m)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_family.py`:

```python
class TestTemplateRegistryAgreesWithTheFamily:
    """The freeze hotkey must arm for every bindable model and no others — two lists that drift
    would either strand a supported model or promise continuation for an unbindable one."""

    def test_template_prefixes_are_exactly_the_family(self):
        from core import chat_template, model_family as mf

        covered = tuple(p for t in chat_template.TEMPLATES for p in t.prefixes)
        assert sorted(covered) == sorted(mf.FAMILY_PREFIXES)

    def test_every_ladder_tag_is_supported_for_continuation(self):
        from core import chat_template, model_family as mf

        for _key, tag in mf.SIZE_LADDER:
            assert chat_template.supported(tag), tag

    def test_a_retired_family_is_no_longer_supported(self):
        from core import chat_template

        assert not chat_template.supported("gemma4:e4b")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_model_family.py::TestTemplateRegistryAgreesWithTheFamily -v`
Expected: FAIL — the covered prefixes still include `gemma4`

- [ ] **Step 3: Write the implementation**

In `core/chat_template.py`, delete the `gemma4` `ChatTemplate(...)` entry from `TEMPLATES`. Replace the docstring's gemma4 BOS paragraph with a one-line note that the entry was removed with the family lock, keeping the qwen3.x facts intact:

```
- BOS: qwen3.x adds no BOS at all (`'hello'` -> 1 token) and `<bos>` tokenizes as ordinary text,
  so we render WITHOUT one. (The gemma4 entry — whose runner auto-added `<bos>` in raw mode —
  was removed with the family lock, 2026-08-16: gemma4 is no longer bindable. Its recovered
  format is in git history if the family ever widens.)
```

Update the `UnsupportedModel` message in `template_for` so it points at the gate:

```python
    raise UnsupportedModel(
        f"no raw-mode chat template for model {model!r} — Saturday.ai supports the "
        f"{', '.join(t.family for t in TEMPLATES)} family only (core/model_family.py); "
        f"extend the registry only for a model that passes utilities/continuation_contract.py"
    )
```

Then fix the three test assertions that name the retired template. These are the **only** test
edits this task needs — a `grep -rn "gemma4" tests/` confirms the rest are synthetic strings:

- `tests/test_token_steering.py:36` — a `render_continuation("gemma4:e4b", …)` golden. Delete the
  whole test (its qwen3.x sibling above it covers the render contract).
- `tests/test_token_steering.py:46` — `for model in ("qwen3.5:9b", "gemma4:e4b"):`. Reduce the
  tuple to `("qwen3.5:9b",)`.
- `tests/test_token_steering.py:65` — `assert chat_template.supported("gemma4:26b")`. Replace with
  `assert not chat_template.supported("gemma4:26b")` and extend the surrounding docstring to say
  the family lock retired it.

**Leave these alone** — they are synthetic fixtures, not family bindings, and the gate never sees
them: `tests/test_core.py:475` (a YAML-scalar test using the literal string `workstation`),
`tests/test_onboarding.py:69,78,100,108` (invented tier names for the doctor's honesty line), and
`tests/test_onboarding.py:115-118` (`gemma4:e4b` as a stand-in "model not pulled" string for
`_should_offer_pull`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_model_family.py -v`
Expected: PASS

Run: `python -m pytest tests/ -q`
Expected: **the whole suite passes** — this is the task that clears the failures noted in Task 2

- [ ] **Step 5: Commit**

```bash
git add core/chat_template.py tests/
git commit -m "chat_template: drop the unreachable gemma4 entry

With the family gate in place gemma4 can never be bound, so supported() is
now exactly in_family(). A test pins the two prefix lists together.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Documentation

**Files:**
- Modify: `CHANGELOG.md` (the `[Unreleased]` section)
- Modify: `README.md` (model requirements / install)
- Modify: `CLAUDE.md` (Architecture key-files, Slash commands, Data layout, config.yaml notes)
- Modify: `documentation.md` (untracked dev log — dated entry)

**Interfaces:**
- Consumes: everything above
- Produces: no code

- [ ] **Step 1: Update CHANGELOG.md**

Under `## [Unreleased]`, add:

```markdown
### Changed
- **One model family.** Saturday.ai now binds qwen3.5 / qwen3.6 / qwen3.8 only, as six tiers
  keyed by parameter size (`0.8b`, `2b`, `4b`, `9b`, `27b`, `35b`), each on the most advanced
  tag the family offers at that size. Confidence coloring is calibrated per model, and a red run
  only means "worse than 95 % of this model's clean output" for a model that was measured —
  every shipped tag is. `/models tier` lists parameters, context window and calibration state
  instead of a nickname.
- A binding left over from an older config (gemma4, qwen3-coder) is substituted in memory with
  the nearest size class and reported at startup. **config.yaml is never rewritten** — rebind
  with `/models tier <size>` to make it permanent.
- The fresh-install pull drops from `gemma4:e4b` (9.6 GB) to `qwen3.5:4b` (3.4 GB).

### Added
- **`/confidence`** — the front door for confidence coloring: `on`/`off` (on by default),
  `tune` to re-measure the active model against your daemon, `set <enter> [exit]` to type your
  own thresholds, `reset` to go back to the shipped calibration. Your values live in
  `database/confidence_calibration.json` and survive `/update`.
```

- [ ] **Step 2: Update README.md**

Replace the model-requirements section's `gemma4:e4b` references with `qwen3.5:4b`, and add:

```markdown
Saturday.ai runs the **qwen3.5–3.8 family only**, as one tier per parameter size. That is a
deliberate limit, not a missing feature: confidence coloring marks what the model itself was
least sure of, and "least sure" is calibrated per model — a threshold borrowed from a model of a
different size is meaningless. `/models tier` shows the ladder; `/confidence` shows what your
active model is calibrated at.
```

- [ ] **Step 3: Update CLAUDE.md**

Make these edits:

1. **Running** section — no change needed.
2. **Key files** — add after the `core/confidence.py` entry:
   - `core/model_family.py` — the supported-family gate (leaf, stdlib only): `FAMILY_PREFIXES` (qwen3.5/3.6/3.8, ANCHORED match), `SIZE_LADDER` (size class → the most advanced tag at that size), `in_family`/`classes`/`tag_for`/`migrate`. THE one answer to "may this model be bound"; `config.model_for_role` is the enforcement seam (substitute + record, never rewrite config.yaml), `commands/runtime._bind` refuses at the front door, `llms.check_models` reports substitutions at startup. The embedder is exempt.
   - `core/confidence_store.py` — the USER's per-model threshold overlay (leaf, config + json): `database/confidence_calibration.json`, written by `/confidence tune|set`, read by `confidence.calibration_for` ahead of the shipped table. Fail-soft (a garbled file degrades to the shipped values).
   - `core/calibration.py` — the confidence measurement, moved out of `utilities/` 2026-08-16 so it SHIPS (`/confidence tune` re-measures at runtime; utilities/ is excluded from the wheel). One measurement, two callers: the CLI writes the shipped baseline table, the command writes the overlay.
3. **`config.py` + `config.yaml`** entry — note the six size-class tiers, and that `capabilities.<model>.context_window` is the RUNTIME window while `max_context_window` is display-only (every family tag reports 262144; requesting it per call exhausts consumer VRAM).
4. **Slash commands** — bump the count from 18 to 19, add `/confidence` to the list and to the `_GROUPS` observability row description, and add a Notable entry describing the subcommands.
5. **Data layout** — add `confidence_calibration.json  # per-model confidence thresholds you tuned (/confidence tune|set)` under `database/`.
6. **Frozen surfaces** — replace "the provider set (Ollama ONLY…)" clause with a note that the MODEL FAMILY is now frozen too: qwen3.5/3.6/3.8, extended only through `utilities/continuation_contract.py` **and** a calibration run.

- [ ] **Step 4: Update documentation.md**

Add a dated entry describing the change, the decisions, and the two accepted risks (the disk-vs-running divergence, the `context_window` / `max_context_window` split).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md README.md CLAUDE.md
git commit -m "docs: the qwen3.5-3.8 family lock and /confidence

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: The calibration pass (live daemon)

**Files:**
- Modify: `core/confidence_calibration.py` (regenerated)

**Interfaces:**
- Consumes: `core.calibration.measure` via `utilities/confidence_calibrate.py` (Task 8)
- Produces: a shipped table covering all six ladder tags with real measurements

**This is the only task that needs a running Ollama daemon.** Everything above is offline.

- [ ] **Step 1: Confirm every ladder tag is pulled**

Run: `ollama list`
Expected: `qwen3.5:0.8B`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`, `qwen3.8:27b`, `qwen3.6:35b` all present. Pull anything missing with `ollama pull <tag>`.

- [ ] **Step 2: Run the calibration**

Run:

```bash
python utilities/confidence_calibrate.py --models qwen3.5:0.8B qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.8:27b qwen3.6:35b
```

Expected: a per-prompt progress stream per model, then a rewritten `core/confidence_calibration.py`. Budget several minutes per model.

- [ ] **Step 3: Verify the table**

Run:

```bash
python -c "from core.confidence_calibration import CALIBRATION as C; from core.model_family import SIZE_LADDER; [print(t, C.get(t.lower())) for _k,t in SIZE_LADDER]"
```

Expected: every tag has a record with `tokens` > 0 and `0 < enter < exit < 1`. **No record may carry `inherited_from`** — that was the unmeasured `qwen3.8:27b` entry this pass replaces.

If a model reports no logprobs, the utility prints a NOTE and records nothing for it. Do not paper over that with an inherited value: leave the tag uncalibrated (it falls back to 0.20) and record the gap in the commit message.

- [ ] **Step 4: Confirm the retired rows are gone**

Run:

```bash
python -c "from core.confidence_calibration import CALIBRATION as C; print([k for k in C if 'gemma' in k or 'coder' in k])"
```

Expected: `[]`. If not, remove those keys by hand — they are unbindable now and would only mislead.

Then the full suite one more time:

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/confidence_calibration.py
git commit -m "confidence: calibrate every shipped tag on the family ladder

Replaces the inherited, never-measured qwen3.8:27b entry with real
measurements and drops the rows for models that are no longer bindable.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Verification

After Task 12, confirm end to end:

- [ ] `python -m pytest tests/ -q` — the whole suite passes
- [ ] `python -c "from config import get_config; print(get_config().active_tier)"` — prints `27b`
- [ ] `python agent.py -p "what is 17 * 23?"` — a turn completes and the receipt renders
- [ ] In the REPL: `/confidence` shows `on`, the active model, and `shipped calibration` as the source
- [ ] In the REPL: `/models tier` shows six size classes with params, context and calibration state
- [ ] In the REPL: `/models synthesizer gemma4:e4b` is refused and prints the ladder
- [ ] `python utilities/continuation_contract.py --all` — the freeze/continue contract passes on the family

"""
Runtime configuration for Saturday.ai (Phase 3).

Loads `config.yaml` once and exposes it through a small typed accessor so the rest of the
codebase never hard-codes a model id or a filesystem path again. The agent references model
*roles* (planner, tool_caller, synthesizer, utility, judge) and the factory in `llms.py`
resolves each role to a concrete `(provider, model)` against the active hardware tier.

Nothing here imports from the rest of the project, so it is safe to import from anywhere
(no circular-import risk).

Live edits: `set(dotted_key, value)` mutates the in-memory config for the session (used by
the `/config` slash command). `reload()` re-reads the file from disk. Neither writes back to
`config.yaml`; persistence is a deliberate non-goal for the MVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Risk tiers, ordered low -> high. Shared with the approval gate (registry.risk_of returns
# one of these strings). A tool runs without prompting iff its tier <= the configured
# `runtime.auto_approve` tier.
RISK_ORDER = ["read_only", "side_effecting", "destructive"]

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_REPO_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class ModelSpec:
    """A resolved role binding: which provider serves which model id."""

    provider: str
    model: str


@dataclass(frozen=True)
class Capability:
    """What a model can do. The MVP requires tools + structured output for the roles that
    drive the loop; the factory warns when a bound model falls short."""

    supports_tools: bool = True
    supports_structured_output: bool = True
    context_window: int = 8192
    supports_vision: bool = False


class Config:
    """Thin wrapper over the parsed YAML dict with typed, role-aware accessors."""

    def __init__(self, data: dict):
        self._data = data

    # --- generic dotted access (used by /config) ---------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """Mutate the in-memory config (session-only; not written back to disk)."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)

    # --- tier / roles ------------------------------------------------------
    @property
    def active_tier(self) -> str:
        return self._data.get("active_tier", "workstation")

    def _tier(self) -> dict:
        tiers = self._data.get("tiers", {})
        tier = tiers.get(self.active_tier)
        if tier is None:
            raise KeyError(
                f"active_tier '{self.active_tier}' not found in config.yaml tiers: "
                f"{list(tiers)}"
            )
        return tier

    def model_for_role(self, role: str) -> ModelSpec:
        """Resolve a role to a concrete (provider, model). Falls back to the `utility`
        role, then to the first role defined, so a missing role never crashes the graph."""
        tier = self._tier()
        roles = tier.get("roles", {})
        entry = roles.get(role) or roles.get("utility")
        if entry is None and roles:
            entry = next(iter(roles.values()))
        if entry is None:
            raise KeyError(f"tier '{self.active_tier}' defines no roles")

        default_provider = tier.get("provider", "ollama")
        if isinstance(entry, dict):
            return ModelSpec(
                provider=entry.get("provider", default_provider),
                model=entry["model"],
            )
        return ModelSpec(provider=default_provider, model=str(entry))

    @property
    def embedder_model(self) -> str:
        return self._tier().get("embedder", "qwen3-embedding:8b")

    def capability_of(self, model: str) -> Capability:
        caps = self._data.get("capabilities", {})
        spec = caps.get(model)
        if not spec:
            return Capability()  # conservative defaults
        return Capability(
            supports_tools=spec.get("supports_tools", True),
            supports_structured_output=spec.get("supports_structured_output", True),
            context_window=spec.get("context_window", 8192),
            supports_vision=spec.get("supports_vision", False),
        )

    # --- runtime knobs -----------------------------------------------------
    @property
    def max_iterations(self) -> int:
        return int(self.get("runtime.max_iterations", 8))

    @property
    def auto_approve(self) -> str:
        tier = self.get("runtime.auto_approve", "read_only")
        return tier if tier in RISK_ORDER else "read_only"

    def auto_approves(self, risk: str) -> bool:
        """True if a tool of the given risk tier runs without prompting under the policy."""
        try:
            return RISK_ORDER.index(risk) <= RISK_ORDER.index(self.auto_approve)
        except ValueError:
            return False  # unknown risk -> fail safe (always prompt)

    # --- paths (resolved against the repo root) ----------------------------
    def path(self, name: str) -> Path:
        rel = self.get(f"paths.{name}")
        if rel is None:
            raise KeyError(f"paths.{name} not defined in config.yaml")
        return (_REPO_ROOT / rel).resolve()


def _coerce(value: Any) -> Any:
    """Coerce a string from the /config command into int/float/bool where it obviously is one."""
    if not isinstance(value, str):
        return value
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load() -> Config:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


# Module-level singleton — loaded once, shared everywhere.
_config = _load()


def get_config() -> Config:
    return _config


def reload() -> Config:
    """Re-read config.yaml from disk (used by /config reload)."""
    global _config
    _config = _load()
    return _config

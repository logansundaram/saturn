"""
Runtime configuration for Saturday.ai (Phase 3).

Loads `config.yaml` once and exposes it through a small typed accessor so the rest of the
codebase never hard-codes a model id or a filesystem path again. The agent references model
*roles* (planner, tool_caller, synthesizer, utility, judge) and the factory in `llms.py`
resolves each role to a concrete `(provider, model)` against the active hardware tier.

Nothing here imports from the rest of the project, so it is safe to import from anywhere
(no circular-import risk).

Live edits: `set(dotted_key, value)` mutates the in-memory config for the session (used by
the `/config` slash command). `reload()` re-reads the file from disk. `persist(dotted_key)`
writes a single scalar leaf back to `config.yaml` *in place* — preserving every comment and the
existing layout — so a `/config … --save` change survives a restart. (A full YAML rewrite would
shred the heavily-commented file; the surgical line edit keeps it intact.)
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Risk tiers, ordered low -> high. Shared with the approval gate (registry.risk_of returns
# one of these strings). A tool runs without prompting iff its tier <= the configured
# `runtime.auto_approve` tier.
RISK_ORDER = ["read_only", "side_effecting", "destructive"]


def _resolve_config_path() -> Path:
    """Locate the live config.yaml.

    Clone mode (the curl installers and manual installs): config.yaml sits next to this file
    at the repo root — the historical behavior, unchanged.

    Installed mode (pipx/uv/pip wheel): the user's editable copy lives under SATURDAY_HOME
    (default ~/.saturday), seeded on first run from the packaged default that the wheel ships
    to <venv>/share/saturn/ (see pyproject.toml). Keeping the live copy out of site-packages
    means `/config … --save` edits survive a `pipx upgrade`.
    """
    local = Path(__file__).parent / "config.yaml"
    if local.exists():
        return local
    home = Path(os.environ.get("SATURDAY_HOME") or Path.home() / ".saturday")
    user_cfg = home / "config.yaml"
    if not user_cfg.exists():
        default = Path(sys.prefix) / "share" / "saturn" / "config.yaml"
        if not default.exists():
            raise FileNotFoundError(
                "config.yaml not found: not running from a Saturn clone, and the packaged "
                f"default ({default}) is missing. Reinstall Saturn, or point SATURDAY_HOME "
                "at a directory containing a config.yaml."
            )
        home.mkdir(parents=True, exist_ok=True)
        shutil.copy(default, user_cfg)
    return user_cfg


_CONFIG_PATH = _resolve_config_path()
# Data root: every `paths.*` entry resolves against the directory holding the live config.yaml —
# the repo root in clone mode, SATURDAY_HOME for a wheel install. User data never lands in
# site-packages, where an upgrade would clobber it. (diag.py and env_keys.py mirror this lookup;
# they deliberately import nothing project-side, so keep the three in step.)
_REPO_ROOT = _CONFIG_PATH.parent

# THE five model roles the loop binds (config.yaml `roles:`, llms.get_model's vocabulary). One
# home so every surface that iterates roles (the readout commands, llms.check_models, the
# locality classifier behind the posture line) walks the SAME tuple — a role added to one stale
# copy would silently vanish from the others (e.g. a posture line claiming `all_local` without
# ever seeing the new binding).
MODEL_ROLES = ("planner", "tool_caller", "synthesizer", "utility", "judge")


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
            model = entry.get("model")
            if not model:
                raise KeyError(
                    f"role '{role}' on tier '{self.active_tier}' is a mapping without a "
                    f"'model' key: {entry!r}"
                )
            return ModelSpec(provider=entry.get("provider", default_provider), model=model)
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

    # --- context window (Ollama num_ctx) -----------------------------------
    @property
    def num_ctx_override(self) -> "int | None":
        """Session/config override for the Ollama context window (`runtime.num_ctx`), or None to
        let each model use its capability `context_window`. Set live by /config context."""
        v = self.get("runtime.num_ctx")
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def num_ctx_for(self, model: str) -> int:
        """Effective Ollama context window (`num_ctx`) for a model: the explicit
        `runtime.num_ctx` override if set, else the model's capability `context_window`. This is
        the number actually passed to ChatOllama and shown in the TUI gauge, so the displayed fill
        is truthful (Ollama otherwise silently defaults to 2048)."""
        return self.num_ctx_override or self.capability_of(model).context_window

    @property
    def llm_timeout(self) -> "float | None":
        """Read timeout (seconds) for a single local-model call (`runtime.llm_timeout`), or None to
        disable. Bounds a wedged Ollama daemon so a turn fails cleanly instead of hanging forever;
        set high enough not to false-trip a slow-but-healthy generation (connect is capped
        separately in llms._build)."""
        v = self.get("runtime.llm_timeout", 600)
        try:
            n = float(v)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

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


def _dump_scalar(value: Any) -> str:
    """Render a Python scalar back to its YAML literal for an in-place line edit. Quotes a string
    only when it would otherwise be misread (empty, leading/trailing space, YAML-significant
    punctuation, or a word like `true`/`null` that would round-trip as a non-string)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    bare_ok = (
        s
        and s.strip() == s
        and not any(c in s for c in ":#{}[],&*!|>%@`\"'")
        and s.lower() not in ("null", "~", "true", "false", "yes", "no", "on", "off")
    )
    if bare_ok:
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# A `key: …` mapping line. Stops the key at the first colon, so a deep quoted key like
# `"gemma4:e4b":` misparses — harmless, since persist only ever targets the shallow scalar leaves
# that appear before those blocks (it returns or raises before reaching them).
_YAML_KEY_LINE = re.compile(r"^(\s*)([^\s#][^:]*):(.*)$")


def _set_yaml_scalar(text: str, dotted_key: str, value: Any) -> str:
    """Return `text` (raw config.yaml) with the scalar at `dotted_key` replaced by `value`,
    preserving comments, indentation, and every unrelated line. Tracks nesting by indentation so a
    dotted path resolves to the right leaf. Raises KeyError if the key isn't present as an existing
    scalar leaf — including when it matches a bare `web:`-style SECTION HEADER, which is not a leaf
    (see the header guard below) — and ValueError if the in-memory value is a container (only
    scalars are persistable)."""
    if isinstance(value, (dict, list)):
        raise ValueError(f"{dotted_key} is a container, not a scalar — edit config.yaml by hand")

    target = dotted_key.split(".")
    lines = text.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []  # (indent, key) for the active nesting path

    for i, raw in enumerate(lines):
        body = raw.rstrip("\n").rstrip("\r")
        m = _YAML_KEY_LINE.match(body)
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2).strip()
        after = m.group(3)  # everything past the colon: " value  # comment"

        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))

        if [k for _, k in stack] != target:
            continue

        # Header guard: a bare `key:` whose value position is empty (an inline comment aside)
        # and whose next real line opens a more-deeply-indented block (or a block-sequence
        # item) is a SECTION HEADER, not a scalar leaf. Rewriting it to `key: value` while its
        # children stay behind yields unparseable YAML ("mapping values are not allowed here")
        # — and _load() runs at module import, so the NEXT launch of the app would die before
        # anything renders, leaving the user to hand-repair config.yaml. A bare `key:` with no
        # such follower is a genuinely-null scalar leaf and stays editable.
        content = after.strip()
        if not content or content.startswith("#"):
            for nxt in lines[i + 1:]:
                nxt_body = nxt.rstrip("\n").rstrip("\r")
                nxt_strip = nxt_body.strip()
                if not nxt_strip or nxt_strip.startswith("#"):
                    continue  # blank/comment lines say nothing about nesting
                nxt_indent = len(nxt_body) - len(nxt_body.lstrip())
                # A same-indent `- item` after a bare `key:` is that key's block sequence
                # (YAML allows sequence items at the parent key's indent) — a container too.
                if nxt_indent > indent or (
                    nxt_indent == indent
                    and (nxt_strip == "-" or nxt_strip.startswith("- "))
                ):
                    raise KeyError(
                        f"{dotted_key} is a section header, not an editable scalar leaf — "
                        "set one of its child keys instead"
                    )
                break

        ws = re.match(r"^(\s*)(.*)$", after)
        sep = ws.group(1) or " "
        cmt = re.search(r"(\s+#.*)$", ws.group(2))
        comment = cmt.group(1) if cmt else ""
        eol = raw[len(body):]  # keep the original line ending (\n / \r\n / none)
        lines[i] = f"{m.group(1)}{key}:{sep}{_dump_scalar(value)}{comment}{eol}"
        return "".join(lines)

    raise KeyError(f"{dotted_key} not found in config.yaml as an editable scalar")


def persist(dotted_key: str) -> Path:
    """Write the current in-memory value of `dotted_key` back to config.yaml in place (comments and
    layout preserved) so it survives a restart. Returns the config path. Covers the scalar leaves a
    user actually tunes — the runtime knobs, `active_tier`, the web/shell settings, the paths.
    Deeper structural edits (per-tier role bindings) belong in the file or `/models`."""
    value = get_config().get(dotted_key)
    # newline="" both ways: read_text's universal-newline mode would fold CRLF to \n before
    # _set_yaml_scalar captures each line's ending, and write_text would then re-expand every
    # \n to os.linesep — rewriting the WHOLE file's line endings on a one-line edit. Disabling
    # translation keeps the per-line eol capture honest and the diff to the single edited line.
    with open(_CONFIG_PATH, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    with open(_CONFIG_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(_set_yaml_scalar(text, dotted_key, value))
    return _CONFIG_PATH


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

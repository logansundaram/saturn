"""
The gate policy — ONE object behind every approval decision.

What used to be five separate gate-relaxation mechanisms are now views over this module
(the v1.x "policy-as-configuration" consolidation):

  runtime.auto_approve   the policy's tier threshold — tools AT OR BELOW it run without
                         prompting. The baseline lives in config.yaml like every other knob;
                         `tier()`/`set_tier()` are the read/write path (Shift+Tab cycles it).
  /autoapprove (/yolo)   a view that sets the threshold to `destructive` (gate fully open)
                         and restores the previous threshold on `off` — not a separate switch.
  --yolo (headless)      the same view, applied at process start.
  /risk                  edits a TOOL's tier (live in registry.TOOL_RISK; persisted here).
  /allow                 edits the run_shell prefix allowlist (persisted here).

Durable state is one small, versionable JSON file at `config.path("permissions")`
(database/permissions.json): `risk_overrides` ({tool: tier}, applied over declared tiers at
startup by registry.py) + `shell_allow` ([prefix, ...]). The tier threshold persists via
config.yaml (`/config --save runtime.auto_approve`), not here.

The one gate question is `approves(name, risk, args)` — the approval node asks it for every
tool call, and nothing else decides whether a call skips the human.

Prefix matching is deliberately strict: it is TOKEN-based ("git status" matches "git status
--short" but not "git statusx"), case-insensitive, and refuses to exempt any command containing
shell metacharacters (;, |, &, redirection, substitution, newlines). Without that refusal,
allowing "git status" would also wave through "git status; rm -rf ~" — the gate must fail closed
on anything it can't read at a glance. A background run_shell call (detached, timeout-free) is
never prefix-exempt either: the prefix was granted for a bounded foreground run, not a daemon.

Imports only config (a leaf), so registry.py, the approval node, and the TUI can import this
freely.
"""

from __future__ import annotations

import json
import re

from config import get_config, persist, RISK_ORDER

# Any of these in a command means it can do more than its first tokens say — chaining, piping,
# redirection, substitution. Such a command is never prefix-exempt; the human reads it at the gate.
_SHELL_META = re.compile(r"[;&|<>`$\n\r]")


# --- the tier threshold (runtime.auto_approve and its views) -----------------------------


def tier() -> str:
    """The effective auto-approve threshold: tools at or below it never prompt."""
    return get_config().auto_approve


def set_tier(new_tier: str, save: bool = False) -> str:
    """Set the threshold (session-scoped, like every cfg.set; `save` persists to config.yaml).
    Unknown tiers fail closed to read_only. Returns the tier actually set."""
    if new_tier not in RISK_ORDER:
        new_tier = "read_only"
    get_config().set("runtime.auto_approve", new_tier)
    if save:
        persist("runtime.auto_approve")
    return new_tier


def auto_approves(risk: str) -> bool:
    """True if a tool of the given risk tier runs without prompting under the current policy."""
    return get_config().auto_approves(risk)


# /autoapprove is not a sixth mechanism — it's this: threshold = destructive (every tier passes).
# Remember what the threshold was so `off` restores it instead of guessing.
_tier_before_gate_off: "str | None" = None


def gate_off() -> bool:
    """Whether the gate is fully open (threshold at `destructive` — nothing prompts)."""
    return tier() == "destructive"


def set_gate_off(off: bool) -> None:
    """The /autoapprove · --yolo view: open the gate by raising the threshold to `destructive`;
    close it by restoring the prior threshold (read_only if none was recorded — fail closed)."""
    global _tier_before_gate_off
    if off:
        if not gate_off():
            _tier_before_gate_off = tier()
        set_tier("destructive")
    else:
        set_tier(_tier_before_gate_off or "read_only")
        _tier_before_gate_off = None


# --- the one gate question ----------------------------------------------------------------


def approves(name: str, risk: str, args: "dict | None" = None) -> bool:
    """Whether a tool call runs WITHOUT facing the human. The approval node asks this for every
    pending call; the only two ways through are the tier threshold and (for run_shell only) a
    user-persisted /allow prefix on the exact command."""
    if auto_approves(risk):
        return True
    if name == "run_shell":
        call_args = args or {}
        # /allow prefixes were granted for ordinary foreground runs, bounded by shell.timeout.
        # background=True changes the semantics of the same command (detached, timeout-free job),
        # so it always faces the human regardless of any matching prefix.
        if call_args.get("background"):
            return False
        command = str(call_args.get("command", ""))
        return shell_allowed(command) is not None
    return False


# --- durable storage (database/permissions.json) -------------------------------------------


def _path():
    return get_config().path("permissions")


def _load() -> dict:
    """The stored policy file, with safe defaults when missing or unreadable."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("risk_overrides", {})
    data.setdefault("shell_allow", [])
    return data


def _save(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- risk-tier overrides (/risk … --save) -----------------------------------------------


def risk_overrides() -> dict:
    """{tool_name: tier} as persisted. Validation against the live registry happens at the
    call site (registry.py) — a stale name for a removed tool is simply ignored there."""
    return dict(_load()["risk_overrides"])


def set_risk_override(tool: str, tier: str) -> None:
    data = _load()
    data["risk_overrides"][tool] = tier
    _save(data)


def clear_risk_override(tool: str) -> bool:
    """Remove a persisted override; True if one was stored."""
    data = _load()
    if tool not in data["risk_overrides"]:
        return False
    del data["risk_overrides"][tool]
    _save(data)
    return True


# --- run_shell prefix allowlist (/allow) ------------------------------------------------


def shell_allow() -> list[str]:
    return list(_load()["shell_allow"])


def add_shell_allow(prefix: str) -> bool:
    """Store a prefix; False if it (case-insensitively) is already stored."""
    prefix = " ".join(prefix.split())
    data = _load()
    if any(p.lower() == prefix.lower() for p in data["shell_allow"]):
        return False
    data["shell_allow"].append(prefix)
    _save(data)
    return True


def remove_shell_allow(token: str) -> "str | None":
    """Remove a prefix by 1-based index or exact text; returns what was removed, or None."""
    data = _load()
    allow = data["shell_allow"]
    removed = None
    if token.isdigit() and 1 <= int(token) <= len(allow):
        removed = allow.pop(int(token) - 1)
    else:
        for i, p in enumerate(allow):
            if p.lower() == token.strip().lower():
                removed = allow.pop(i)
                break
    if removed is not None:
        _save(data)
    return removed


def shell_allowed(command: str) -> "str | None":
    """The allowlisted prefix that exempts `command` from the gate, or None.

    A command is exempt only when (a) it contains no shell metacharacters at all and (b) its
    leading whitespace-split tokens equal some stored prefix's tokens, case-insensitively. Token
    equality (not startswith) so "git status" never matches "git statusx"."""
    if _SHELL_META.search(command):
        return None
    cmd_tokens = [t.lower() for t in command.split()]
    if not cmd_tokens:
        return None
    for prefix in _load()["shell_allow"]:
        p_tokens = [t.lower() for t in prefix.split()]
        if p_tokens and cmd_tokens[: len(p_tokens)] == p_tokens:
            return prefix
    return None

"""
Persistent permissions — the user's durable approval decisions.

Two kinds of state, both stored in one small JSON file at `config.path("permissions")` (so they
survive a restart, unlike the session-only TOOL_RISK edits):

  risk_overrides  {tool_name: tier}  — persisted /risk changes, applied over the declared tiers
                                       at startup (registry.py).
  shell_allow     [prefix, ...]      — run_shell command prefixes the user has allowlisted via
                                       /allow; a matching command skips the approval gate.

Prefix matching is deliberately strict: it is TOKEN-based ("git status" matches "git status
--short" but not "git statusx"), case-insensitive, and refuses to exempt any command containing
shell metacharacters (;, |, &, redirection, substitution, newlines). Without that refusal,
allowing "git status" would also wave through "git status; rm -rf ~" — the gate must fail closed
on anything it can't read at a glance.

Imports only config (a leaf), so registry.py and the approval node can import this freely.
"""

from __future__ import annotations

import json
import re

from config import get_config

# Any of these in a command means it can do more than its first tokens say — chaining, piping,
# redirection, substitution. Such a command is never prefix-exempt; the human reads it at the gate.
_SHELL_META = re.compile(r"[;&|<>`$\n\r]")


def _path():
    return get_config().path("permissions")


def _load() -> dict:
    """The stored permissions, with safe defaults when the file is missing or unreadable."""
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

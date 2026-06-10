"""
Session token-budget accounting — the spend ceiling behind `runtime.token_budget`.

A trust-stack guard for cloud-bound tiers: `/trace cost` reports spend after the fact, but nothing
used to *enforce* a ceiling. This module keeps a process-wide running total of tokens observed on
the loop's LLM calls (input + output, fed by `agent_node` and `synthesize_node` via
`llms.extract_total_tokens`) and answers one question for the router: has the session spent its
budget? When it has, `route_after_agent` stops starting new tool rounds and wraps every turn up at
synthesize immediately — the same forced-landing shape as the `max_iterations` cap.

Accounting is deliberately best-effort: the planner/judge structured-output calls don't surface
usage metadata, so the total slightly undercounts. The agent + synthesizer calls it does count are
the heavy spenders, and a ceiling that undercounts by a few percent still does its job — this is a
cost guard, not a billing meter (`/trace cost` is the precise record).

The budget is per-process (one Saturn session). `runtime.token_budget` is read live from config on
every check, so `/config runtime.token_budget <n>` raises/clears it mid-session; 0 (the default)
disables enforcement entirely. Imports only config — a leaf, safe to import from any node.
"""

from __future__ import annotations

from config import get_config

# Tokens observed on this process's LLM calls so far (input + output, best-effort).
_spent: int = 0


def limit() -> int:
    """The configured session ceiling (`runtime.token_budget`), or 0 when disabled. Read live so
    a /config edit applies to the very next check."""
    v = get_config().get("runtime.token_budget", 0)
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def add(tokens: int) -> None:
    """Record tokens spent by one LLM call. Ignores junk so a provider that reports nothing
    (or garbage) can never corrupt the running total."""
    global _spent
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return
    if n > 0:
        _spent += n


def spent() -> int:
    """Tokens observed so far this session."""
    return _spent


def remaining() -> "int | None":
    """Tokens left under the ceiling (never negative), or None when no budget is set."""
    cap = limit()
    if not cap:
        return None
    return max(0, cap - _spent)


def exceeded() -> bool:
    """True when a budget is set and the session has spent it."""
    cap = limit()
    return bool(cap) and _spent >= cap


def near(fraction: float = 0.8) -> bool:
    """True when a budget is set and spend has crossed `fraction` of it (the soft-warning line)."""
    cap = limit()
    return bool(cap) and _spent >= cap * fraction


def reset() -> None:
    """Zero the running total. Not called by the app (the budget is a per-process ceiling by
    design); exists for tests and for a deliberate operator reset via the REPL."""
    global _spent
    _spent = 0

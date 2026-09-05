"""
The idle prefix primes (2026-09-04): between turns, re-send each node lineage's STABLE prefix —
its system prompt plus the stable half of the grounding — so the daemon holds a context
checkpoint at that message boundary and the next turn's call prefills only what is new.

Why a prime rather than the previous turn's own prompt: llama-server (Ollama 0.33, qwen3.5 —
a hybrid/recurrent model, so no partial KV reuse) restores a prompt only up to a saved
checkpoint, and it saves them 1024 tokens before a prompt's end and 4 tokens before it, never
at message boundaries. A prompt that differs from its predecessor inside the last 1024 tokens
therefore reprocesses ~1024 tokens at best; a prompt that EXTENDS a cached prompt at a message
boundary restores that prompt's N-4 checkpoint and reprocesses only the extension (measured:
5083 tokens / 12.7 s → 20 tokens / 0.2 s for the plan call). The prime is that cached prompt:
`[system][user: stable grounding]`, one predicted token, think ON (think off appends the empty
think block after the boundary and pushes N-4 past it — measured, tests pin the flag), the
same load options as every turn (num_ctx, the runner options) so it never reloads the model.

When: after the weights load at startup (the planner lineage only — the one the first call
needs; the others would queue the user's first request behind ~15 s of cold prefill each) and
after every turn (every lineage, ~0.2 s each once warm, rebuilt from disk so a write this turn
is already in the manifest the next turn's grounding renders). Never during a turn: the REPL
marks the turn busy and the sequence stops before its next request. Never in headless mode
(one turn per process) and never under tests (conftest disables it — no test may reach a
model). Never raises; a down daemon is logged and counted as zero.
"""

from __future__ import annotations

import threading

import diag

# Flipped off by tests/conftest.py (autouse) — the only production value is True.
ENABLED = True

_busy = threading.Event()


def _config_enabled() -> bool:
    try:
        from config import get_config

        return bool(get_config().get("runtime.prime", True))
    except Exception:
        return True


def set_busy(busy: bool) -> None:
    """The REPL's turn-in-progress flag: a prime sequence never starts its next request while a
    turn runs (the daemon serves one request at a time, so it would queue the turn's call)."""
    if busy:
        _busy.set()
    else:
        _busy.clear()


def lineages(stable: str) -> list:
    """`(role, task, messages)` per prompt lineage, each ending at the message boundary the
    node's real prompt extends: the planner (nodes/plan.py), the tool-step and reasoning-step
    executes (nodes/execute.py), the answer (nodes/synthesize.py). The user message must be
    BYTE-IDENTICAL to the node's own first user message — the tests pin each pairing."""
    from langchain.messages import HumanMessage

    from core.messages import (
        EXECUTE_REASONING_SYS,
        EXECUTE_TOOL_SYS,
        planner_sys_msg,
        synthesize_sys_msg,
    )

    return [
        ("planner", [planner_sys_msg(), HumanMessage(content="Grounding context:\n" + stable)]),
        ("tool_caller", [EXECUTE_TOOL_SYS, HumanMessage(content=stable)]),
        ("synthesizer", [synthesize_sys_msg, HumanMessage(content="Relevant context:\n" + stable)]),
        ("tool_caller", [EXECUTE_REASONING_SYS, HumanMessage(content=stable)]),
    ]


def prime(stable: str, only: "tuple | None" = None) -> int:
    """Send the prime for each lineage (or for the roles in `only`, in lineage order); returns
    how many were sent. Stops early when a turn starts. Never raises."""
    if not ENABLED or not stable:
        return 0
    from core.llms import generate, get_model
    from core.structured import _invoke_kwargs, _model_tag

    sent = 0
    for role, messages in lineages(stable):
        if only is not None and role not in only:
            continue
        if _busy.is_set():
            diag.log(f"prime: turn in progress — stopping after {sent} lineage(s)")
            break
        try:
            # The planner task's kwargs for EVERY lineage: think ON (see the module docstring)
            # at the shared num_ctx/runner options; one token so the request is all prefill.
            kwargs = _invoke_kwargs(role, None, 0.0, task="plan")
            kwargs.setdefault("options", {})["num_predict"] = 1
            generate(get_model(role), messages, tag=_model_tag(role), **kwargs)
            sent += 1
        except Exception as exc:
            diag.log(f"prime: {role} lineage skipped ({exc})")
            break  # a down daemon fails every lineage the same way
    return sent


def prime_now(only: "tuple | None" = None) -> int:
    """Rebuild the stable grounding from disk and prime with it. Never raises."""
    if not ENABLED or not _config_enabled():
        return 0
    try:
        from nodes.ground import stable_grounding

        stable = stable_grounding()
    except Exception as exc:
        diag.log(f"prime: grounding rebuild failed ({exc})")
        return 0
    return prime(stable, only)


def start_priming(only: "tuple | None" = None) -> "threading.Thread | None":
    """`prime_now` on a daemon thread, or None when priming is off (tests, runtime.prime)."""
    if not ENABLED or not _config_enabled():
        return None
    t = threading.Thread(target=prime_now, args=(only,), name="prefix-prime", daemon=True)
    t.start()
    return t

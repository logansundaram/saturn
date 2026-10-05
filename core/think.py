"""
Adaptive thinking — which passes think (spec:
docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md).

The harness decides, per pass, whether the model reasons before it answers: Qwen3.5 has no
in-model switch, so the decision is made here, from the KIND OF STEP the pass is — a pure
function of this turn's messages. No model call, no reading of the request text: the decision
is deterministic, so it can be tested, shown on the rail and explained by `/think`.

    level   the user's dial: fast (never) · auto (think before acting) · deep (every
            uncapped pass)
    kind    what the pass is reacting to (step_kind)

`auto` is ONE rule, chosen by measurement (2026-10-04, loop benchmark, 4b and 9b): think
before a pass ACTS, never before a text answer. Every deciding pass is DRAFTED think-off; a
text answer stands at no cost, a tool call is retracted and the pass rethought. After an error
or a steer the pass thinks outright. `recover` — the rule before it (think only after an
error) — is kept as the benchmark's baseline (`benchmark.py --think recover`), not a setting.

`nodes/agent.py` asks `decide` once per pass and records one `entry` per pass in
`state["think"]`; the rail, `/think`, `/trace why` and the loop benchmark read those entries.
"""

from __future__ import annotations

from typing import NamedTuple

from langchain.messages import AIMessage, ToolMessage

from config import get_config
from core.state import is_steer_message

LEVELS = ("fast", "auto", "deep")
POLICIES = ("act", "recover")  # what auto does · the baseline the benchmark compares it to
KINDS = ("capped", "recovery", "steered", "first", "wrap-up", "information")
OUTCOMES = ("none", "thought", "empty", "cut-budget", "cut-esc", "malformed", "unsupported")

# The old spellings (and what YAML makes of a bare `on` / `off`: a boolean).
_LEVEL_WORDS = {"fast": "fast", "off": "fast", "false": "fast",
                "auto": "auto", "adaptive": "auto",
                "deep": "deep", "on": "deep", "true": "deep"}

# The most tokens one thought may spend (runtime.think_budget): twice the longest thought the
# loop benchmark has produced (509 tokens, 4b, 2026-10-04).
BUDGET = 1024

# How a kind reads to the user — the rail leaf, /think and /trace why share these words.
KIND_WORDS = {"capped": "budget spent", "recovery": "after an error", "steered": "you steered",
              "first": "first move", "wrap-up": "wrap-up", "information": "new information"}


def normalise(raw) -> "tuple[str, bool]":
    """(level, recognised) for a `runtime.think` value. An absent key is `auto`; anything that
    is not a known word runs as `auto` and is reported once at startup (`problems`)."""
    if raw is None:
        return "auto", True
    if isinstance(raw, bool):
        return ("deep" if raw else "fast"), True
    word = _LEVEL_WORDS.get(str(raw).strip().lower())
    return (word, True) if word else ("auto", False)


def is_level_word(word) -> bool:
    """Whether `word` names a level (either spelling) — `/think`'s level-word rule."""
    return str(word or "").strip().lower() in _LEVEL_WORDS


def level(state=None) -> str:
    """This pass's level: the turn's own override (`/think <request>` sets
    `state["think_level"]`), else `runtime.think`."""
    override = str((state or {}).get("think_level") or "").strip().lower()
    if override in LEVELS:
        return override
    return normalise(get_config().get("runtime.think"))[0]


_POLICY = "act"


def policy() -> str:
    """What `auto` does: `act`. Not a setting — `set_policy` exists for the loop benchmark's
    baseline run and for tests."""
    return _POLICY


def set_policy(name: str) -> str:
    """Point `auto` at another policy for THIS process (benchmark.py --think recover)."""
    global _POLICY
    if name not in POLICIES:
        raise ValueError(f"{name!r} is not one of {' | '.join(POLICIES)}")
    _POLICY = name
    return name


def budget() -> int:
    """The most reasoning tokens one thought may spend (`runtime.think_budget`)."""
    try:
        return max(0, int(get_config().get("runtime.think_budget", BUDGET) or 0))
    except (TypeError, ValueError):
        return BUDGET


def problems() -> "list[str]":
    """The startup lines for a think setting that is not one of its values — the setting runs
    as its default, and says so, instead of silently meaning something else."""
    out = []
    raw = get_config().get("runtime.think")
    if not normalise(raw)[1]:
        out.append(f"runtime.think is {raw!r} — not one of fast | auto | deep; running as auto")
    return out


def supported() -> bool:
    """Whether the bound model can be asked to think. False once the daemon has rejected the
    think flag for this tag (learned on the first rejection, core/llms)."""
    from core import llms

    return llms.model_tag() not in llms._NO_THINK_SUPPORT


# ── the kind of step ─────────────────────────────────────────────────────────────────────────


def _status(m) -> str:
    return (getattr(m, "additional_kwargs", None) or {}).get("saturn_status") or "done"


def latest_round(this_turn: list) -> list:
    """The ToolMessages answering the last tool-calling message this turn — the round the
    coming pass reacts to. A steer note the user typed after it is skipped, not a boundary."""
    out = []
    for m in reversed(this_turn):
        if isinstance(m, ToolMessage):
            out.append(m)
        elif is_steer_message(m):
            continue
        else:
            break
    return out


def _steered(this_turn: list) -> bool:
    """A steer note arrived after the turn's last agent message."""
    for m in reversed(this_turn):
        if is_steer_message(m):
            return True
        if isinstance(m, AIMessage):
            return False
    return False


def step_kind(this_turn: list, capped: bool, mechanical: tuple = ()) -> str:
    """What the coming pass is reacting to — first match wins:

      capped       the pass is at or past runtime.max_iterations
      recovery     the latest tool round has an error: a tool failure or a hygiene refusal.
                   `mechanical` names refusal texts that are not evidence (ask_user's
                   "ask first": nothing failed, the sibling call is simply re-issued)
      steered      the user typed a correction after the turn's last agent message
      first        no tool round yet this turn
      wrap-up      the round had no error and every completed call was an action (a write, a
                   send, a move), or nothing completed (all declined or blocked): what is left
                   is to report. A prediction, not a restriction — the pass can still call
      information  anything else: a read, a search, a plan, an ask_user answer came back
    """
    if capped:
        return "capped"
    rnd = latest_round(this_turn)
    if any(_status(m) == "error" and str(m.content) not in mechanical for m in rnd):
        return "recovery"
    if _steered(this_turn):
        return "steered"
    if not any(isinstance(m, ToolMessage) for m in this_turn):
        return "first"
    from tools.registry import is_action

    completed = [m for m in rnd if _status(m) == "done"]
    if all(is_action(str(getattr(m, "name", "") or "")) for m in completed):
        return "wrap-up"
    return "information"


# ── the decision ─────────────────────────────────────────────────────────────────────────────


class Decision(NamedTuple):
    """`think`: the pass's call thinks. `draft`: the pass runs think-off first and is rethought
    only if that draft calls a tool. Never both. `why`: the words for the rail."""

    think: bool
    draft: bool
    why: str


# What `auto` does per kind of step: (kinds that THINK outright, kinds that are DRAFTED — run
# think-off first and rethought only if the draft calls a tool).
#   act      think before acting, never before a text answer. Measured 2026-10-04: 29.5 of 34
#            loop tasks on the 4b against 23.5 for `recover` (two runs each), 31 against 30 on
#            the 9b (one run), with no empty thoughts — every empty thought the other
#            candidates produced was a turn's final answer, which `act` never thinks on.
#   recover  the rule shipped 2026-09-29: only the pass right after an error thinks.
# Three other candidates were measured and dropped (docs/engine.md item 11): thinking on every
# deciding pass (`decide`), that with the first move drafted (`decide-draft`), and thinking on
# the first move only (`first`).
_AUTO = {
    "act": ({"recovery", "steered"}, {"first", "information", "wrap-up"}),
    "recover": ({"recovery"}, set()),
}


def decide(level: str, kind: str, policy: "str | None" = None, supported: bool = True) -> Decision:
    """Whether this pass thinks. The capped pass never does (its job is the answer); a model
    that rejects the think flag never does; `fast` never does; `deep` thinks on every other
    pass; `auto` thinks before acting: a kind either thinks outright, is DRAFTED (think-off
    first, rethought only if the draft calls a tool — a text answer costs nothing), or does
    neither. `policy` defaults to `policy()`."""
    policy = policy or _POLICY
    why = KIND_WORDS.get(kind, kind)
    if kind == "capped" or not supported or level == "fast":
        return Decision(False, False, why)
    if level == "deep":
        return Decision(True, False, why)
    thinks, drafts = _AUTO.get(policy, _AUTO["act"])
    if kind in drafts:
        return Decision(False, True, why)
    return Decision(kind in thinks, False, why)


# ── the record ───────────────────────────────────────────────────────────────────────────────

_TEXT_CAP = 400  # the thought's opening, for the rail leaf; the whole thought is in llm_calls


def entry(*, n: int, kind: str, decision: Decision, outcome: str = "none", draft: bool = False,
          thought: "dict | None" = None, prompt_s: float = 0.0) -> dict:
    """One pass's think record (`state["think"]`, plain dict — the checkpointer round-trips
    nothing else). `asked`: a thinking call was made. `draft`: a think-off draft was discarded
    for the thought. `outcome`: what came of it (OUTCOMES)."""
    thought = thought or {}
    return {"pass": n, "kind": kind, "why": decision.why,
            "asked": outcome not in ("none", "unsupported"), "draft": bool(draft),
            "outcome": outcome,
            "seconds": round(float(thought.get("seconds") or 0.0), 2),
            "tokens": int(thought.get("tokens") or 0),
            "text": " ".join(str(thought.get("text") or "").split())[:_TEXT_CAP],
            "prompt_s": round(float(prompt_s or 0.0), 3)}


def describe(e: dict) -> str:
    """One entry in the user's words — `/think`, `/trace why` and the rail share it:
    `first move · thought 1.8s`, `wrap-up · no thought`, `after an error · thought cut at the
    budget — answered without it`."""
    why = str(e.get("why") or KIND_WORDS.get(str(e.get("kind")), e.get("kind") or "?"))
    outcome = e.get("outcome")
    if outcome == "thought":
        said = f"thought {float(e.get('seconds') or 0.0):.1f}s"
        if e.get("draft"):
            said += " (rethought a drafted call)"
    elif outcome == "empty":
        said = "thought came back empty — answered without it"
    elif outcome == "cut-budget":
        said = f"thought cut at {int(e.get('tokens') or 0)} tokens — answered without it"
    elif outcome == "cut-esc":
        said = "thought stopped by Esc — answered without it"
    elif outcome == "malformed":
        said = "the thinking call's output was malformed"
    elif outcome == "unsupported":
        said = "this model has no thinking mode"
    else:
        said = "no thought"
    return f"{why} · {said}"

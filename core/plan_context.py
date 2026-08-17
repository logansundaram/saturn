"""
Plan-as-data-bus context builders (transplanted from the agentic_benchmark harness, 2026-07-03).

The engine's LLM calls never see the raw message history — each one gets a CURATED context built
from the plan's own step results: the user's request, a compact results block, an explicit
"the previous step" callout, and the current step. This is what keeps small local models on
task: the context is exactly the data the step needs, in a stable shape, with nothing stale.

All helpers are pure over the plain step dicts ({step_id, label, status, intended_tool, result,
needs_resolution} — see core/state.py) so they are trivially testable offline.
"""

from __future__ import annotations

import re

from config import get_config
from textutil import head_tail

# Cap each earlier result inside the results block. The full observation still lives on the step
# (and in tool_results for synthesize's numbered sections); this bound keeps a per-step context
# from re-sending every prior read in full.
_RESULT_CAP = 800

# The "previous step" callout carries more of its result than the block (it is the referent of
# "the previous step's result" in step labels), but still bounded — a ~12k clamped observation
# must not ride every per-step prompt in full.
_CALLOUT_CAP = 4000

# THE engine-wide tool classifications (one home — execute's write gate, rectify's resolution
# exemption, and synthesize's write verification all key off these; a copy per node is how a
# new tool silently escapes one of the three).
WRITE_TOOLS = ("write_file", "edit_file")
SEARCH_TOOLS = {"search_knowledge_base", "search_files", "find_files", "web_search"}

# Two questions about computation, deliberately NOT one tuple:
#   COMPUTE_TOOLS        — "did this turn compute anything?" (rectify's gap check). A shell
#                          one-liner is a legitimate way to do arithmetic, so it counts.
#   DERIVED_FIGURE_TOOLS — "which figures does the answer OWE?" (synthesize's computed-value
#                          check). A shell step's output is not a computation: `ls -l` yields file
#                          sizes, a digest yields hash fragments, and both parse as figures — so
#                          counting run_shell here made the answer obliged to state them, spent a
#                          corrective regeneration steering toward them, and then disclosed them
#                          under "from the plan's own calculation step".
COMPUTE_TOOLS = ("calculate", "run_shell")
DERIVED_FIGURE_TOOLS = ("calculate",)


def original_request(state) -> str:
    """The user's request as the engine's prompts see it — the current turn's query."""
    return str(state.get("current_query") or "")


# --- targets: what a request / a step is ABOUT (transplanted from the engine isolate) ---------
#
# THE one notion of a workspace target for the engine's deterministic checks (rectify's coverage
# branch; the plan-review revocation lock). Deliberately syntactic: a slash-joined path, or a
# name with a file extension — never a stat, never a guess.
_PATH_RE = re.compile(r"[A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)+|[A-Za-z0-9_\-]+\.[A-Za-z]{2,8}")


def target_tokens(text) -> set:
    """The workspace paths named anywhere in `text`, lowercased."""
    return {m.group(0).strip(".").lower() for m in _PATH_RE.finditer(str(text or ""))}


# --- the plan-review revocation lock + effect authorization (from the engine isolate) ---------
#
# Revocation: what removing an un-run step at plan review REVOKES, as targets — the label is a
# sentence a redraft can reword; the target is what the user actually removed, and `execute`
# refuses any state-changing action that lands on one for the rest of the turn (on the label
# before a generation is spent, on the generated ARGUMENTS at the last point before the call is
# emitted). Authorization: a state-changing step drafted AFTER results exist may only act on a
# target the user's own words name — results may inform HOW the request is carried out, never
# WHAT is done to the workspace.

REVOKE_ALL = "*"
ORIGIN_REPLAN = "replan"

# The effect class that MINTS `REVOKE_ALL` — and therefore the exact class it covers. A filesystem
# or shell write is the one effect a redraft can walk around by omitting the filename from its
# wording, which is what the blanket sentinel is for; a non-path effect is revoked as
# `tool:<name>` instead. Keeping the two symmetric is what stops one vague dropped write from
# cancelling every unrelated effect left in the plan (a `remember`, a `mcp_*` call the user
# deliberately KEPT) — those still face their own gate.
_FS_EFFECT_TOOLS = WRITE_TOOLS + ("run_shell",)


def state_changing(tool) -> bool:
    """Whether `tool` can change state — REGISTRY-derived: any tier above read_only (write/edit,
    run_shell, remember, every MCP tool; an unknown name fails closed to destructive). Never a
    hand list a new tool silently escapes.

    Read from the DECLARED tiers, frozen at definition time — NEVER the live `TOOL_RISK`, which
    the gate's always-allow grant and `/policy risk` both edit. Those answer "does this call need
    a prompt"; this answers "can this tool change state at all", and the two must not be the same
    question: keyed on the live tier, one `a` keypress at an unrelated gate prompt drops
    write_file to read_only and silently switches off BOTH the revocation lock and effect
    authorization for the rest of the turn — and, under `grant_scope: persist`, for every session
    after it. Relaxing a prompt threshold is not a statement about what a tool does."""
    if not tool:
        return False
    try:
        # lazy: plan_context is imported by the registry's users
        from tools.registry import DECLARED_RISK
    except Exception:
        return str(tool) in _FS_EFFECT_TOOLS
    return DECLARED_RISK.get(str(tool), "destructive") != "read_only"


def revoked_targets(step) -> set:
    """What removing this un-run step revokes, as target tokens. ONLY a state-changing step
    revokes anything (removing a read is not removing an effect). A state-changing TOOL revokes
    the paths its label names — or REVOKE_ALL for a filesystem/shell effect that names none (the
    conservative reading a redraft cannot walk around by omitting the filename), or `tool:<name>`
    for a non-path effect (a dropped `remember` revokes further memory writes, not every write).
    A write folded into a read-only step's DESCRIPTION ("…and save it to x.txt") is trusted for
    the paths it names and nothing more."""
    from core.request_intent import wants_state_change

    tool = step.get("intended_tool")
    label = str(step.get("label") or "")
    if state_changing(tool):
        named = target_tokens(label)
        if named:
            return named
        return {REVOKE_ALL} if (tool in WRITE_TOOLS or tool == "run_shell") else {f"tool:{tool}"}
    if wants_state_change(label):
        return target_tokens(label)
    return set()


def _same_target(revoked: str, named: str) -> bool:
    """Whether a path an action names IS the revoked target: the same token, or the same file
    reached by a longer/shorter path — compared as whole path SEGMENTS, never substrings
    (revoking out.txt must not refuse handout.txt)."""
    return revoked == named or named.endswith("/" + revoked) or revoked.endswith("/" + named)


def revocation_kind(revoked, tool, *texts) -> "str | None":
    """WHY a state-changing action with `tool` over `texts` (label, generated arguments) is
    refused — or None when nothing the user removed covers it. Read-only tools are never revoked.

    THE one producer of the distinction, because the three kinds are three different statements
    about the user, and `execute` stamps one of them onto the refused step:
      "target"  — they removed a step naming this path.
      "tool"    — they removed this tool's only effect (a dropped `remember`).
      "blanket" — they removed a filesystem/shell write that named NO file, so no file write runs
                  this turn. This action is collateral of that veto, not the step they deleted —
                  a disclosure claiming otherwise asserts an intent the state never recorded.
    """
    if not revoked or not state_changing(tool):
        return None
    revoked = [str(r).lower() for r in revoked]
    if f"tool:{tool}".lower() in revoked:
        return "tool"
    named = target_tokens(" ".join(str(t or "") for t in texts))
    if any(_same_target(r, n) for r in revoked if r != REVOKE_ALL for n in named):
        return "target"
    if REVOKE_ALL in revoked and str(tool) in _FS_EFFECT_TOOLS:
        return "blanket"
    return None


def is_revoked(revoked, tool, *texts) -> bool:
    """Whether a state-changing action with `tool` over `texts` (label, generated arguments)
    lands on something the user revoked at plan review. Read-only tools are never revoked."""
    return revocation_kind(revoked, tool, *texts) is not None


def request_authorized(state, step, *texts) -> bool:
    """Whether a state-changing action from `step` is authorized by the user's own words. A step
    drafted before any result existed (no `origin: replan`), or one that cannot change state, is
    authorized — nothing had been read, so nothing could have steered it. A request that asks
    for NO workspace change authorizes none ("tell me the late fee" cannot license a write,
    whatever the file being read says). A request that asks for a change and NAMES the target
    authorizes exactly that target (whole path segments); one that names no path ("save the
    summary somewhere") authorizes the model's choice — the deliberate residual. The authorizing
    text is `authorization_basis` (request + this turn's steers)."""
    from core.request_intent import wants_state_change

    if step.get("origin") != ORIGIN_REPLAN or not state_changing(step.get("intended_tool")):
        return True
    request = authorization_basis(state)
    if not wants_state_change(request):
        return False
    requested = target_tokens(request)
    if not requested:
        return True
    named = target_tokens(" ".join([str(step.get("label") or "")] + [str(t or "") for t in texts]))
    return any(_same_target(r, n) for r in requested for n in named)


def authorization_basis(state) -> str:
    """What the HUMAN typed this turn — the request plus every mid-turn steering correction
    (plan_gate records a steer as a HumanMessage: merged onto the turn's message or standalone
    with STEER_PREFIX). The one text the deterministic completeness/authorization checks read
    targets from: reading them from RESULTS would let text inside a file or a web page
    manufacture work the engine then demands of itself — the injection hazard the engine exists
    to refuse. A steer carries no such hazard, and reading the request alone would let a target
    the user named mid-turn never register."""
    from core.state import is_turn_start  # lazy: core.state imports nothing from here

    parts = [original_request(state)]
    this_turn: list = []
    for m in reversed(state.get("messages") or []):
        this_turn.append(m)
        if is_turn_start(m):
            break
    for m in reversed(this_turn):
        if getattr(m, "type", "") == "human":
            text = str(getattr(m, "content", "") or "")
            if text and text != parts[0]:
                parts.append(text)
    return "\n".join(parts)


def vetoes_block(state) -> str:
    """The user's plan-review vetoes (state["plan_vetoes"], written by plan_gate) as a prompt
    block — '' when there are none. THE one framing both the rectify judge and the replanner
    receive: work the human explicitly removed at the plan-review editor is deliberately out of
    scope, and the plan must never be changed or extended to reinstate it — the human's edit
    outranks the engine's self-correction (the gate's guarded-outcome principle, applied to
    review edits)."""
    vetoes = [str(v).strip() for v in (state.get("plan_vetoes") or []) if str(v).strip()]
    if not vetoes:
        return ""
    return (
        "The user EDITED the plan at the plan-review prompt and REMOVED these steps — they are "
        "deliberately out of scope for this turn at the user's own request. Do NOT change or "
        "extend the plan to reinstate them (or equivalent work), and do not treat their absence "
        "as a gap in the plan:\n" + "\n".join(f"- {v}" for v in vetoes)
    )


def clean(text) -> str:
    """Normalize an observation before it lands on a step: absolute workspace paths (run_shell
    output routinely embeds them) collapse to workspace-relative so prompts and the rendered
    plan stay readable and machine-independent. Best-effort; unknown shapes pass through."""
    s = str(text)
    try:
        raw = str(get_config().path("workspace"))
    except Exception:
        return s
    for form in {raw, raw.replace("\\", "/")}:
        if form:
            s = s.replace(form + "/", "").replace(form + "\\", "").replace(form, "workspace")
    return s


def steps_before(plan, step) -> list:
    """Steps strictly BEFORE `step` in plan order (identity match). A LATER step can carry a
    result too — plan-review retirement stamps one onto a step the user skipped — so "has a
    result" does NOT mean "ran earlier": positional slicing is the correct prior-work filter.
    A step not found in the plan (defensive) yields the whole plan, the pre-fix behavior."""
    for i, s in enumerate(plan or []):
        if s is step:
            return list(plan[:i])
    return list(plan or [])


def results_block(plan) -> str:
    """The 'Results from earlier steps' block: every completed step's label -> result (capped),
    numbered in plan order. Empty string when nothing has run."""
    done = [s for s in plan or [] if s.get("result") is not None]
    if not done:
        return ""
    lines = ["Results from earlier steps (use these exact values):"]
    for i, s in enumerate(done, 1):
        r = str(s.get("result") or "").strip()
        if len(r) > _RESULT_CAP:
            r = r[:_RESULT_CAP] + " …(truncated)"
        lines.append(f"{i}. {s.get('label')} -> {r}")
    return "\n".join(lines)


def exec_context(state, step) -> str:
    """The curated context for executing ONE step: request + earlier results + an explicit
    'the previous step' callout (the referent of 'the previous step's result' in step labels)
    + the current step. The grounding context rides along so workspace manifests / attachments /
    memory stay visible without the raw history."""
    parts = [f"User's overall request: {original_request(state)}"]
    grounding = str(state.get("context") or "").strip()
    if grounding:
        parts.append(grounding)
    plan = state.get("plan") or []
    # Only steps BEFORE the current one, in plan order: a LATER step can already carry a result
    # (the user retired it at plan review), and prior[-1] over the whole plan would present its
    # retirement stamp as "the previous step's result" — the model would compute from the stamp
    # text instead of the real preceding step's output.
    before = steps_before(plan, step)
    prior = [s for s in before if s.get("result") is not None]
    block = results_block(before)
    if block:
        parts.append(block)
    # The callout referent is the nearest prior step that PRODUCED a result (status done): an
    # incident/review-retired step's stamp is not "the previous step's result" — and replan's
    # done-first merge can reposition a retired step directly before the redrafted ones, so
    # position alone isn't enough.
    producers = [s for s in prior if s.get("status") == "done"]
    if producers:
        last = producers[-1]
        parts.append(
            f'The immediately preceding step ("the previous step") was: '
            f"{last.get('label')}\n  its result: "
            f"{head_tail(str(last.get('result') or '').strip(), _CALLOUT_CAP)}"
        )
    parts.append(f"Your current step: {step.get('label')}")
    return "\n\n".join(parts)


def plan_txt(plan) -> str:
    """The whole plan as text for the rectify/replan prompts: DONE steps with their results
    (capped like the results block — several ~12k clamped observations would otherwise overflow
    a small model's window and front-truncate the very system prompt the call depends on),
    PENDING steps with their intended tool."""
    lines = []
    for i, s in enumerate(plan or [], 1):
        tool = s.get("intended_tool") or "none"
        if s.get("result") is None:
            lines.append(f"{i}. [PENDING] tool={tool} | {s.get('label')}")
        else:
            r = str(s.get("result") or "").strip()
            if len(r) > _RESULT_CAP:
                r = r[:_RESULT_CAP] + " …(truncated)"
            lines.append(
                f"{i}. [DONE] tool={tool} | {s.get('label')}\n   result: {r}"
            )
    return "\n".join(lines)

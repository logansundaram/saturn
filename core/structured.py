"""
Hardened structured-output layer for the plan/execute/rectify engine (transplanted from the
agentic_benchmark harness, 2026-07-03).

Small local models mis-handle the full Pydantic JSON schema (`$ref`/`$defs`) that
`.with_structured_output` sends, and intermittently wrap their JSON in prose. This layer is the
defensive plumbing around every judgment call the engine makes:

  - FLAT, hand-written JSON schemas (the `*_FORMAT` dicts) constrain Ollama's decoder without
    `$ref` indirection, plus a one-line JSON "shape" hint appended as a system message so a model
    that ignores the grammar still sees the exact expected spelling.
  - `_extract_json` salvages the outermost `{...}` from prose-wrapped output.
  - LENIENT parse models (`_PlanOut`, `RectifyBool`, `ResolutionCheck`, `WriteGate`) with
    defaults, so a missing field degrades instead of raising.
  - temperature-escalating retries (0.0 first for determinism, then sampled variety), and a
    caller-supplied `default` so a total parse failure degrades to a safe verdict instead of
    aborting the turn.

Every call goes through `core.llms.get_model(role)`, so the trust boundary is preserved: a
cloud-bound (or remote-Ollama) role is redacted + recorded to the egress ledger, and the air-gap
guard applies. Constrained decoding (`format=`) and per-attempt temperature are passed only to
Ollama-served roles — other providers get the shape hint + salvage path alone.
"""

from __future__ import annotations

from typing import List, Optional

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field, ValidationError

import diag
from config import get_config


# ── lenient parse models ──────────────────────────────────────────────────────────────────────
# Boundary-only Pydantic (gotcha #4 still holds: the plan lives in state as plain dicts).


class _PlanItem(BaseModel):
    description: str = ""
    tool: Optional[str] = None
    needs_resolution: bool = False


class _PlanOut(BaseModel):
    plan: List[_PlanItem] = []


class RectifyBool(BaseModel):
    reasoning: str = Field(default="", description="why the plan is or isn't correct")
    rectify: bool = Field(default=False, description="True if the plan must change/extend")


class ResolutionCheck(BaseModel):
    """Whether the gathered results actually contain the item a deferred step refers to.
    `evidence` precedes `found` so the constrained decoder forces the model to quote its source
    before committing to the boolean — a chain-of-thought the grammar actually enforces."""

    evidence: str = ""
    found: bool = True  # fail-open: a parse failure must not cancel a legitimate plan


class WriteGate(BaseModel):
    """Whether the value a write step wants to persist is actually present in trusted sources
    (the request itself, or the gathered results). Same evidence-first design as above."""

    evidence: str = ""
    # fail-CLOSED: the gate is only ARMED when a search ran or a step failed, i.e. exactly when a
    # value could be getting bridged in from somewhere it wasn't verified — so an unparseable /
    # missing verdict must BLOCK the write, never wave it through. (ResolutionCheck above stays
    # fail-open on purpose: over-cancelling a legitimate plan is its worse failure; here the
    # worse failure is persisting a possibly-fabricated value.) execute.py's call-site `default`
    # matches this.
    present: bool = False


# ── flat schemas + shape hints ────────────────────────────────────────────────────────────────


def plan_format(tool_names: list[str]) -> dict:
    """The planner's flat JSON schema, with the tool enum built from the LIVE registry (plus
    "none" for pure reasoning steps) so an /mcp reload reaches the constrained decoder too."""
    return {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "tool": {"type": "string", "enum": sorted(set(tool_names)) + ["none"]},
                        "needs_resolution": {"type": "boolean"},
                    },
                    "required": ["description", "tool", "needs_resolution"],
                },
            }
        },
        "required": ["plan"],
    }


PLAN_SHAPE = (
    "Respond with ONLY this JSON and nothing else: "
    '{"plan":[{"description":"<what this step does>",'
    '"tool":"<one exact tool name from the list, or none for a pure reasoning step>",'
    '"needs_resolution":<true if this step\'s exact file/value/items are not yet '
    "known and depend on an earlier result, else false>}]}"
)

RECTIFY_FORMAT = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "rectify": {"type": "boolean"}},
    "required": ["reasoning", "rectify"],
}
RECTIFY_SHAPE = (
    'Respond with ONLY this JSON: {"reasoning":"<why>","rectify":<true|false>}'
)

RESOLUTION_FORMAT = {
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "found": {"type": "boolean"}},
    "required": ["evidence", "found"],
}
RESOLUTION_SHAPE = (
    'Respond with ONLY this JSON: {"evidence":"<quote the exact result text that '
    "contains the referenced item and name its source — or say nothing "
    'matches>","found":<true|false>}'
)

WRITE_GATE_FORMAT = {
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "present": {"type": "boolean"}},
    "required": ["evidence", "present"],
}
WRITE_GATE_SHAPE = (
    'Respond with ONLY this JSON: {"evidence":"<quote the exact result text that '
    "contains the requested value and name its source file — or say nothing "
    'matches>","present":<true|false>}'
)


# ── tool-name normalization (planner output → live registry) ─────────────────────────────────

# Names small models emit from their priors, mapped onto the real registry tool. Includes the
# benchmark harness's tool vocabulary so prompts/exemplars written against it keep resolving.
TOOL_SYNONYMS = {
    "calc": "calculate",
    "calculator": "calculate",
    "list_dir": "list_directory",
    "ls": "list_directory",
    "search_text": "search_files",
    "grep": "search_files",
    "rag_search": "search_knowledge_base",
    "knowledge_base": "search_knowledge_base",
    "search": "web_search",
    "web": "web_search",
    "shell": "run_shell",
    "bash": "run_shell",
    "fetch": "web_extract",
    "web_fetch": "web_extract",
    "ask": "ask_user",
    "ask_human": "ask_user",
    "ask_the_user": "ask_user",
    "user_input": "ask_user",
    "question": "ask_user",
}


def norm_tool(raw, valid: "set[str] | None" = None) -> Optional[str]:
    """Normalize a planner-emitted tool name onto the live registry ("none"/junk → None).
    `valid` overrides the registry lookup for tests."""
    if not raw:
        return None
    tokens = str(raw).replace("=", " ").split("|")[0].split()
    if not tokens:  # degenerate emissions like "|read_file", "=", or whitespace
        return None
    token = tokens[0].strip().lower()
    token = TOOL_SYNONYMS.get(token, token)
    if valid is None:
        valid = registered_tools()
    return token if token in valid else None


def registered_tools() -> set[str]:
    """Live registry tool names; lazy so importing this module never loads tool deps."""
    try:
        from tools.registry import tools_by_name

        return set(tools_by_name)
    except Exception:
        return set()


def to_steps(draft: _PlanOut) -> list[dict]:
    """Planner structured output → the plain step dicts stored in state (the checkpointer
    serializer never sees a custom type). `result=None` marks a step not yet executed — the
    engine's execution pointer. Blank descriptions are dropped; step_ids renumber 1..N."""
    steps: list[dict] = []
    for s in draft.plan:
        desc = (s.description or "").strip()
        if not desc:
            continue
        steps.append(
            {
                "step_id": len(steps) + 1,
                "label": desc,
                "status": "pending",
                "intended_tool": norm_tool(s.tool),
                "result": None,
                "needs_resolution": bool(s.needs_resolution),
            }
        )
    return steps


# ── the hardened calls ────────────────────────────────────────────────────────────────────────

_ATTEMPT_TEMPS = (0.0, 0.3, 0.3)  # deterministic first; a resample often parses when 0.0 didn't


def _extract_json(text: str) -> str:
    """Salvage the outermost {...} from prose-wrapped model output."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _role_is_ollama(role: str) -> bool:
    try:
        return get_config().model_for_role(role).provider == "ollama"
    except Exception:
        return False


def _invoke_kwargs(role: str, fmt: "dict | None", temp: float) -> dict:
    """Constrained decoding + per-attempt temperature ride the invoke kwargs for Ollama roles
    (ChatOllama forwards `format`/`options` to the daemon); other providers take neither — they
    get the shape hint + salvage parsing alone.

    The options dict must carry `num_ctx` too: langchain_ollama treats an invoke-time `options`
    as a FULL REPLACEMENT for the constructor-built options (which is the only place the
    configured context window lives), so temperature alone would silently revert the daemon to
    its ~2048 default and front-truncate long prompts."""
    if not _role_is_ollama(role):
        return {}
    options: dict = {"temperature": temp}
    try:
        cfg = get_config()
        options["num_ctx"] = cfg.num_ctx_for(cfg.model_for_role(role).model)
    except Exception:  # a broken binding must not fail the call that would surface it
        pass
    kwargs: dict = {"options": options}
    if fmt is not None:
        kwargs["format"] = fmt
    return kwargs


def structured(role, messages, schema, fmt, shape, default=None, attempts=3):
    """One structured judgment call through the role's trust-wrapped model: shape hint appended,
    constrained decoding where supported, JSON salvage + lenient validation, temp-escalating
    retries, and a safe `default` when nothing parses (None default → raise)."""
    from core.llms import get_model

    payload = list(messages) + [SystemMessage(content=shape)]
    for i in range(attempts):
        temp = _ATTEMPT_TEMPS[min(i, len(_ATTEMPT_TEMPS) - 1)]
        try:
            resp = get_model(role).invoke(payload, **_invoke_kwargs(role, fmt, temp))
        except Exception as exc:
            diag.log(f"structured[{role}/{schema.__name__}] attempt {i + 1} call failed: {exc}")
            continue
        content = str(getattr(resp, "content", "") or "").strip()
        if not content:
            continue
        try:
            return schema.model_validate_json(_extract_json(content))
        except (ValidationError, ValueError):
            diag.log(
                f"structured[{role}/{schema.__name__}] attempt {i + 1} did not parse: "
                f"{content[:160]!r}"
            )
    if default is not None:
        return default
    raise ValueError(f"{schema.__name__}: no valid JSON after {attempts} attempts")


# (A `text(role, messages)` plain-text twin shipped with the transplant but never gained a
# caller — the engine's one plain-text path is nodes/execute._reasoning_call, which needs the
# raw response object for its metrics. Deleted 2026-07-04 rather than left as a second,
# unexercised text-call path someone "fixes" believing it drives the engine.)

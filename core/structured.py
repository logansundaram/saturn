"""
Hardened structured-output layer (transplanted from the agentic_benchmark harness, 2026-07-03).
Its engine callers left with the plan engine (2026-09-27); it still serves the out-of-loop
judgment calls (the memory review's proposals) and owns `_invoke_kwargs`, the ONE builder of
the per-task decoding options every model call sends.

Small local models mis-handle the full Pydantic JSON schema (`$ref`/`$defs`) that
`.with_structured_output` sends, and intermittently wrap their JSON in prose. This layer is the
defensive plumbing around a judgment call:

  - FLAT, hand-written JSON schemas constrain Ollama's decoder without `$ref` indirection, plus
    a one-line JSON "shape" hint appended as a trailing HumanMessage (see `structured`'s
    docstring — NEVER a SystemMessage, which Ollama 0.32.13 rejects mid-conversation for qwen3.8
    models) so a model that ignores the grammar still sees the exact expected spelling.
  - `_extract_json` salvages the outermost `{...}` from prose-wrapped output.
  - lenient parse models with defaults, so a missing field degrades instead of raising.
  - temperature-escalating retries (0.0 first for determinism, then sampled variety), and a
    caller-supplied `default` so a total parse failure degrades to a safe verdict instead of
    aborting the turn.

Every call goes through `core.llms.get_model(role)`, so the trust boundary is preserved: a
cloud-bound (or remote-Ollama) role is redacted + recorded to the egress ledger, and the air-gap
guard applies. Constrained decoding (`format=`) and per-attempt temperature are passed only to
Ollama-served roles — other providers get the shape hint + salvage path alone.
"""

from __future__ import annotations

from langchain.messages import HumanMessage
from pydantic import ValidationError

import diag
from config import get_config


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


def _model_tag(role: str) -> str:
    """The concrete model id serving `role`, '' when the binding can't be read."""
    try:
        return str(get_config().model_for_role(role).model)
    except Exception:
        return ""


def _invoke_kwargs(role: str, fmt: "dict | None", temp: float, task: "str | None" = None, *,
                   repetition: bool = False) -> dict:
    """Constrained decoding + per-attempt temperature + the serving layer's per-TASK decisions
    ride the invoke kwargs for Ollama roles (ChatOllama forwards `format`/`options`/`reasoning`
    to the daemon); other providers take none — they get the shape hint + salvage parsing alone.

    The options dict must carry `num_ctx` too: langchain_ollama treats an invoke-time `options`
    as a FULL REPLACEMENT for the constructor-built options (which is the only place the
    configured context window lives), so temperature alone would silently revert the daemon to
    its ~2048 default and front-truncate long prompts. Since 2026-08-15 (from the engine
    isolate) it also carries the task's `num_predict` bound, and `reasoning` (think) is set
    EXPLICITLY per task (`core/serving.thinks`) — never the model's default — unless the daemon
    already rejected the flag for this tag (`llms._NO_THINK_SUPPORT`). `repetition=True` adds
    the retry-only repeat penalty after a degenerate draw."""
    if not _role_is_ollama(role):
        return {}
    from core import llms, serving  # lazy: structured is imported by the registry's users

    task = task or serving.task_for_role(role)
    options: dict = {"temperature": temp}
    tag = _model_tag(role)
    try:
        cfg = get_config()
        options["num_ctx"] = cfg.num_ctx_for(cfg.model_for_role(role).model)
    except Exception:  # a broken binding must not fail the call that would surface it
        pass
    if task is not None:
        options["num_predict"] = serving.num_predict(task)
    if repetition:
        options.update(serving.repetition_options())
    kwargs: dict = {"options": options}
    if task is not None and tag not in llms._NO_THINK_SUPPORT:
        kwargs["reasoning"] = serving.thinks(task)
    if fmt is not None:
        kwargs["format"] = fmt
    return kwargs


def structured(role, messages, schema, fmt, shape, default=None, attempts=3):
    """One structured judgment call through the role's trust-wrapped model: shape hint appended,
    constrained decoding where supported, JSON salvage + lenient validation, temp-escalating
    retries, and a safe `default` when nothing parses (None default → raise).

    The shape hint rides as a trailing HumanMessage, never a SystemMessage (2026-08-16): Ollama
    0.32.13 raises `system message must be at the beginning (status code: 500)` for qwen3.8
    models — the shipped `27b` tier's default binding — when a SystemMessage follows any other
    message, which made EVERY structured call on that tier fail all `attempts` and silently fall
    back to `default` (an empty plan every turn, the rectify verdict, the resolution check, and
    the write gate all degraded). A trailing HumanMessage validates universally (measured against
    the live daemon on both qwen3.8:27b and qwen3.5:9b) and reads naturally as the final user
    turn — it IS a formatting instruction. Do not move this back to SystemMessage."""
    from core.llms import get_model

    from textutil import looks_repetitive

    payload = list(messages) + [HumanMessage(content=shape)]
    repetition = False  # a degenerate draw arms the retry-only repeat penalty for the next rung
    for i in range(attempts):
        temp = _ATTEMPT_TEMPS[min(i, len(_ATTEMPT_TEMPS) - 1)]
        try:
            from core.llms import generate  # the think-rejection fallback rides every call

            resp = generate(get_model(role), payload, tag=_model_tag(role),
                            **_invoke_kwargs(role, fmt, temp, repetition=repetition))
        except Exception as exc:
            diag.log(f"structured[{role}/{schema.__name__}] attempt {i + 1} call failed: {exc}")
            continue
        content = str(getattr(resp, "content", "") or "").strip()
        if not content:
            continue
        try:
            return schema.model_validate_json(_extract_json(content))
        except (ValidationError, ValueError):
            from core.llms import was_truncated

            if was_truncated(resp):
                diag.log(
                    f"structured[{role}/{schema.__name__}] attempt {i + 1} was cut off at "
                    f"num_predict — the {role} task's bound is below this verdict's length"
                )
            diag.log(
                f"structured[{role}/{schema.__name__}] attempt {i + 1} did not parse: "
                f"{content[:160]!r}"
            )
            repetition = repetition or looks_repetitive(content)
    if default is not None:
        return default
    raise ValueError(f"{schema.__name__}: no valid JSON after {attempts} attempts")


# (A `text(role, messages)` plain-text twin shipped with the transplant but never gained a
# caller — the engine's one plain-text path is nodes/execute._reasoning_call, which needs the
# raw response object for its metrics. Deleted 2026-07-04 rather than left as a second,
# unexercised text-call path someone "fixes" believing it drives the engine.)

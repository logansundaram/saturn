"""
/confidence — the front door for confidence coloring.

Coloring is ON by default and part of the ordinary experience: while the answer streams, runs of
consecutive low-probability tokens render red. What "low" means is CALIBRATED PER MODEL
(core/confidence_calibration.py, the shipped baseline) — a red run says "worse than 95 % of this
model's clean output", which is why Saturday.ai binds one model family (qwen3.5/qwen3.6/qwen3.8).

This command owns every user-facing lever over that: the on/off switch, re-measuring the active
synthesizer against the live daemon, and typing your own numbers. Measured and typed values land
in the user overlay (core/confidence_store), which wins over the shipped table.
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import parse_toggle_status, split_persist_flags

_USAGE = (
    "  usage: /confidence [on|off] | set <enter> [exit] | tune [--prompts N] | reset"
)


def _persist(cfg, key: str) -> None:
    """Persist one scalar through the ONE persist seam (same machinery as /config --save)."""
    from commands.config import _persist_key

    _persist_key(cfg, key)


def _active_model() -> str:
    from core import confidence

    return confidence._synthesizer_model()


def _source_of(model: str) -> str:
    """Where the live thresholds come from, for the status line."""
    from core import confidence, confidence_store

    if confidence._configured_threshold() is not None:
        return "pinned (runtime.confidence_threshold)"
    rec = confidence_store.entry_for(model)
    if rec:
        return f"{rec.get('source', 'overlay')} · {rec.get('at', '')}".strip(" ·")
    from core import confidence_calibration as table

    if table.CALIBRATION.get(str(model or "").lower()):
        return "shipped calibration"
    return "built-in default (uncalibrated model)"


def _status() -> None:
    from config import get_config
    from core import confidence, confidence_store

    cfg = get_config()
    on = bool(cfg.get("runtime.confidence", True))
    model = _active_model()

    _print(f"  coloring   {'on' if on else 'off'}")
    _print(f"  model      {model or '—'}")
    _print(f"  enter      {confidence.threshold():.4f}   (a run opens below this probability)")
    _print(f"  exit       {confidence.exit_threshold():.4f}   (an open run extends below this)")
    _print(f"  source     {_source_of(model)}")

    rec = confidence_store.entry_for(model)
    if rec and rec.get("tokens"):
        _print(f"  measured   {rec['tokens']} tokens over {rec.get('prompts', 0)} prompts")
    problem = confidence_store.load_problem()
    if problem:
        _print(f"  ! {problem}")
    if not on:
        _print("  turn it back on with `/confidence on`.")


def _set(args: list) -> None:
    from core import confidence, confidence_store

    if not args:
        _print("  usage: /confidence set <enter> [exit]   (probabilities, 0 < p < 1)")
        return
    try:
        enter = float(args[0])
    except (TypeError, ValueError):
        _print(f"  not a probability: {args[0]!r} (expected a number between 0 and 1)")
        return
    if not 0.0 < enter < 1.0:
        _print(f"  enter must be between 0 and 1 (got {enter})")
        return

    if len(args) > 1:
        try:
            exit_p = float(args[1])
        except (TypeError, ValueError):
            _print(f"  not a probability: {args[1]!r}")
            return
        if not 0.0 < exit_p < 1.0:
            _print(f"  exit must be between 0 and 1 (got {exit_p})")
            return
        if exit_p <= enter:
            _print(f"  exit ({exit_p}) must be ABOVE enter ({enter}) — the exit threshold is the "
                   "looser one an open run extends through.")
            return
    else:
        # The same derivation an uncalibrated model gets: looser by 1.5x, capped.
        exit_p = min(confidence._EXIT_CAP, enter * confidence._EXIT_FACTOR)

    model = _active_model()
    confidence_store.write_entry(model, enter, exit_p, source="manual")
    _print(f"  {model}: enter {enter:.4f} · exit {exit_p:.4f} (yours).")
    _print("  `/confidence reset` restores the shipped calibration.")


def _reset() -> None:
    from core import confidence_store

    model = _active_model()
    if confidence_store.clear_entry(model):
        _print(f"  {model}: your values dropped — back to the shipped calibration.")
    else:
        _print(f"  {model}: nothing of yours stored; already on the shipped calibration.")


def _tune(args: list) -> None:
    from core import calibration, confidence_store

    prompts = None
    if "--prompts" in args:
        i = args.index("--prompts")
        try:
            n = int(args[i + 1])
        except (IndexError, TypeError, ValueError):
            _print("  usage: /confidence tune [--prompts N]")
            return
        if n < 1:
            _print("  --prompts must be at least 1")
            return
        prompts = calibration.PROMPTS[:n]

    model = _active_model()
    if not model:
        _print("  no synthesizer model bound — nothing to calibrate.")
        return

    total = len(prompts) if prompts is not None else len(calibration.PROMPTS)
    _print(f"  calibrating {model} over {total} prompts — this runs the live daemon.")

    def _progress(i, n, prompt, added):
        _print(f"    [{i:>2}/{n}] {prompt[:44]:<44} +{added:>4} tokens")

    try:
        result = calibration.measure(model, prompts, on_progress=_progress)
    except Exception as exc:
        _print(f"  calibration failed: {exc}")
        _print("  nothing was written — the previous thresholds still apply.")
        return

    # Below MIN_TOKENS scored content tokens a quantile is noise, not a threshold — the usual
    # cause is a daemon that returned no per-token logprobs for this model at all.
    if result.get("tokens", 0) < calibration.MIN_TOKENS:
        _print(f"  only {result.get('tokens', 0)} scored token(s) came back (need at least "
               f"{calibration.MIN_TOKENS}) — the daemon likely returned no per-token logprobs "
               "for this model.")
        _print("  nothing was written. Confidence marks stay on the previous thresholds.")
        return

    confidence_store.write_entry(
        model, result["enter"], result["exit"], source="tuned",
        tokens=result["tokens"], prompts=result.get("prompts", total),
    )
    _print(f"  {model}: enter {result['enter']:.4f} · exit {result['exit']:.4f} "
           f"({result['tokens']} tokens).")


@command(
    "confidence",
    "Confidence coloring: on/off, re-calibrate, or set your own thresholds.",
    usage="/confidence [on|off] | set <enter> [exit] | tune [--prompts N] | reset",
    details="""
Confidence coloring marks the parts of an answer the model itself was least sure of: while the
answer streams, runs of consecutive low-probability tokens render RED — live, in the freeze editor
(Esc), and on the final render. It is ON by default.

What counts as "low" is calibrated PER MODEL, because the same probability means different things
at different model sizes. A red run means "worse than 95 % of this model's clean output". That is
why Saturday.ai binds one model family (qwen3.5 / qwen3.6 / qwen3.8) — see /models tier.

  /confidence                    status: on/off, the active model, its thresholds and where
                                 they came from
  /confidence off                stop capturing logprobs entirely (nothing is marked)
  /confidence on                 back on
  /confidence tune               re-measure the ACTIVE synthesizer against the live daemon and
                                 store the result as yours. Takes a few minutes.
  /confidence tune --prompts 20  a quicker, coarser pass
  /confidence set 0.31 0.52      type your own enter/exit probabilities for the active model.
                                 LOWER enter = stricter = fewer marks. Omit the exit and it is
                                 derived (1.5x, capped at 0.95).
  /confidence reset              drop your values; back to the shipped calibration

on/off persists to config.yaml by default; add --session to apply it for this session only.
Your tuned and typed values live in database/confidence_calibration.json and survive /update.
""".strip(),
)
def _confidence(ctx, args):
    from config import get_config

    args = list(args or [])
    args, session, _save = split_persist_flags(args)
    sub = args[0].lower() if args else ""

    if sub == "set":
        _set(args[1:])
        return
    if sub == "tune":
        _tune(args[1:])
        return
    if sub == "reset":
        _reset()
        return

    # Bare = STATUS (never a flip); on/off mutates. The one toggle grammar.
    verdict = parse_toggle_status(args)
    if verdict is None:
        _status()
        return
    if verdict == "invalid":
        _print(_USAGE)
        return

    cfg = get_config()
    cfg.set("runtime.confidence", bool(verdict))
    _print(f"  confidence coloring {'on' if verdict else 'off'}"
           f"{' (session only)' if session else ''}.")
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        _persist(cfg, "runtime.confidence")

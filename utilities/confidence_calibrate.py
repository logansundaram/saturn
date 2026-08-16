"""
confidence_calibrate.py — derive per-model confidence thresholds against the LIVE daemon (C3,
from the confidence_coloring isolate, 2026-08-16). Sibling of continuation_contract.py: a dev
utility, not shipped, run by hand.

    python utilities/confidence_calibrate.py                    # every tier's synthesizer that is installed
    python utilities/confidence_calibrate.py --models qwen3.8:27b gemma4:e4b
    python utilities/confidence_calibrate.py --prompts 20       # a quick, coarser pass

The measurement itself — streaming known-good prompts through the synthesizer's real prompt
shape, scoring every content token by its sampled probability exactly as core/confidence.low_runs
does, and reporting the 5 % point as `enter` and the 10 % point as `exit` — lives in
core/calibration.py (moved there 2026-08-16 because it SHIPS: /confidence tune re-measures the
active synthesizer at runtime, and utilities/ is excluded from the wheel). This CLI is a thin
wrapper: it picks which models to measure, prints progress, and writes the result.

Writes core/confidence_calibration.py (a generated data module the wheel carries) — merged, so
calibrating one model never drops another. Goes through core.llms' Ollama boundary (air-gap /
locality honored); nothing here is a new egress path.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_config  # noqa: E402
from core import calibration, llms  # noqa: E402

OUT = ROOT / "core" / "confidence_calibration.py"


def _installed() -> set:
    """Model tags pulled into the daemon — through core.llms (no client import of our own; the
    no-new-egress guard holds for utilities too)."""
    return {m.name.lower() for m in llms.list_local_models()}


def _tier_synthesizers() -> list:
    cfg = get_config()
    out = []
    for tier, spec in (cfg.get("tiers") or {}).items():
        m = ((spec or {}).get("roles") or {}).get("synthesizer")
        if isinstance(m, dict):
            m = m.get("model")
        if m and str(m) not in out:
            out.append(str(m))
    return out


def _write_table(table: dict) -> None:
    src = OUT.read_text(encoding="utf-8")
    head, sep, _tail = src.partition("CALIBRATION: dict = ")
    body = pprint.pformat(table, width=96, sort_dicts=True)
    OUT.write_text(head + sep + body + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model tags to calibrate (default: every tier's synthesizer)")
    ap.add_argument("--prompts", type=int, default=len(calibration.PROMPTS), help="how many prompts to stream")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--inherit", nargs="*", default=[], metavar="TAG=FROM",
                    help="record TAG with FROM's thresholds, marked inherited (for a model the "
                         "daemon returns no per-token logprobs for yet — e.g. qwen3.8 on Ollama "
                         "0.32, which emits logprobs on the first chunk only); a later measured "
                         "run overwrites it")
    args = ap.parse_args(argv)

    from core import confidence_calibration as current

    installed = _installed()
    wanted = args.models if args.models is not None else _tier_synthesizers()
    models = [m for m in wanted if not installed or m.lower() in installed]
    skipped = [m for m in wanted if m not in models]
    if skipped:
        print(f"not installed, skipped: {', '.join(skipped)}", file=sys.stderr)
    if not models and not args.inherit:
        print("nothing to calibrate", file=sys.stderr)
        return 1

    table = dict(current.CALIBRATION)
    prompts = calibration.PROMPTS[: max(1, args.prompts)]
    for tag in models:
        try:
            result = calibration.measure(
                tag, prompts,
                on_progress=None if args.quiet else (
                    lambda i, n, q, added: print(
                        f"  [{i:>2}/{n}] {q[:48]:<48} +{added:>4} tokens", file=sys.stderr)
                ),
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        table[tag.lower()] = {
            "enter": result["enter"], "exit": result["exit"],
            "tokens": result["tokens"], "prompts": result["prompts"],
            "at": _dt.date.today().isoformat(),
        }
    for spec in args.inherit:
        tag, _, src = spec.partition("=")
        rec = table.get(src.lower())
        if not (tag and rec):
            print(f"--inherit {spec}: {src!r} is not calibrated — nothing recorded", file=sys.stderr)
            continue
        table[tag.lower()] = {"enter": rec["enter"], "exit": rec["exit"], "tokens": 0,
                              "prompts": 0, "at": _dt.date.today().isoformat(),
                              "inherited_from": src.lower()}
        print(f"  {tag}: inherited enter={rec['enter']} exit={rec['exit']} from {src} "
              "(NOT measured — re-run without --inherit once the daemon returns per-token "
              "logprobs for it)", file=sys.stderr)
    _write_table(table)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(table)} model(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

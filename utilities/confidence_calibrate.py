"""
confidence_calibrate.py — derive per-model confidence thresholds against the LIVE daemon (C3,
from the confidence_coloring isolate, 2026-08-16). Sibling of continuation_contract.py: a dev
utility, not shipped, run by hand.

    python utilities/confidence_calibrate.py                    # every tier's synthesizer that is installed
    python utilities/confidence_calibrate.py --models qwen3.8:27b gemma4:e4b
    python utilities/confidence_calibrate.py --prompts 20       # a quick, coarser pass

For each model it streams a set of KNOWN-GOOD prompts (concise factual Q&A + enumerations of
memorized sequences — the genre synthesize produces on a clean turn) through the synthesizer's
real prompt shape with the same serving kwargs (think off, the answer task's bounds, logprobs on),
scores every content token by its sampled probability EXACTLY as core/confidence.low_runs does
(neutral punctuation and closed-class stopwords excluded), and records the 5 % point as `enter`
and the 10 % point as `exit`: "marked" then means "worse than 95 % of this model's clean output".
The isolate's data showed a 6-10x spread across model sizes, so this is per model by design.

Writes core/confidence_calibration.py (a generated data module the wheel carries) — merged, so
calibrating one model never drops another. Goes through core.llms' Ollama boundary (air-gap /
locality honored); nothing here is a new egress path.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain.messages import HumanMessage  # noqa: E402

from config import get_config  # noqa: E402
from core import confidence, llms  # noqa: E402
from core.messages import synthesize_sys_msg  # noqa: E402
from core.serving import num_predict  # noqa: E402
from trust import egress  # noqa: E402

OUT = ROOT / "core" / "confidence_calibration.py"

PROMPTS = [
    "What is the capital of France?", "What is the capital of Japan?",
    "What is the capital of Germany?", "What is the capital of Italy?",
    "What is the capital of Spain?", "Who wrote Romeo and Juliet?", "Who wrote 1984?",
    "Who painted the Mona Lisa?", "Who painted the ceiling of the Sistine Chapel?",
    "Who developed the theory of general relativity?",
    "Who was the first president of the United States?",
    "Who was the first person to walk on the Moon?", "Who discovered penicillin?",
    "Who invented the telephone?", "Who composed the Ninth Symphony with the Ode to Joy?",
    "What is the largest planet in the solar system?", "Which planet is closest to the Sun?",
    "Which planet is known as the Red Planet?", "What is the chemical symbol for gold?",
    "What is the chemical symbol for oxygen?", "What is the chemical formula for water?",
    "Which element has atomic number 1?", "What gas do plants absorb during photosynthesis?",
    "What is the hardest natural substance on Earth?",
    "What is the boiling point of water in Celsius at sea level?",
    "What is the freezing point of water in Fahrenheit?", "How many continents are there?",
    "How many days are in a leap year?", "How many sides does a hexagon have?",
    "How many legs does a spider have?", "How many strings does a standard guitar have?",
    "How many minutes are in an hour?", "What is the smallest prime number?",
    "What is the square root of 64?", "What is the tallest mountain on Earth?",
    "What is the longest river in South America?", "What is the largest ocean?",
    "Which continent is the Sahara Desert in?", "Which country is home to the kangaroo?",
    "What is the currency of Japan?", "What is the currency of the United Kingdom?",
    "What language is primarily spoken in Brazil?", "What is the largest mammal?",
    "What do bees produce?", "What is the largest organ of the human body?",
    "What is the main language spoken in Mexico?", "In which city is the Eiffel Tower located?",
    "In which city is the Colosseum located?",
    # Enumerations of memorized sequences — each contributes many known-good tokens.
    "List the planets of the solar system in order from the Sun.",
    "List the seven continents.", "List the days of the week.",
    "List the twelve months of the year.", "Write out the numbers from one to twenty as words.",
    "List the first ten elements of the periodic table.",
    "Name the primary colors and the secondary colors.",
]


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


def _score_model(tag: str, prompts: list, quiet: bool) -> list:
    """Every content-token probability from clean answers, scored like low_runs."""
    if not egress.ollama_is_local() and egress.airgap_on():
        raise SystemExit("air-gap is ON and OLLAMA_HOST is off-machine — refusing to calibrate")
    model = llms._build("ollama", tag)
    cfg = get_config()
    kwargs = {
        "options": {"temperature": 0.7, "num_ctx": cfg.num_ctx_for(tag),
                    "num_predict": num_predict("answer")},
        "reasoning": False,
        "logprobs": True,
    }
    ps: list = []
    chunks = lp_chunks = 0
    for i, q in enumerate(prompts, 1):
        msgs = [synthesize_sys_msg, HumanMessage(content=f"Current user query:\n{q}")]
        text = ""
        entries: list = []
        for chunk in llms.stream(model, msgs, tag=tag, **kwargs):
            piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            lp = (getattr(chunk, "response_metadata", None) or {}).get("logprobs")
            if piece:
                chunks += 1
                if lp:
                    lp_chunks += 1
                    entries += confidence.align_chunk(piece, lp, offset=len(text))
                text += piece
        n_before = len(ps)
        for e in entries:
            tok = text[e["start"]:e["end"]]
            if not any(ch.isalnum() for ch in tok) or confidence.is_stopword(tok):
                continue  # neutral for low_runs — neutral for the calibration too
            ps.append(math.exp(min(float(e["logprob"]), 0.0)))
        if not quiet:
            print(f"  [{i:>2}/{len(prompts)}] {q[:48]:<48} +{len(ps) - n_before:>4} tokens",
                  file=sys.stderr)
    if chunks and lp_chunks < chunks // 2:
        print(f"  NOTE: the daemon returned logprobs on only {lp_chunks}/{chunks} content chunks "
              f"for {tag} — its runner does not report per-token logprobs yet; confidence marks "
              "are unmeasured for this model (use --inherit TAG=FROM to record a provisional "
              "threshold from a sibling)", file=sys.stderr)
    return ps


def _quantile(sorted_ps: list, q: float) -> float:
    if not sorted_ps:
        return 0.0
    i = min(len(sorted_ps) - 1, max(0, int(round(q * (len(sorted_ps) - 1)))))
    return sorted_ps[i]


def _write_table(table: dict) -> None:
    src = OUT.read_text(encoding="utf-8")
    head, sep, _tail = src.partition("CALIBRATION: dict = ")
    body = pprint.pformat(table, width=96, sort_dicts=True)
    OUT.write_text(head + sep + body + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model tags to calibrate (default: every tier's synthesizer)")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS), help="how many prompts to stream")
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
    prompts = PROMPTS[: max(1, args.prompts)]
    for tag in models:
        print(f"calibrating {tag} over {len(prompts)} prompts…", file=sys.stderr)
        ps = sorted(_score_model(tag, prompts, args.quiet))
        if len(ps) < 50:
            print(f"  too few scored tokens ({len(ps)}) — no logprobs from the daemon? skipping", file=sys.stderr)
            continue
        enter = round(_quantile(ps, 0.05), 4)
        exit_ = round(max(enter, _quantile(ps, 0.10)), 4)
        table[tag.lower()] = {
            "enter": enter, "exit": exit_, "tokens": len(ps), "prompts": len(prompts),
            "at": _dt.date.today().isoformat(),
        }
        print(f"  {tag}: enter={enter} (p05)  exit={exit_} (p10)  over {len(ps)} content tokens  "
              f"(p50={_quantile(ps, 0.5):.3f})", file=sys.stderr)
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

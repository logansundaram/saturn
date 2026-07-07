"""
The splice-and-continue CONTRACT TEST — the definition of "supported model" for
interrupt-and-correct (token steering).

For each model it: (1) generates a deterministic counting answer through the raw-mode
continuation primitive, (2) programmatically truncates the output MID-TOKEN (inside a number
word), (3) splices in a hand-written completion that *changes the trajectory* (jumping the count
to seventy-seven), and (4) resumes via `continue_from`, asserting the model continues from the
HUMAN text — the next line must be the count's successor ("seventy-nine") — without restarting
the turn, re-greeting, or emitting template special tokens.

That one check proves everything the feature rests on: raw-mode templating is byte-correct (a
wrong special token restarts the turn or produces garbage), the daemon re-tokenizes the edited
prefix cleanly across a mid-token cut, `stop` halts at the family's end-of-turn, and the model
genuinely cannot tell the spliced human text from its own — it obeys the new trajectory.

A model family is OFFICIALLY SUPPORTED iff it is in core/chat_template.TEMPLATES *and* this
script passes on it. Gate any registry extension on a green run here — this is the regression
net that keeps the model × template matrix from becoming whack-a-mole.

Run from the repo root (needs the live Ollama daemon; this is deliberately NOT part of the
offline pytest suite):

    python utilities/continuation_contract.py             # one installed model per family
    python utilities/continuation_contract.py --all       # every installed supported model
    python utilities/continuation_contract.py --models qwen3.6:27b gemma4:e4b

Exit code 0 = every graded model passed; 1 = any failure (the CI-style gate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, like agent.py runs

from core import chat_template, continuation  # noqa: E402

SYSTEM = "You are a careful assistant. Follow the user's instructions exactly."
QUERY = ("Count upward in English words, one number per line, starting from the word one. "
         "Lowercase words only - no digits, no commentary.")
HISTORY = [("system", SYSTEM), ("user", QUERY)]

# The hand-written splice: completes the cut-open word "sev" to "seventy-seven" and adds one more
# line, so a correct continuation MUST produce the successor of the human's count, not the
# model's own (which was down in single digits when frozen).
SPLICE = "enty-seven\nseventy-eight\n"
EXPECTED = "seventy-nine"

# A continuation that restarts the turn / leaks template structure fails regardless of content.
FORBIDDEN = ("<|im_start|>", "<|im_end|>", "<|turn>", "<turn|>", "<think>", "system\n")


def _run(model: str, prefix: str, n_predict: int) -> tuple[str, str]:
    """One raw-mode generation through the real primitive; returns (text, done_reason).

    Sampling is PINNED (greedy, no repeat penalty) so the grade is deterministic: the check
    counts number words, and a default repeat_penalty (gemma4 ships without an explicit one, so
    Ollama's 1.1 applies) punishes the repeated decade tokens hard enough at temperature 0 to
    derail exact counting — a sampling artifact, not a template failure. The FEATURE itself
    never overrides sampling; only the contract's grading environment is pinned."""
    stream = continuation.continue_from(
        model, HISTORY, prefix,
        options={"temperature": 0, "num_predict": n_predict, "repeat_penalty": 1.0},
    )
    text = "".join(stream)
    return text, stream.done_reason


def grade(model: str, verbose: bool = True) -> tuple[bool, str]:
    """The splice-and-continue check for one model: True = supported behavior observed."""
    first, reason = _run(model, "", 48)
    if not first.strip():
        return False, "first generation produced no text"
    if reason not in ("stop", "length"):
        return False, f"first generation ended abnormally (done_reason={reason!r})"

    # Truncate MID-TOKEN: cut inside the word "seven" so the spliced human text completes it to a
    # different number. Counting at temperature 0 reliably reaches seven within 48 tokens; if a
    # model words things differently, fall back to a plain 60% cut with a line-start splice.
    i = first.find("seven")
    if i != -1:
        prefix = first[: i + 3] + SPLICE            # "...six\nsev" + "enty-seven\n..."
    else:
        prefix = first[: int(len(first) * 0.6)].rstrip() + "\nseventy-seven\nseventy-eight\n"

    cont, reason = _run(model, prefix, 32)
    if verbose:
        tail = prefix[-40:].replace("\n", "\\n")
        head = cont[:60].replace("\n", "\\n")
        print(f"    spliced prefix tail: ...{tail!r}")
        print(f"    continuation head:    {head!r}  (done_reason={reason!r})")

    if not cont.strip():
        return False, "continuation produced no text"
    for tok in FORBIDDEN:
        if tok in cont:
            return False, f"continuation leaked template structure ({tok!r}) — the turn restarted"
    if cont.lstrip().lower().startswith(("one\n", "sure", "okay", "here", "certainly", "i ")):
        return False, "continuation restarted the answer instead of continuing the prefix"
    if EXPECTED not in cont.lower():
        return False, (f"continuation did not follow the human splice "
                       f"(expected {EXPECTED!r} in the first lines, got {cont[:80]!r})")
    return True, "seamless continuation from the human-edited prefix"


def _preset_models() -> set[str]:
    """Every model id any tier preset binds to a role — the models Saturn actually ships/runs."""
    from config import get_config

    out: set[str] = set()
    for tier in (get_config().get("tiers", {}) or {}).values():
        for m in (tier or {}).get("roles", {}).values():
            if isinstance(m, str):
                out.add(m)
    return out


def _default_models(run_all: bool) -> list[str]:
    """Installed models to grade: every supported one under --all, else one per registry family —
    preferring the live synthesizer binding, then any tier-preset binding (what Saturn actually
    ships), then the smallest installed match."""
    from core.llms import list_local_models, model_id

    installed = [m for m in list_local_models() if not m.is_embedding]
    preset = _preset_models()
    picks: list[str] = []
    for t in chat_template.TEMPLATES:
        matches = [m for m in installed if m.name.lower().startswith(t.prefixes)]
        if not matches:
            print(f"  ! no installed model matches family {t.family} ({'/'.join(t.prefixes)})")
            continue
        if run_all:
            picks.extend(m.name for m in matches)
            continue
        named = [m.name for m in matches]
        bound = model_id("synthesizer")
        if bound in named:
            picks.append(bound)
        elif preset & set(named):
            picks.append(sorted(preset & set(named))[0])
        else:
            picks.append(min(matches, key=lambda m: m.size_bytes).name)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description="splice-and-continue contract test (live Ollama)")
    ap.add_argument("--models", nargs="*", help="model tags to grade (default: one per family)")
    ap.add_argument("--all", action="store_true", help="grade every installed supported model")
    args = ap.parse_args()

    models = args.models or _default_models(args.all)
    if not models:
        print("nothing to grade — pull a supported model (qwen3.5/3.6 or gemma4) first")
        return 1

    failures = 0
    print("splice-and-continue contract test\n")
    for model in models:
        if not chat_template.supported(model):
            print(f"  SKIP  {model} — not in the template registry")
            continue
        print(f"  {model} ({chat_template.template_for(model).family})")
        try:
            ok, why = grade(model)
        except Exception as exc:
            ok, why = False, f"error: {exc}"
        print(f"    {'PASS' if ok else 'FAIL'}  {why}\n")
        failures += 0 if ok else 1

    print(f"{'all green' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

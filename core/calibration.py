"""
Confidence calibration — the measurement (2026-08-16, moved out of utilities/ on the family lock).

Streams KNOWN-GOOD prompts through the synthesizer's real prompt shape and serving kwargs, scores
every content token by its sampled probability EXACTLY as core/confidence.low_runs does (neutral
punctuation and closed-class stopwords excluded), and reports the 5 % point as `enter` and the
10 % point as `exit`: "marked" then means "worse than 95 % of this model's clean output".

This lives in core/ (not utilities/) because it SHIPS: `/confidence tune` re-measures the active
synthesizer at runtime, and utilities/ is excluded from the wheel. Two callers, one measurement:
utilities/confidence_calibrate.py writes the shipped baseline table, /confidence tune writes the
user overlay (core/confidence_store).

Goes through core.llms' Ollama boundary (air-gap / locality honored); not a new egress path.
"""

from __future__ import annotations

import math

PROMPTS: list[str] = [
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


def quantile(values: list, q: float) -> float:
    """The q-quantile of `values` (sorted internally). 0.0 for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    i = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[i]


def summarize(ps: list) -> dict:
    """Token probabilities -> the calibrated pair. `enter` = the 5 % point (a run opens under
    it), `exit` = the 10 % point (an open run extends under it)."""
    return {
        "enter": round(quantile(ps, 0.05), 4),
        "exit": round(quantile(ps, 0.10), 4),
        "tokens": len(ps),
    }


def measure(tag: str, prompts: "list | None" = None, on_progress=None) -> dict:
    """Stream `prompts` through `tag` and return {enter, exit, tokens, prompts}.

    `on_progress(i, total, prompt, added)` is called after each prompt so a caller can render
    a live line (the CLI prints to stderr; /confidence tune prints through the TUI).
    Raises RuntimeError when the air-gap forbids an off-machine daemon."""
    from config import get_config
    from core import confidence, llms
    from core.messages import synthesize_sys_msg
    from core.serving import num_predict
    from langchain.messages import HumanMessage
    from trust import egress

    if not egress.ollama_is_local() and egress.airgap_on():
        raise RuntimeError("air-gap is ON and OLLAMA_HOST is off-machine — refusing to calibrate")

    prompts = list(prompts if prompts is not None else PROMPTS)
    model = llms._build("ollama", tag)
    cfg = get_config()
    kwargs = {
        "options": {"temperature": 0.7, "num_ctx": cfg.num_ctx_for(tag),
                    "num_predict": num_predict("answer")},
        "reasoning": False,
        "logprobs": True,
    }

    ps: list = []
    for i, q in enumerate(prompts, 1):
        text = ""
        entries: list = []
        for chunk in llms.stream(model, [synthesize_sys_msg,
                                         HumanMessage(content=f"Current user query:\n{q}")],
                                 tag=tag, **kwargs):
            piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            lp = (getattr(chunk, "response_metadata", None) or {}).get("logprobs")
            if piece:
                if lp:
                    entries += confidence.align_chunk(piece, lp, offset=len(text))
                text += piece
        before = len(ps)
        for e in entries:
            tok = text[e["start"]:e["end"]]
            if not any(ch.isalnum() for ch in tok) or confidence.is_stopword(tok):
                continue  # neutral for low_runs — neutral for the calibration too
            ps.append(math.exp(min(float(e["logprob"]), 0.0)))
        if on_progress:
            on_progress(i, len(prompts), q, len(ps) - before)

    out = summarize(ps)
    out["prompts"] = len(prompts)
    return out

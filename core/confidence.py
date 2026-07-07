"""
Token-confidence grading (interrupt-and-correct's companion, 2026-07-06).

While the final answer streams, the daemon's per-token logprobs are aligned to character ranges
and carried on the provenance buffer as a parallel `confidence` overlay (core/provenance.py).
This module owns the two pure halves of that:

  - `align_chunk(text, logprobs)` — turn one streamed chunk's logprob entries into
    CHUNK-RELATIVE `{"start", "end", "logprob"}` dicts (plain dicts — gotcha #4: the buffer
    rides the checkpointer). When the daemon's token strings don't reassemble the chunk text
    exactly (rare), the whole chunk gets ONE mean-logprob entry — an honest coarse reading,
    never mis-attributed character offsets.
  - `low_runs(entries, text)` — the display question: which character ranges should render red?
    A token is LOW when its sampled probability sits under `runtime.confidence_threshold`;
    a run is >= `_MIN_RUN` consecutive low tokens (neutral tokens — whitespace/punctuation,
    whose probabilities say nothing about content — ride along without counting or breaking a
    run). A gap in the ledger (a chunk that carried no logprobs) always breaks a run: unmeasured
    text is never bridged. Single low tokens are noise (an open synonym choice, a sentence
    start); the STRUNG-TOGETHER run is the hallucination signature this feature marks.

Renderers treat everything here as additive: no entries -> no marks, and every consumer wraps
its call so a confidence failure can never cost the answer. `runtime.confidence: false` turns
the capture off at the source (nobody requests logprobs, the overlay stays empty).

Leaf module: imports only config + stdlib, so the tests exercise it fully offline.
"""

from __future__ import annotations

import math

from config import get_config

# A token counts LOW below this sampled probability (runtime.confidence_threshold overrides).
# LOWER = stricter = fewer, higher-confidence-of-uncertainty marks; RAISE = more aggressive.
_DEFAULT_THRESHOLD = 0.20

# Consecutive low (non-neutral) tokens before a run is worth marking — one uncertain token is
# an open word choice; three strung together is the drifting-generation signature.
_MIN_RUN = 3


def enabled() -> bool:
    """Whether confidence grading is on (`runtime.confidence`, default true). Fail-open to the
    default — an unreadable config must not silently change what the stream requests."""
    try:
        return bool(get_config().get("runtime.confidence", True))
    except Exception:
        return True


def threshold() -> float:
    """The low-token probability threshold (`runtime.confidence_threshold`)."""
    try:
        return float(get_config().get("runtime.confidence_threshold", _DEFAULT_THRESHOLD))
    except Exception:
        return _DEFAULT_THRESHOLD


def _read_entry(e) -> "tuple[str, float] | None":
    """(token, logprob) from one daemon logprob entry, tolerating both the raw-JSON dict shape
    (the /api/generate path) and the ollama client's attribute-shaped objects (the chat path,
    which langchain forwards untouched). None for anything unreadable."""
    if isinstance(e, dict):
        tok, lp = e.get("token"), e.get("logprob")
    else:
        tok, lp = getattr(e, "token", None), getattr(e, "logprob", None)
    if tok is None or lp is None:
        return None
    try:
        return str(tok), float(lp)
    except (TypeError, ValueError):
        return None


def align_chunk(text: str, logprobs, offset: int = 0) -> list[dict]:
    """One streamed chunk's logprob entries as character-ranged confidence dicts (offsets
    relative to the chunk start + `offset`). Empty when the chunk carried no readable logprobs —
    a gap in the ledger, which low_runs treats as unmeasured (never marked, never bridged)."""
    if not text or not logprobs:
        return []
    toks = [t for t in map(_read_entry, logprobs) if t is not None]
    if not toks:
        return []
    if "".join(t for t, _ in toks) == text:
        out, pos = [], offset
        for tok, lp in toks:
            if tok:
                out.append({"start": pos, "end": pos + len(tok), "logprob": lp})
                pos += len(tok)
        return out
    # Token strings don't reassemble the chunk (multi-token chunk drift, unicode split): one
    # mean-logprob entry over the whole chunk — coarse but never at wrong character offsets.
    lps = [lp for _, lp in toks]
    return [{"start": offset, "end": offset + len(text), "logprob": sum(lps) / len(lps)}]


def low_runs(entries, text: str, threshold_p: "float | None" = None,
             min_run: int = _MIN_RUN) -> list[tuple[int, int]]:
    """The character ranges to mark red: runs of >= `min_run` consecutive low-probability
    tokens over `text`, per the module docstring's rules. Entries must be in text order (every
    producer appends in stream order). Edges are trimmed to non-whitespace so a mark never
    starts on the space before a word."""
    th = threshold() if threshold_p is None else float(threshold_p)
    runs: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []  # the low tokens of the run being built

    def close() -> None:
        if len(cur) >= min_run:
            s, e = cur[0][0], cur[-1][1]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                runs.append((s, e))
        cur.clear()

    prev_end = None
    for ent in entries or []:
        try:
            s, e, lp = int(ent["start"]), int(ent["end"]), float(ent["logprob"])
        except (KeyError, TypeError, ValueError):
            close()
            prev_end = None
            continue
        e = min(e, len(text))
        if e <= s or s >= len(text):
            continue
        if prev_end is not None and s != prev_end:
            close()  # a ledger gap: never bridge a run across unmeasured text
        prev_end = e
        tok = text[s:e]
        neutral = not any(ch.isalnum() for ch in tok)
        if neutral:
            continue  # rides along: neither counts toward, nor breaks, the run
        if math.exp(min(lp, 0.0)) < th:
            cur.append((s, e))
        else:
            close()
    close()
    return runs


def buffer_runs(buf) -> list[tuple[int, int]]:
    """low_runs over a provenance buffer's overlay — THE one convenience every renderer calls
    (live tail excepted: it grades its own ledger). Tolerates None/garbage as no-marks."""
    try:
        if not isinstance(buf, dict):
            return []
        return low_runs(buf.get("confidence") or [], str(buf.get("text") or ""))
    except Exception:
        return []

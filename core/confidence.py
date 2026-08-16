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
    whose probabilities say nothing about content, and since 2026-08-15 the closed-class
    STOPWORDS (the/of/is/…), which draw low mass from many valid continuations — ride along
    without counting or breaking a run, and never form one on their own). Two-threshold
    HYSTERESIS (2026-08-15, from the confidence_coloring isolate): a run OPENS on tokens under
    the enter threshold and, once building, EXTENDS through tokens under the looser exit
    threshold (`exit_threshold`) instead of one p=0.21 token closing it mid-phrase — the onset
    floor is unchanged, hysteresis only governs the TAIL. A gap in the ledger (a chunk that
    carried no logprobs) always breaks a run: unmeasured text is never bridged. Single low
    tokens are noise (an open synonym choice, a sentence start); the STRUNG-TOGETHER run is the
    hallucination signature this feature marks.

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


# The exit threshold's default relation to the enter threshold: LOOSER by this factor (a run
# already open survives a token that is merely unlikely, not surprising), capped so it can never
# swallow ordinary prose. `runtime.confidence_exit_threshold` overrides with an absolute value.
_EXIT_FACTOR = 1.5
_EXIT_CAP = 0.95


def exit_threshold(enter: "float | None" = None) -> float:
    """The probability under which an already-open run keeps extending (>= the enter threshold).
    Config `runtime.confidence_exit_threshold` wins when set and sane; otherwise derived."""
    th = threshold() if enter is None else float(enter)
    try:
        raw = get_config().get("runtime.confidence_exit_threshold", None)
        if raw is not None:
            v = float(raw)
            if th <= v <= 1.0:
                return v
    except Exception:
        pass
    return min(_EXIT_CAP, th * _EXIT_FACTOR)


# Closed-class words are never graded on their own: function words draw low mass from many valid
# continuations, so a low-probability "the" says nothing about content. Casefolded whole-token
# match after trimming whitespace and punctuation; the '-prefixed entries are BPE contraction
# tails (from the confidence_coloring isolate's stoplist).
STOPWORDS = frozenset("""
a an the and or but nor so yet for if then than that because although though while whereas unless
until since as whether once when where why how of in to on at by with from up down into onto over
under about against between among through during before after above below off out around near per
via within without upon across behind beyond toward towards along am is are was were be been being
have has had having do does did doing will would shall should can could may might must ought not i
me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she
her hers herself it its itself they them their theirs themselves this these those who whom whose
which what there here it's i'm i've i'll i'd you're you've you'll you'd he's she's we're we've
we'll they're they've they'll that's there's what's who's let's isn't aren't wasn't weren't don't
doesn't didn't won't wouldn't can't couldn't shouldn't hasn't haven't hadn't 's 't 're 'll 've 'd
'm n't
""".split())

_TRIM_CHARS = " \t\r\n.,;:!?()[]{}<>\"`*_-/\\|~^&%$#@+="


def is_stopword(tok: str) -> bool:
    """Whether a token is a closed-class word (neutral for run grading)."""
    t = str(tok).replace("\u2019", "'").casefold()
    if t.strip() in STOPWORDS:
        return True
    core = t.strip(_TRIM_CHARS)
    if not core:
        return False
    return core in STOPWORDS or core.strip("'") in STOPWORDS


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
             min_run: int = _MIN_RUN, exit_p: "float | None" = None) -> list[tuple[int, int]]:
    """The character ranges to mark red: runs of >= `min_run` consecutive low-probability
    tokens over `text`, per the module docstring's rules — a run opens on tokens under
    `threshold_p` (the enter threshold) and, once building, extends through tokens under
    `exit_p` (hysteresis; derived from the enter threshold when None). Entries must be in text
    order (every producer appends in stream order). Edges are trimmed to non-whitespace so a
    mark never starts on the space before a word."""
    th = threshold() if threshold_p is None else float(threshold_p)
    ex = max(th, exit_threshold(th) if exit_p is None else float(exit_p))
    runs: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []  # the tokens of the run being built (enter- or exit-low)
    n_enter = 0                       # how many of them are under the ENTER threshold (the floor)

    def close() -> None:
        nonlocal n_enter
        if n_enter >= min_run:
            s, e = cur[0][0], cur[-1][1]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                runs.append((s, e))
        cur.clear()
        n_enter = 0

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
        neutral = not any(ch.isalnum() for ch in tok) or is_stopword(tok)
        if neutral:
            continue  # rides along: neither counts toward, nor breaks, the run
        p = math.exp(min(lp, 0.0))
        if p < th:
            cur.append((s, e))
            n_enter += 1
        elif cur and p < ex:
            cur.append((s, e))  # hysteresis: an open run survives a merely-unlikely token
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

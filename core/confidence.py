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
  - `low_runs(entries, text)` — the display question: which character ranges should be marked?
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

import bisect
import math

from config import get_config

# A token counts LOW below this sampled probability (runtime.confidence_threshold overrides).
# LOWER = stricter = fewer, higher-confidence-of-uncertainty marks; RAISE = more aggressive.
_DEFAULT_THRESHOLD = 0.20

# Consecutive low (non-neutral) tokens before a run is worth marking — one uncertain token is
# an open word choice; three strung together is the drifting-generation signature.
_MIN_RUN = 3

# The runner option (2026-09-02). Ollama 0.33.2 runs qwen3.8 under multi-token-prediction
# speculative decoding (`llama-server --spec-type draft-mtp --spec-draft-backend-sampling`, see
# ~/.ollama/logs/server.log), and llama-server reports a logprob only for the target-sampled token
# of each draft batch — every draft-ACCEPTED token arrives unmeasured, so a 40-token qwen3.8:27b
# answer carried exactly ONE entry on every API surface (/api/chat, /api/generate, /v1, streamed
# or not), while qwen3.5 and qwen3-vl (no drafter) report every token. The 2026-08-16 reading
# "logprobs on the first chunk only" was this. `draft_num_predict: 0` turns drafting off:
# measured 11/12 chunks with logprobs. A daemon that doesn't know the option logs a WARN and
# ignores it; models without a drafter are unaffected.
#
# It is a runner LOAD option, not a sampling option: a request carrying it relaunches
# llama-server without the spec flags, and the next request WITHOUT it relaunches with them —
# each time reloading the weights (~25s for the 27b). Sent on the logprob requests only (the
# first cut), every qwen3.8 turn reloaded at least twice. So the decision is per TOGGLE STATE,
# never per role (`runner_options`): grading on -> EVERY chat request carries it; off -> none
# does. It is read live, so `/confidence on|off` mid-session costs exactly one reload (the
# requests before and after the toggle each agree among themselves). `core.llms` applies it at the one chokepoint all chat traffic passes; the raw
# continuation stream adds it itself. Cost: none — drafting was SLOWER on an M-series Mac with or
# without logprobs (7-8.7 vs 12.2-12.5 tok/s in the daemon's own timings), presumably because
# the verify pass re-scores every token anyway.
LOGPROB_OPTIONS: dict = {"draft_num_predict": 0}


def runner_options() -> dict:
    """The options EVERY chat request must carry so the daemon's runner is loaded once per
    toggle state (LOGPROB_OPTIONS while grading is on, nothing when it is off) — every request
    agrees, so the only reload is the one a `/confidence on|off` costs. A new dict."""
    return dict(LOGPROB_OPTIONS) if enabled() else {}


def request_options(options: "dict | None" = None) -> dict:
    """`options` plus what a logprob-carrying request needs REGARDLESS of the grading toggle
    (LOGPROB_OPTIONS) — the calibration measurement, a standalone process that must read every
    token. A new dict — the caller's is never mutated — and the caller's keys (num_ctx above
    all) are all kept. In-process callers don't need this: core.llms adds runner_options()."""
    return {**(options or {}), **LOGPROB_OPTIONS}


def enabled() -> bool:
    """Whether confidence grading is on (`runtime.confidence`, default true). Fail-open to the
    default — an unreadable config must not silently change what the stream requests."""
    try:
        return bool(get_config().get("runtime.confidence", True))
    except Exception:
        return True


def _synthesizer_model() -> str:
    """The model id serving the synthesizer — the one whose calibration applies to the answer."""
    try:
        return str(get_config().model_for_role("synthesizer").model)
    except Exception:
        return ""


def calibration_for(model: str) -> "dict | None":
    """The calibrated {enter, exit, …} record for `model` (tag case-insensitive), or None.

    The USER's overlay wins (core/confidence_store — written by /confidence tune|set), then the
    SHIPPED table (core/confidence_calibration, regenerated by
    utilities/confidence_calibrate.py). An unreadable overlay degrades to the shipped table:
    marking is additive, and a confidence failure must never cost the answer."""
    tag = str(model or "").lower()
    try:
        from core import confidence_store

        rec = confidence_store.entry_for(tag)
        if rec:
            return rec
    except Exception:
        pass

    from core import confidence_calibration as table  # generated data module

    try:
        return table.CALIBRATION.get(tag)
    except Exception:
        return None


def _configured_threshold():
    """The raw config value: a float (an explicit fixed override), or None for `auto` /
    absent / garbage (→ the per-model table, then the built-in default)."""
    try:
        raw = get_config().get("runtime.confidence_threshold", "auto")
    except Exception:
        return None
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None  # "auto" — or garbage, which must never pick a threshold at random


def threshold() -> float:
    """The low-token probability threshold, in resolution order: an explicit numeric
    `runtime.confidence_threshold` pin, then the synthesizer model's calibrated `enter` —
    the user's overlay first, then the shipped table (calibration_for) — then the built-in
    default. `auto` (the default) means "skip the pin"."""
    fixed = _configured_threshold()
    if fixed is not None:
        return fixed
    rec = calibration_for(_synthesizer_model())
    try:
        if rec and 0.0 < float(rec["enter"]) < 1.0:
            return float(rec["enter"])
    except (KeyError, TypeError, ValueError):
        pass
    return _DEFAULT_THRESHOLD


# The exit threshold's default relation to the enter threshold: LOOSER by this factor (a run
# already open survives a token that is merely unlikely, not surprising), capped so it can never
# swallow ordinary prose. `runtime.confidence_exit_threshold` overrides with an absolute value.
_EXIT_FACTOR = 1.5
_EXIT_CAP = 0.95


def derive_exit(enter: float) -> float:
    """The exit threshold DERIVED from an enter threshold (looser by `_EXIT_FACTOR`, capped at
    `_EXIT_CAP`). THE one home for that derivation: `exit_threshold`'s fallback and
    `/confidence set`'s omitted-exit branch must never drift apart — a command reaching into
    the private factor/cap to recompute it by hand is how they would."""
    return min(_EXIT_CAP, float(enter) * _EXIT_FACTOR)


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
    # Under `auto` the calibrated pair belongs together: the table's exit rides with its enter.
    if enter is None and _configured_threshold() is None:
        rec = calibration_for(_synthesizer_model())
        try:
            if rec and th <= float(rec["exit"]) <= 1.0:
                return float(rec["exit"])
        except (KeyError, TypeError, ValueError):
            pass
    return derive_exit(th)


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


def _resolve_pair(threshold_p: "float | None", exit_p: "float | None") -> tuple[float, float]:
    """The (enter, exit) threshold pair — resolved TOGETHER. `exit_threshold()` with no argument
    is what consults the model's calibrated `exit`; handing it the enter value we just resolved
    looks harmless but pins its `enter is None` guard shut, so it falls through to the derived
    1.5x — which is how the calibrated exit column, `/confidence tune` and `/confidence set
    <enter> <exit>` came to feed a number no renderer ever read, while `/confidence` displayed
    it. A caller-pinned enter derives its own exit — a typed enter never mixes with a measured
    exit."""
    th = threshold() if threshold_p is None else float(threshold_p)
    if exit_p is not None:
        ex = float(exit_p)
    elif threshold_p is None:
        ex = exit_threshold()
    else:
        ex = exit_threshold(th)
    return th, max(th, ex)


def grade_start(entries, text: str, pos: int, threshold_p: "float | None" = None,
                exit_p: "float | None" = None, max_lookback: int = 4096) -> int:
    """The earliest index in `entries` grading may start from so that every run `low_runs` would
    mark at or beyond char offset `pos` is reproduced EXACTLY — the windowed live tail's seam
    (grading the whole ledger per repaint is quadratic in answer length).

    A fixed entry margin is not enough: a run OPENS on `_MIN_RUN` tokens under the enter
    threshold but EXTENDS indefinitely through hysteresis, so a run longer than any margin
    whose opening lies above the slice would lose its in-window tail. Instead, walk back from
    the first entry reaching `pos` to the nearest run BREAKER — a ledger gap, a malformed
    entry, or a measured non-neutral token at/above the exit threshold — since no run can
    extend across one. Bounded by `max_lookback` (a pathological all-low answer would
    otherwise walk the whole ledger per repaint); hitting the bound degrades to the old margin
    behavior, never a crash."""
    entries = entries or []
    _th, ex = _resolve_pair(threshold_p, exit_p)
    # First entry that reaches into the visible window (entries are in text order).
    i = bisect.bisect_right(
        entries, pos, key=lambda ent: ent.get("end", 0) if isinstance(ent, dict) else 0
    )
    floor = max(0, i - max_lookback)
    j = i
    while j > floor:
        ent = entries[j - 1]
        try:
            s, e, lp = int(ent["start"]), int(ent["end"]), float(ent["logprob"])
        except (KeyError, TypeError, ValueError):
            return j  # malformed entry: low_runs closes every run here — safe boundary
        nxt = entries[j] if j < len(entries) else None
        if isinstance(nxt, dict) and nxt.get("start") != e:
            return j  # ledger gap: unmeasured text is never bridged — safe boundary
        e = min(e, len(text))
        if e <= s or s >= len(text):
            j -= 1
            continue  # out-of-range entry: low_runs skips it without breaking a run
        tok = text[s:e]
        if not any(ch.isalnum() for ch in tok) or is_stopword(tok):
            j -= 1
            continue  # neutral: rides along without breaking
        if math.exp(min(lp, 0.0)) >= ex:
            return j  # a confident content token closes any open run — safe boundary
        j -= 1
    return floor


def low_runs(entries, text: str, threshold_p: "float | None" = None,
             min_run: int = _MIN_RUN, exit_p: "float | None" = None) -> list[tuple[int, int]]:
    """The character ranges to mark (the renderers use tui.ui._base._LOW_CONF_STYLE): runs of
    >= `min_run` consecutive low-probability
    tokens over `text`, per the module docstring's rules — a run opens on tokens under
    `threshold_p` (the enter threshold) and, once building, extends through tokens under
    `exit_p` (hysteresis; derived from the enter threshold when None). Entries must be in text
    order (every producer appends in stream order). Edges are trimmed to non-whitespace so a
    mark never starts on the space before a word."""
    th, ex = _resolve_pair(threshold_p, exit_p)
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

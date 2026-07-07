"""
The provenance-tagged answer buffer — the canonical artifact of an interrupt-and-correct turn.

While the final answer streams, it accumulates in a buffer that tracks WHO authored every
character range: `model` spans (generated) and `human` spans (typed at the freeze editor). The
buffer — not "the model's output" — is the canonical turn artifact: generation appends to it,
the freeze editor edits it, and everything downstream reads from it (the continuation prompt
reads clean `text`; the TUI marks human spans distinctly; the trace/audit record and the /trace
replay carry the spans and edit records). The model itself never sees provenance markers —
`text` is clean; the two representations diverge on purpose.

Shape (plain dicts, like the plan — gotcha #4: state must round-trip the checkpointer):

    {"text": str,
     "spans": [{"start": int, "end": int, "author": "model"|"human"}, ...],   # cover text exactly
     "edits": [{"at": int, "cut": str, "typed": str}, ...],                   # audit previews
     "confidence": [{"start": int, "end": int, "logprob": float}, ...]}       # per-token overlay

`confidence` is the token-confidence overlay (core/confidence.py builds + grades it): unlike the
author spans it does NOT tile the text — it covers exactly the model-generated characters whose
logprobs the daemon reported, and gaps mean "unmeasured", never "confident". An edit clears the
overlay inside the changed region (a human-typed character has no model confidence) and shifts
the suffix like the spans.

plus a `state` key managed by the ENGINE (nodes/synthesize + nodes/answer_gate), which this
module preserves but never reads — provenance is pure text+span math.

Every operation is copy-and-return, never in-place mutation: checkpointing, resume, and audit
replay all depend on an earlier buffer staying exactly what it was.

Edits operate on the buffer as a STRING (the user selects characters; the model thinks in
tokens): a truncation may cut mid-token, and that is fine — the continuation layer re-tokenizes
the whole assistant prefix at resume time, so no token-index surgery can ever produce an
invalid sequence.

Leaf module: imports only textutil, so the tests exercise it fully offline.
"""

from __future__ import annotations

from textutil import clip

MODEL = "model"
HUMAN = "human"

# Edit-record previews (`cut`/`typed`) are clipped for the audit trail — the spans mark the exact
# ranges and the resulting `text` is carried in full, so the record loses nothing by bounding the
# quoted excerpts.
_EDIT_PREVIEW_CAP = 200


def new_buffer() -> dict:
    return {"text": "", "spans": [], "edits": [], "confidence": []}


def _merged(spans: list[dict]) -> list[dict]:
    """Normalize a span list: drop empties, merge adjacent same-author runs. Spans arrive
    in text order (every producer appends/rebuilds in order), so one linear pass suffices."""
    out: list[dict] = []
    for s in spans:
        if s["end"] <= s["start"]:
            continue
        if out and out[-1]["author"] == s["author"] and out[-1]["end"] == s["start"]:
            out[-1] = {**out[-1], "end": s["end"]}
        else:
            out.append(dict(s))
    return out


def append_model(buf: dict, text: str, confidence=None) -> dict:
    """A new buffer with `text` appended as model-authored (streamed tokens land here).
    `confidence`, when given, is the chunk's CHUNK-RELATIVE confidence entries
    (core/confidence.align_chunk) — shifted onto the buffer's offsets here, so callers never
    track the running length themselves."""
    if not text:
        return dict(buf)
    old = buf.get("text", "")
    spans = list(buf.get("spans") or [])
    spans.append({"start": len(old), "end": len(old) + len(text), "author": MODEL})
    conf = list(buf.get("confidence") or [])
    for c in confidence or []:
        conf.append({"start": int(c["start"]) + len(old), "end": int(c["end"]) + len(old),
                     "logprob": float(c["logprob"])})
    return {**buf, "text": old + text, "spans": _merged(spans), "confidence": conf}


def apply_edit(buf: dict, new_text: str) -> dict:
    """A new buffer whose text is `new_text`, with the changed region recorded as ONE
    human-authored span and an audit edit record. The change is located by longest common
    prefix/suffix — the freeze editor produces a single contiguous edit (truncate-and-append or
    replace-in-place), so that diff is exact for the operations offered. A no-op edit returns a
    plain copy (no span change, no edit record)."""
    old = buf.get("text", "")
    if new_text == old:
        return dict(buf)

    # Longest common prefix, then longest common suffix over what remains — the two must not
    # overlap, or a repeated region would be counted twice.
    limit = min(len(old), len(new_text))
    p = 0
    while p < limit and old[p] == new_text[p]:
        p += 1
    s = 0
    while s < limit - p and old[len(old) - 1 - s] == new_text[len(new_text) - 1 - s]:
        s += 1

    cut = old[p:len(old) - s]
    typed = new_text[p:len(new_text) - s]
    shift = len(new_text) - len(old)

    spans: list[dict] = []
    for sp in buf.get("spans") or []:
        # The part of the old span living in the untouched prefix survives as-is …
        if sp["start"] < p:
            spans.append({"start": sp["start"], "end": min(sp["end"], p), "author": sp["author"]})
    if typed:
        spans.append({"start": p, "end": p + len(typed), "author": HUMAN})
    for sp in buf.get("spans") or []:
        # … and the part living in the untouched suffix survives shifted by the length delta.
        tail_start = len(old) - s
        if sp["end"] > tail_start:
            spans.append({
                "start": max(sp["start"], tail_start) + shift,
                "end": sp["end"] + shift,
                "author": sp["author"],
            })

    # The confidence overlay follows the same prefix/suffix split: entries wholly in the
    # untouched prefix survive, entries wholly in the untouched suffix shift, and anything
    # touching the changed region is DROPPED — the text those tokens measured no longer exists
    # (and the human's replacement carries no model confidence at all).
    tail_start = len(old) - s
    conf: list[dict] = []
    for c in buf.get("confidence") or []:
        cs, ce = int(c.get("start", 0)), int(c.get("end", 0))
        if ce <= p:
            conf.append(dict(c))
        elif cs >= tail_start:
            conf.append({**c, "start": cs + shift, "end": ce + shift})

    edits = list(buf.get("edits") or [])
    edits.append({"at": p, "cut": clip(cut, _EDIT_PREVIEW_CAP), "typed": clip(typed, _EDIT_PREVIEW_CAP)})
    return {**buf, "text": new_text, "spans": _merged(spans), "edits": edits, "confidence": conf}


def human_spans(buf: dict) -> list[tuple[int, int]]:
    """The human-authored character ranges, for display marking and the audit record."""
    return [
        (int(s["start"]), int(s["end"]))
        for s in buf.get("spans") or []
        if s.get("author") == HUMAN and s.get("end", 0) > s.get("start", 0)
    ]


def corrected(buf) -> bool:
    """Whether this buffer carries at least one human edit — the fact the receipt, the rail
    echo, and the answer render all key on. Tolerates None/garbage (absent-as-no)."""
    return bool(isinstance(buf, dict) and buf.get("edits"))

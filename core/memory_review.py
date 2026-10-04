"""
Learn at session end, gated — the review pass that turns the memory notepad into something that
grows.

Learnable signal lives in the compaction summary, the steer notes / gate denials inside a
turn's state, and the trace. A failed tool call is not one: it says what went wrong once (a
missing permission, a malformed argument), not what holds about the user — the trace and the
answer's incidents note already carry it. This module collects it as CANDIDATES — typed, provenance-stamped proposals for the
memory file — and puts every one in front of the user before anything is written:

  collect_turn(state, run_id)     after each interactive turn: mechanical candidates from the
                                  turn's steer notes (agent) and gate denials (negative).
                                  Appended to the PENDING file, never to memory.
  note_compaction(summary, run)   when auto-compaction fires: the summary's bullets as memo
                                  candidates (the summary itself is persisted beside the
                                  memory file as last_summary.md — the record, not a fact).
  llm_candidates(messages)        optional: the model proposes facts from the session's own
                                  words — what the user typed and what Saturn answered, never
                                  a tool result or a summary (memory.review_llm). Same pending
                                  queue, same gate; a session that read outside content says
                                  so on each proposal.
  run_review(candidates, ask, …)  the screen: each candidate rendered as a `+` diff line
                                  against the memory file, accepted one at a time (y / n / e to
                                  edit / a for all / q to stop; Ctrl-C or Ctrl-D = q). Accepted facts land through
                                  memory_registry.add_memory with by=inferred and the source run
                                  id; rejected ones are dropped; the rest stay pending.

Never a silent write: this is Claude Code's auto-memory with the approval gate in front of it.
The pending queue survives a crash or a bare Ctrl-D (the next launch says how many candidates
are waiting), so learning is deferred, never lost — and never applied unattended.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain.messages import HumanMessage

import diag
from core.state import STEER_PREFIX, this_turn
from stores.memory_registry import _atomic_write, normalize_layer
from textutil import clip, visible_text

# Candidate sources, in the order the review lists them.
SOURCES = ("steer", "gate", "compaction", "model")
_SOURCE_LABEL = {
    "steer": "you corrected the agent mid-task",
    "gate": "you declined a tool at the gate",
    "compaction": "from the compacted conversation — the model's summary, not your words",
    "model": "proposed by the model from this session",
}

_MAX_PER_TURN = 12
_MAX_PENDING = 60
_SUMMARY_BULLETS = 8

# What `ask` returns when the user interrupts the review (Ctrl-C / Ctrl-D): the caller passes it
# as ui.ask's on_interrupt so the interrupt resolves in the review's own vocabulary — "stop,
# leave the rest pending" — never as the empty reply a y/N prompt reads as "drop".
INTERRUPT = "\x03"


def _candidate(layer: str, text: str, source: str, run_id=None, *, due=None,
               category: str = "general", outside: bool = False) -> dict | None:
    # Cleaned as the write boundary cleans it (memory_registry._clean_text), so the line the
    # user says yes to is the text that is stored.
    text = " ".join(visible_text(str(text or "")).split())
    if not text:
        return None
    return {
        "layer": normalize_layer(layer),
        "text": clip(text, 300),
        "source": source if source in SOURCES else "model",
        "run": int(run_id) if isinstance(run_id, int) and run_id > 0 else None,
        "due": due or None,
        "category": category or "general",
        # The session it came from read content from outside the trust boundary (a web page,
        # mail, a file): said on the review screen, so a yes is given knowing it.
        "outside": bool(outside),
    }


# ── mechanical candidates from one turn ───────────────────────────────────────────────────────

def collect_turn(state: dict, run_id=None) -> list[dict]:
    """Mechanical candidates from a finished turn's state. Pure (no I/O) — the REPL appends the
    result to the pending queue with `add_pending`. Deterministic sources only:

      steer notes      → agent      "when asked …, the user corrected: …"
      gate denials     → negative   "the user declined <tool> for: <step>"
    """
    out: list[dict] = []
    query = clip(" ".join(str(state.get("current_query") or "").split()), 90)
    ctx = f" (while: {query})" if query else ""

    for m in this_turn(state.get("messages") or []):
        # A steer is a standalone STEER_PREFIX HumanMessage (nodes/agent.py); the merged form
        # older records carry ("<query>\n<STEER_PREFIX> <reason>") is read the same way.
        if not isinstance(m, HumanMessage) or STEER_PREFIX not in str(m.content):
            continue
        for segment in str(m.content).split(STEER_PREFIX)[1:]:
            note = segment.strip().splitlines()[0].strip() if segment.strip() else ""
            if note:
                out.append(_candidate("agent", f"The user corrected me mid-task{ctx}: {note}",
                                      "steer", run_id))

    for ev in state.get("gate_events") or []:
        if not isinstance(ev, dict) or ev.get("decision") == "approved":
            continue
        step = ev.get("step")
        for call in ev.get("calls") or []:
            if isinstance(call, dict) and not call.get("approved") and call.get("name"):
                what = f" for: {step}" if step else ctx
                out.append(_candidate("negative",
                                      f"The user declined {call['name']} at the gate{what}",
                                      "gate", run_id))

    return [c for c in out if c][:_MAX_PER_TURN]


def summary_candidates(summary: str, run_id=None) -> list[dict]:
    """The bullets of a compaction summary as memo candidates (the summary's own structure:
    COMPACTION_PROMPT asks for terse bullets — facts established, decisions, preferences, open
    threads). Preference-shaped lines land in the user layer, open-thread-shaped ones in
    commitments; the rest are dated memo notes."""
    out: list[dict] = []
    for raw in str(summary or "").splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if len(line) < 12:
            continue
        low = line.lower()
        layer = "memo"
        if low.startswith(("user prefers", "the user prefers", "preference", "user wants",
                           "the user wants", "user likes", "the user likes")):
            layer = "user"
        elif low.startswith(("open:", "open thread", "todo", "to do", "pending", "unfinished",
                             "next:", "next step", "still to")):
            layer = "commitments"
        out.append(_candidate(layer, line, "compaction", run_id))
        if len(out) >= _SUMMARY_BULLETS:
            break
    return [c for c in out if c]


# ── the pending queue (paths.memory's sibling) ────────────────────────────────────────────────

def _memory_dir() -> Path:
    from config import get_config
    return get_config().path("memory").parent


def pending_path() -> Path:
    return _memory_dir() / "pending_review.json"


def summary_path() -> Path:
    return _memory_dir() / "last_summary.md"


def load_pending() -> list[dict]:
    path = pending_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("candidates") if isinstance(data, dict) else data
        # A queue written before failed tool calls stopped being candidates still holds them.
        return [c for c in (items or []) if isinstance(c, dict) and c.get("text")
                and c.get("source") != "failed"]
    except Exception as exc:
        diag.log(f"memory review: pending queue unreadable ({exc}) — starting empty")
        return []


def save_pending(candidates: list[dict]) -> None:
    path = pending_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not candidates:
            if path.exists():
                path.unlink()
            return
        _atomic_write(path, json.dumps({"saved_at": datetime.now().isoformat(timespec="seconds"),
                                        "candidates": candidates[-_MAX_PENDING:]}, indent=1))
    except Exception as exc:
        diag.log(f"memory review: pending queue write failed: {exc}")


def add_pending(candidates: list[dict]) -> int:
    """Append candidates to the queue, skipping ones already pending or already remembered
    (same text, case-insensitive) and any that holds a credential (the queue is a plain-text
    file too — memory_registry.secret_problem). Returns how many were added. Best-effort."""
    from stores.memory_registry import secret_problem

    new = [c for c in (candidates or [])
           if c and c.get("text") and not secret_problem(c["text"])]
    if not new:
        return 0
    try:
        from stores.memory_registry import entries

        known = {e["text"].lower() for e in entries()}
    except Exception:
        known = set()
    pending = load_pending()
    seen = {c["text"].lower() for c in pending} | known
    added = 0
    for c in new:
        key = c["text"].lower()
        if key in seen:
            continue
        pending.append(c)
        seen.add(key)
        added += 1
    if added:
        save_pending(pending)
    return added


def note_compaction(summary: str, run_id=None) -> int:
    """Persist a compaction summary beside the memory file (the last session's brief — a record
    the user can read, never loaded as a fact) and queue its bullets for review."""
    text = str(summary or "").strip()
    if not text:
        return 0
    try:
        path = summary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="minutes")
        _atomic_write(path, f"# Last compaction summary ({stamp})\n\n{text}\n")
    except Exception as exc:
        diag.log(f"memory review: summary persist failed: {exc}")
    return add_pending(summary_candidates(text, run_id))


# ── model-proposed candidates ─────────────────────────────────────────────────────────────────

def llm_enabled() -> bool:
    try:
        from config import get_config
        return bool(get_config().get("memory.review_llm", True))
    except Exception:
        return True


def llm_candidates(messages: list, run_id=None) -> list[dict]:
    """Ask the model for durable facts worth keeping from this session — from `own_words`:
    what the user typed and what Saturn answered, never a tool result or a summary.
    Proposals only — every one still faces the review screen. ONE constrained call (a flat JSON
    schema for the decoder plus the shape hint as a trailing HumanMessage — never a
    SystemMessage, which Ollama rejects mid-conversation for qwen3.8 models); the outermost
    {...} is salvaged from prose-wrapped output. Empty on any failure (the mechanical candidates
    stand on their own). Monkeypatched in tests; never reached offline."""
    if not messages:
        return []
    try:
        from langchain.messages import HumanMessage
        from pydantic import BaseModel

        from core import provenance
        from core.llms import generate, get_model, invoke_kwargs, model_tag
        from core.messages import MEMORY_REVIEW_FORMAT, MEMORY_REVIEW_PROMPT, MEMORY_REVIEW_SHAPE

        class _Item(BaseModel):
            layer: str = "memo"
            text: str = ""

        class _Proposal(BaseModel):
            facts: list[_Item] = []

        transcript = own_words(messages)
        if not transcript.strip():
            return []
        outside = provenance.of({"messages": messages}).untrusted
        messages = [HumanMessage(content=MEMORY_REVIEW_PROMPT + transcript),
                    HumanMessage(content=MEMORY_REVIEW_SHAPE)]
        resp = generate(get_model(), messages, tag=model_tag(),
                        **invoke_kwargs(MEMORY_REVIEW_FORMAT, 0.0, task="memory_review"))
        content = str(getattr(resp, "content", "") or "")
        start, end = content.find("{"), content.rfind("}")
        out = _Proposal.model_validate_json(content[start:end + 1]) if end > start >= 0 else _Proposal()
    except Exception as exc:
        diag.log(f"memory review: model proposals failed: {exc}")
        return []
    cands = []
    for item in (out.facts or [])[:_MAX_PER_TURN]:
        c = _candidate(item.layer, item.text, "model", run_id, outside=outside)
        if c:
            cands.append(c)
    return cands


# ── the review screen ─────────────────────────────────────────────────────────────────────────

def render_line(c: dict) -> str:
    """One candidate as a diff line against the memory file: `+ [layer] text  (source · run)`,
    and whether the session that produced it read outside content."""
    src = _SOURCE_LABEL.get(c.get("source"), c.get("source") or "")
    run = f" · run #{c['run']}" if c.get("run") else ""
    outside = " · this session read outside content" if c.get("outside") else ""
    return f"+ [{c.get('layer', 'user')}] {c.get('text', '')}  ({src}{run}{outside})"


def own_words(messages) -> str:
    """The transcript the model pass reads: what the user typed (core/provenance.is_typed —
    requests and steer notes) and what Saturn answered, nothing else. No tool result, no
    compaction summary, no tool-calling preamble: text Saturn fetched or wrote for itself is
    where a planted "the user is…" lives, and extracting from what was injected is how a memory
    fills with junk (spec 2026-10-04-know-the-user R1). "" when the user typed nothing."""
    from langchain.messages import AIMessage

    from core.compaction import _text
    from core.provenance import is_typed

    lines, typed = [], False
    for m in messages or []:
        if is_typed(m):
            typed = True
            lines.append(f"User: {_text(m)}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and _text(m):
            lines.append(f"Assistant: {_text(m)}")
    return "\n".join(lines) if typed else ""


_HELP = "y = keep · n = drop · e = edit then keep · a = keep all · q = stop (rest stay pending)"


def run_review(candidates: list[dict], *, ask, emit=print) -> dict:
    """Walk the candidates one at a time. `ask(prompt) -> str` reads the decision (ui.ask in the
    app; a scripted callable in tests); `emit(line)` prints. Accepted facts are written through
    add_memory (by=inferred, with the source run id and, for commitments, the due date); the
    return value is `{"accepted": [...], "rejected": [...], "remaining": [...]}` — the caller
    persists `remaining` as the new pending queue."""
    from stores.memory_registry import SecretRefused, add_memory

    accepted: list[dict] = []
    rejected: list[dict] = []
    remaining: list[dict] = []
    keep_all = False
    items = list(candidates or [])
    for i, c in enumerate(items):
        if keep_all:
            decision = "y"
        else:
            emit(f"  {render_line(c)}")
            while True:
                decision = str(ask(f"    keep? [{i + 1}/{len(items)}] y / N / e / a / q » ") or "")
                decision = "q" if decision == INTERRUPT else decision.strip().lower()
                if decision not in ("?", "h", "help"):
                    break
                emit(f"    {_HELP}")  # help answers the question, then asks again
        if decision in ("q", "quit", "stop"):
            remaining.extend(items[i:])
            break
        if decision in ("a", "all"):
            keep_all = True
            decision = "y"
        text = c.get("text", "")
        if decision in ("e", "edit"):
            edited = str(ask("    new text » ") or "")
            if edited == INTERRUPT:
                remaining.extend(items[i:])
                break
            edited = edited.strip()
            if not edited:
                rejected.append(c)
                emit("    dropped (empty edit)")
                continue
            text = edited
            decision = "y"
        if decision in ("y", "yes", "k", "keep"):
            try:
                report = add_memory(text, c.get("category") or "general",
                                    layer=c.get("layer") or "user", by="inferred",
                                    run_id=c.get("run"), due=c.get("due"))
            except SecretRefused as exc:  # never stored, and never left waiting in the queue
                rejected.append(c)
                emit(f"    {exc}")
                continue
            except Exception as exc:
                report = f"could not store: {exc}"
                remaining.append(c)
                emit(f"    {report}")
                continue
            accepted.append({**c, "text": text, "report": report})
            emit(f"    {report}")
        else:
            rejected.append(c)
    return {"accepted": accepted, "rejected": rejected, "remaining": remaining}

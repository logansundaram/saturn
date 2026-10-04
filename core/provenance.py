"""
Where the conversation's text came from — the one reading of provenance the gate's holds share.

  of(state) -> Provenance(typed, seen, untrusted)

`typed` is every message the user typed, in order: each turn's request and each mid-turn steer
note (core.state.is_turn_start / is_steer_message — a compaction summary is neither: it is the
model's words about earlier turns). `seen` is everything else that ENTERED the conversation
(completed tool results, attachments, the grounding, a summary). `untrusted` is whether any of
that came from outside the trust boundary: an attachment (`@file`, `@clipboard`, piped stdin,
`!cmd` output) or a ToolMessage from a tool declared untrusted (trust.quarantine.is_untrusted).

The model's own messages are skipped: they are what the holds check, so a URL the model wrote
in a preamble (the issuing message is already in state) or an earlier answer must not vouch
for itself. A tool result counts as `seen` only when the call COMPLETED (`saturn_status`
done): a refusal, a decline or an error is text about the model's own arguments, and usually
repeats them ("+1305… appears nowhere in this conversation"), so counting it let one retry of
an invented number through (review 2026-10-03). A failed call still counts as outside content
having entered.

Read by the URL hold and the handle / chat holds (nodes/approval.provenance →
trust.quarantine), by auto-learn (core/auto_memory) and by the memory review's transcript
(core/memory_review): all ask "did the user type this, or could something else have written
it?", and all must answer from the same facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.messages import AIMessage, ToolMessage

from core.state import is_steer_message, is_turn_start
from trust import quarantine


@dataclass(frozen=True)
class Provenance:
    typed: tuple
    seen: str
    untrusted: bool


def of(state) -> Provenance:
    typed: list[str] = []
    seen = [str(state.get("attachments") or ""), str(state.get("context") or "")]
    untrusted = bool(state.get("attachments"))
    for m in state.get("messages") or []:
        text = str(getattr(m, "content", "") or "")
        if is_turn_start(m) or is_steer_message(m):
            typed.append(text)
            continue
        if isinstance(m, AIMessage):
            continue
        if isinstance(m, ToolMessage):
            if quarantine.is_untrusted(str(m.name or "")):
                untrusted = True
            if ((getattr(m, "additional_kwargs", None) or {}).get("saturn_status") or "done") != "done":
                continue
        seen.append(text)
    return Provenance(tuple(typed), "\n".join(seen), untrusted)

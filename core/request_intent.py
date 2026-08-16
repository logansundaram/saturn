"""
Deterministic REQUEST-ONLY intent detectors (transplanted from the engine isolate, 2026-08-15).

The engine's completeness and authorization checks (execute's ask gate, rectify's request-side
branches, the effect-authorization rule) need a handful of yes/no readings of what the HUMAN asked
for. Each is a plain regex over the human's own words — never over a tool result, never over a
model's paraphrase — so a file's contents or a web page can never manufacture intent, and every
decision they feed stays deterministic (model judgment may tighten to ask, never grant).

Leaf module: stdlib only.
"""

from __future__ import annotations

import re

# ── "the request names a source the engine can search" ─────────────────────────────────────
#
# Motivating failure: "Search my notes and tell me when my passport expires" produced a plan
# whose single step was `ask_user`. The user named a searchable source and was interrogated instead.

_SEARCHABLE_RE = re.compile(
    r"\b(?:my\s+(?:notes?|files?|documents?|docs|knowledge\s*base|workspace|records?)"
    r"|the\s+(?:knowledge\s*base|workspace|notes?)"
    r"|search|look\s+(?:up|through|in)|find\s+(?:in|my)|check\s+my)\b"
)


def names_searchable_source(request) -> bool:
    """Whether the request points at material the engine could look in for itself."""
    return bool(_SEARCHABLE_RE.search(str(request or "").lower()))


# ── "the user asked to be asked" ───────────────────────────────────────────────────────────
#
# A question the user explicitly requested is the task, not a routing failure: "Ask me which
# colour to use, then save that colour" IS the interrupting-tool (ask_user) seam.

_ASK_INVITED_RE = re.compile(
    r"\b(?:ask|check\s+with|confirm\s+with|prompt|query)\s+"
    r"(?:me|us|the\s+user|the\s+human)\b"
    r"|\bask\s+(?:for\s+)?my\b"
    r"|\blet\s+me\s+(?:choose|decide|pick)\b"
)


def invites_a_question(request) -> bool:
    """Whether the request itself asks the engine to put a question to the user."""
    return bool(_ASK_INVITED_RE.search(str(request or "").lower()))


# ── "the user asked for the workspace to CHANGE" ───────────────────────────────────────────
#
# The authorization half of the effect-authorization rule (`plan_context.request_authorized`).
# "Read vendor_terms.txt and tell me the late fee" asks for NO state change at all — the file's
# contents must not be able to put `write_file breach_marker.txt` into the redrafted plan.
# COMMUNICATION verbs are deliberately absent: "Send an email to Petra" is an effect on the world,
# not the workspace, and treating it as write authorization is how a turn writes
# email_to_petra.txt and calls it sent.

_STATE_CHANGE_RE = re.compile(
    r"\b(?:save|saves|saved|saving|write|writes|wrote|writing|store|stores|stored|storing"
    r"|append|appends|appended|appending|create|creates|created|creating"
    r"|delete|deletes|deleted|deleting|remove|removes|removed|removing"
    r"|edit|edits|edited|editing|update|updates|updated|updating"
    r"|rename|renames|renamed|renaming|move|moves|moved|moving|copy|copies|copied|copying"
    r"|record|records|recorded|recording|run|runs|ran|running|execute|executes|executed)\b"
)

# "add the total to notes.md" — the one multi-word form.
_ADD_TO_RE = re.compile(r"\badd(?:s|ed|ing)?\b[^.?!]{0,60}\bto\b")

# Terms above that are also ordinary NOUNS ("search my records", "the run", "a copy"). The
# direction of the error matters: a missed detection costs a blocked write that is DISCLOSED as
# an incident; a false one silently hands an injected step the authorization the gate exists to
# withhold. So an ambiguous term counts only where it cannot be read as a noun.
_AMBIGUOUS_TERMS = frozenset(
    "record records recording run runs running copy copies move moves "
    "store stores update updates edit edits".split()
)
_NOUN_MARKERS = frozenset(
    "a an the my our your his her its their this that these those "
    "any some no every each another other".split()
)
_PRECEDING_WORD_RE = re.compile(r"([a-z']+)[^a-z']*$")


def _reads_as_noun(text: str, start: int) -> bool:
    m = _PRECEDING_WORD_RE.search(text[:start])
    return bool(m) and m.group(1) in _NOUN_MARKERS


def wants_state_change(request) -> bool:
    """Whether the request asks for the WORKSPACE to change."""
    text = str(request or "").lower()
    if _ADD_TO_RE.search(text):
        return True
    return any(
        m.group(0) not in _AMBIGUOUS_TERMS or not _reads_as_noun(text, m.start())
        for m in _STATE_CHANGE_RE.finditer(text)
    )

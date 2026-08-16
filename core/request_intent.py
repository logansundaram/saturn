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

"""
The request-side complexity check (2026-09-08 — PLAN.md, "The common-case contract").

Decides, from the human's own words and nothing else, whether a turn takes the QUICK path
(nodes/quick.py: one router call under a grammar, read-only tools only, no planner and no
judge) or the plan engine. A plain regex — zero tokens, deterministic, the same readings
core/request_intent already trusts — and it errs toward the engine: a request the engine
handles unnecessarily is merely slower, while a request the quick path cannot finish is handed
over mid-way with its observations (the quick node's escalation), so the cheap direction of
error is "plan".

Plan mode is forced by:
  - a state change: write/edit/delete/create/remember/…, the calendar, notes, mail and
    reminder effects (request_intent.wants_state_change) — the quick path has no writing tool;
  - a change verb the effect vocabulary deliberately leaves out ("change 'one' to 'two'",
    "replace", "rename", "modify", "fix");
  - a computed figure (an aggregation word, or an arithmetic expression) — a figure the answer
    owes must come from a calculate step the trace can show;
  - a reference hop ("the file it names"), a request to be asked, more than one workspace path,
    or a multi-clause request ("… then …", "for each …", "…; …", ", and read …").

One exemption: "write/draft/compose me a story" asks for prose, not a file. It stays simple
unless a file, a save, or a path is also named.

Measured before shipping (docs/OPTIMIZATIONS.md §8): over the 31 distinct traced requests the
check agreed with the plan that actually ran on 24, sent 5 to the quick path that escalated
(a news digest, follow-ups to a refused appointment), and sent 2 to the engine that a quick
answer would have served ("write me a story" without the exemption above).
"""

from __future__ import annotations

import re

from core import request_intent as ri
from core.plan_context import target_tokens

# Change verbs that name an edit without a workspace-effect word the authorization vocabulary
# counts (request_intent keeps "change"/"replace" out on purpose — they authorize nothing). Here
# the direction of error is reversed: a false positive costs the engine's latency, a miss costs a
# read-only turn that reads the file and reports it cannot edit it.
_CHANGE_RE = re.compile(
    r"\b(?:change|changes|changed|changing|replace|replaces|replaced|replacing"
    r"|modify|modifies|modified|modifying|rename|renames|renamed|renaming"
    r"|fix|fixes|fixed|fixing|correct|corrects|corrected|correcting"
    r"|insert|inserts|inserted|inserting|convert|converts|converted|converting"
    r"|overwrite|overwrites|overwriting|truncate|truncates|truncating)\b"
)

# "… then …", "for each …", a semicolon, or a second imperative after ", and" / "and then".
# "tell/show/report" are deliberately absent: "read roster.txt and tell me who is on call" is
# one read followed by the answer, which the planner itself drafts as a single step.
_MULTI_CLAUSE_RE = re.compile(
    r"\bthen\b|\bafter that\b|\bafterwards\b|\bfor each\b|\bfor every\b|;"
    r"|\b(?:and|,)\s+(?:then\s+)?(?:read|write|save|list|search|find|compute|calculate"
    r"|summari[sz]e|compare|count|total|extract|open|check|look)\b"
)

# The prose exemption: "write me a story", "draft us a toast" — the object is the user, and the
# request names nothing to persist it in.
_WRITE_ME_RE = re.compile(r"\b(?:write|writes|draft|drafts|compose|composes)\s+(?:me|us)\b")
_PERSIST_RE = re.compile(r"\b(?:file|files|disk|folder|directory|save|saves|saved|saving)\b")


def plan_reason(request) -> str:
    """Why this request needs the plan engine — "" when it can take the quick path."""
    text = str(request or "").strip()
    if not text:
        return ""
    low = text.lower()
    if _WRITE_ME_RE.search(low) and not _PERSIST_RE.search(low) and not target_tokens(text):
        # Judge the rest of the request without the prose verb: "write me a story and save it"
        # keeps its "save"; "write me a story" has nothing left to force the engine.
        low = _WRITE_ME_RE.sub(" ", low)
    if ri.wants_state_change(low):
        return "the request asks for a change"
    if _CHANGE_RE.search(low):
        return "the request asks for an edit"
    if ri.wants_derived_number(low) or ri.states_an_expression(low):
        return "the request wants a computed figure"
    if ri.names_deferred_reference(low):
        return "the request follows a reference"
    if ri.invites_a_question(low):
        return "the request asks to be asked"
    if len(target_tokens(text)) > 1:
        return "the request names several paths"
    if _MULTI_CLAUSE_RE.search(low):
        return "the request has several parts"
    return ""


def is_simple(request) -> bool:
    """Whether the request takes the quick path (see the module docstring)."""
    return not plan_reason(request)

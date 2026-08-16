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


# ── "the request wants a number that has to be DERIVED" ────────────────────────────────────
#
# Motivating failure: "Read ledger_alpha.csv and ledger_beta.csv and tell me which one has the
# larger total" repeatedly produced two reads and nothing else, so the totals were computed in
# the ANSWER'S PROSE — untraceable by construction. Vocabulary is restricted to AGGREGATION —
# words that name a value the `calculate` tool produces. Counting words ("how many", "count")
# are deliberately absent: "how many vessels are listed" is answered by reading a file.

_AGGREGATION_TERMS = (
    "total", "totals", "totalled", "totaled", "totalling", "totaling",
    "sum", "sums", "summed", "subtotal", "grand total",
    "add up", "adds up", "added up", "adding up",
    "average", "averages", "averaged",
    "product of", "multiply", "multiplied",
    "percentage", "percent",
    "difference between",
    "compute", "computes", "computed", "computing",
    "calculate", "calculates", "calculated", "calculating", "calculator",
    "work out", "works out", "worked out",
)
_AGGREGATION_RE = re.compile(
    r"\b(?:" + "|".join(t.replace(" ", r"\s+") for t in _AGGREGATION_TERMS) + r")\b"
)

# A bare arithmetic expression, e.g. "847 * 293" or "(55 + 21) * 0.4". `-` and `x` are only
# honored with spaces on BOTH sides: `2026-07-30` and `q3-2026` are not subtractions, and
# `1920x1080` is not a product.
_EXPRESSION_RE = re.compile(r"\d\s*[+*/×]\s*\d|\d\s+[-x]\s+\d")


def wants_derived_number(request) -> bool:
    """Whether the request asks for a figure that must be COMPUTED rather than read."""
    text = str(request or "").lower()
    return bool(_AGGREGATION_RE.search(text) or _EXPRESSION_RE.search(text))


def states_an_expression(request) -> bool:
    """Whether the request contains the arithmetic itself ("847 * 293") — computable with nothing
    gathered, unlike "the difference between a stack and a queue" (an aggregation word in English)."""
    return bool(_EXPRESSION_RE.search(str(request or "").lower()))


# ── "the request defers one of its own targets to a result" ────────────────────────────────
#
# "the file it names" is the user telling the engine, in their own words, that one of this turn's
# targets will come from an earlier result — both the detection signal and the authorization for
# the hop. A file's contents can never activate this: the marker has to be in the human's
# sentence. The referent must be a FILE-LIKE object ("the three amounts it lists" is not a hop).

_DEFERRED_NOUN = r"(?:file|document|doc|page|url|link|report|sheet|path|note|card|attachment)"
_DEFERRED_VERB = (
    r"(?:names|named|points?\s+(?:at|to)|pointed\s+(?:at|to)|refers?\s+to|references|"
    r"mentions|lists|specifies|identifies)"
)
_DEFERRED_RE = re.compile(
    rf"\b(?:the|whatever|whichever)\s+{_DEFERRED_NOUN}\s+"
    rf"(?:it|that|which|they|this)?\s*{_DEFERRED_VERB}\b"
    rf"|\b{_DEFERRED_NOUN}\s+named\s+in\b"
)


def names_deferred_reference(request) -> bool:
    """Whether the request itself defers a target to something an earlier step will produce."""
    return bool(_DEFERRED_RE.search(str(request or "").lower()))


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

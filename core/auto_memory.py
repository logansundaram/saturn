"""
Auto-learn — a fact the user stated in their own words is remembered without the gate.

`remember` is side_effecting, so by default every call faces the approval gate. That click is
the right price for a fact the model inferred, or one whose words came from a web page, an
email or a file: memory is a persistence channel, and a planted fact reads as trusted context
on every later turn. It is the wrong price for "I'm vegetarian" typed by the user. This module
tells the two apart DETERMINISTICALLY — never by asking a model, which an injected page can
talk into "the user said…":

  why_not(call, state)   None when this `remember` call may skip the gate, else the reason
                         (shown at the gate). It may skip it only when ALL hold:
                           · memory.auto_learn is on (headless turns it off — app/headless.py);
                           · nothing from outside the trust boundary is in the conversation
                             (core/provenance: no attachment, no untrusted tool's result);
                           · every content word of the fact (and of a non-generic category)
                             appears in text the user typed — a turn request or a steer note,
                             each shorter than a paste chip — after light stemming;
                           · a `replaces=#id` drops only words the user typed (Berlin may
                             replace Paris when the user said "not Paris"): a model must not
                             silently delete an unrelated fact;
                           · the fact is short (MAX_FACT_CHARS).
  qualifies(call, state) why_not(...) is None — the approval node exempts the call and the
                         tools node stamps the fact by=user src=said.
  rule_layer(fact, layer)a standing rule the model filed under `negative` ("never…",
                         "don't…", "from now on…") moves to `user`, which loads every turn —
                         `negative` loads only by token match, so "nothing before 10am" would
                         not load for "book the dentist" and the rule would silently not apply.
  fact_id(report)        the #id in add_memory's report, for the after-answer note.

Glue words (pronouns, articles, "user", "is") need not be typed; meaning-bearing words do —
including polarity ("not", "never", "always") and sentiment ("likes", "hates"), so a restatement
that flips the user's meaning ("User likes cilantro" for "I hate cilantro") faces the gate.
"""

from __future__ import annotations

import re

from config import get_config
from core import provenance
from core.state import STEER_PREFIX

# Words a restatement may add without the user having typed them: they name the user or
# carry grammar, never a fact. Polarity and sentiment words are deliberately NOT here.
GLUE = frozenset("""
user user's users the a an is are am was were be been being has have had do does did
i i'm im i've i'd i'll me my mine myself we we're our ours us you your yours they them their
theirs he him his she her hers it its it's this that these those there here of to in on at by
for with as from into about and or but so also too just now currently called named would
should please remember saturn assistant
""".split())
# Categories the remember tool's description suggests: labels, not facts.
GENERIC_CATEGORIES = frozenset("""
general preference preferences identity project projects person people rule rules fact facts
personal work family health money diet food location contact contacts schedule
""".split())
_KEEP_SHORT = frozenset({"no"})          # a two-letter word that carries meaning
_TOKEN_RE = re.compile(r"[^\W_]+(?:['.@+:/_-][^\W_]+)*")
_RULE_RE = re.compile(r"\b(?:always|never|from now on|do not|don't|dont|every time|whenever)\b",
                      re.IGNORECASE)
_REPORT_ID_RE = re.compile(r"^(?:Remembered|Already remembered as) #(\d+)")

# A message this long, or with this many lines, is treated as pasted, not typed — the same
# thresholds at which the prompt compacts a paste into a [paste #N] chip (tui/ui/prompt.py).
MAX_TYPED_CHARS = 600
MAX_TYPED_LINES = 3
MAX_FACT_CHARS = 200


def enabled() -> bool:
    return bool(get_config().get("memory.auto_learn", True))


def _stem(tok: str) -> str:
    if tok.endswith("'s"):
        tok = tok[:-2]
    if not tok.isalpha():
        return tok  # numbers, times, addresses: exact
    for suf in ("ing", "ies", "es", "ed", "s", "e"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def content_words(text) -> dict:
    """{stem: the word as written} for every meaning-bearing word in `text`."""
    out: dict = {}
    for tok in _TOKEN_RE.findall(str(text or "").lower().replace("’", "'")):
        if tok in GLUE or (tok.isalpha() and len(tok) < 3 and tok not in _KEEP_SHORT):
            continue
        out.setdefault(_stem(tok), tok)
    return out


def uncovered(fact, typed) -> list:
    """The fact's content words that appear in none of the `typed` texts, as written."""
    have: set = set()
    for t in typed:
        have |= content_words(t).keys()
    return [word for stem, word in content_words(fact).items() if stem not in have]


def typed_texts(state) -> list:
    """What the user typed in this conversation (turn requests and steer notes, the steer
    prefix removed), minus anything long enough to have been pasted."""
    out = []
    for text in provenance.of(state).typed:
        if text.startswith(STEER_PREFIX):
            text = text[len(STEER_PREFIX):]
        if len(text) <= MAX_TYPED_CHARS and text.count("\n") < MAX_TYPED_LINES:
            out.append(text)
    return out


def _replaced_id(raw) -> int:
    try:
        return int(str(raw).strip().lstrip("#"))
    except (TypeError, ValueError):
        return 0


def why_not(call: dict, state) -> "str | None":
    if not enabled():
        return "auto-learn is off (memory.auto_learn)"
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    fact = " ".join(str(args.get("fact") or "").split())
    if call.get("name") != "remember" or not fact:
        return "not a fact to remember"
    if len(fact) > MAX_FACT_CHARS:
        return f"the fact is longer than {MAX_FACT_CHARS} characters"
    if provenance.of(state).untrusted:
        return ("content from outside — a web page, mail, a file or an attachment — is in this "
                "conversation, so only you can confirm what is remembered")
    typed = typed_texts(state)
    category = str(args.get("category") or "")
    words = fact if category.strip().lower() in GENERIC_CATEGORIES else f"{fact} {category}"
    missing = uncovered(words, typed)
    if missing:
        shown = ", ".join(repr(w) for w in missing[:4])
        return f"{shown} {'is' if len(missing) == 1 else 'are'} not in anything you typed"
    rid = _replaced_id(args.get("replaces")) if args.get("replaces") not in (None, "", 0) else 0
    if rid > 0:
        from stores.memory_registry import entry

        old = entry(rid)
        mentioned: set = set()
        for t in typed:
            mentioned |= content_words(t).keys()
        # What the replacement drops from the old fact ("paris" when Berlin replaces Paris)
        # must be something the user named; a pure refinement drops nothing.
        dropped = content_words(old["text"]).keys() - content_words(fact).keys() if old else set()
        if dropped and not dropped & mentioned:
            return f"it would replace #{rid} ({old['text']!r}), which nothing you typed mentions"
    return None


def qualifies(call: dict, state) -> bool:
    return why_not(call, state) is None


def rule_layer(fact: str, layer: str) -> str:
    """The layer a remembered fact lands in: a standing rule filed under `negative` goes to
    `user` (loaded every turn); everything else stays where the model put it."""
    from stores.memory_registry import normalize_layer

    if normalize_layer(layer) == "negative" and _RULE_RE.search(str(fact or "")):
        return "user"
    return layer


def fact_id(report: str) -> "int | None":
    m = _REPORT_ID_RE.match(str(report or ""))
    return int(m.group(1)) if m else None

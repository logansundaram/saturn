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
                           · nothing from outside the trust boundary has entered the
                             conversation, this turn or any earlier one (core/provenance: no
                             attachment, no untrusted tool's result, `outside_seen`);
                           · the fact has at least one word of substance, and every one of its
                             words (and of a non-generic category or sensitivity mark) that is
                             not glue appears in ONE SENTENCE the user typed — a turn request
                             or a steer note, each shorter than a paste chip — after light
                             inflection matching. Words gathered from several sentences are
                             the model's composition, not the user's statement;
                           · the user STATED it: the words do not come from a question ("Is
                             Petra my manager?", "what if I were vegetarian?");
                           · its "not" is where the user put it: the clauses the words come
                             from negate exactly when the fact does ("I'm not vegetarian"
                             never yields "User is vegetarian");
                           · the layer is one of memory's own;
                           · a `replaces=#id` drops only words the user typed (Berlin may
                             replace Paris when the user said "not Paris"): a model must not
                             silently delete an unrelated fact;
                           · the fact is short (MAX_FACT_CHARS).
  qualifies(call, state) why_not(...) is None — the approval node exempts the call and hands
                         its id to the tools node (state["user_stated"]), which stamps the
                         fact by=user src=said.
  rule_layer(fact, layer)a standing rule the model filed under `negative` ("never…",
                         "don't…", "from now on…") moves to `user`, which loads every turn —
                         `negative` loads only by token match, so "nothing before 10am" would
                         not load for "book the dentist" and the rule would silently not apply.
  fact_id(report)        the #id in add_memory's report, for the after-answer note.
  similar(fact, layer)   the stored facts a new one may contradict — same layer, sharing at
                         least half of the smaller fact's content words. A replacement happens
                         only when the model passes `replaces=#id`; when it does not, "I live
                         in Berlin" lands beside "I live in Paris" and a small model follows
                         whichever it reads. So every surface that reports a write names the
                         neighbours (the after-answer note, the gate, /memory add) and the user
                         resolves it. Nothing is retired automatically: word overlap cannot
                         tell a correction from an elaboration.

What the check proves, and what it does not. It proves the user typed the fact's words in one
statement. It does not prove the fact means what they meant: inside one sentence a restatement
can still reorder ("I hate cilantro but my sister loves sushi" → "User loves cilantro"). That
is what the line after the answer is for — every fact saved this way is shown with its undo.
"""

from __future__ import annotations

import re

from config import get_config
from core import provenance
from core.state import STEER_PREFIX

# Words a restatement may add without the user having typed them: they name the user or
# carry grammar, never a fact. Polarity and sentiment words are deliberately NOT here, and
# neither are the words an instruction is made of ("should", "would", "just", "please",
# "remember"): a fact must not be able to say anything with glue alone.
GLUE = frozenset("""
user user's users the a an is are am was were be been being has have had do does did
i i'm im i've i'd i'll me my mine myself we we're our ours us you your yours they them their
theirs he him his she her hers it its it's this that these those there here of to in on at by
for with as from into about and or but so also too now currently called named saturn assistant
""".split())
# Categories the remember tool's description suggests: labels, not facts.
GENERIC_CATEGORIES = frozenset("""
general preference preferences identity project projects person people rule rules fact facts
personal work family health money diet food location contact contacts schedule
""".split())
# Sensitivity marks the tool's description names: marks, not facts.
GENERIC_SENSITIVITY = frozenset("health money private sensitive medical financial".split())
# Words that turn a statement around. A fact carries one exactly when its source clause does.
NEGATIONS = frozenset("not no never none nothing nobody neither nor without cannot".split())
# A clause that opens with one of these is asked or supposed, not stated.
_UNSTATED_OPENERS = frozenset("""
what who whom whose where when why how which if whether suppose supposing imagine maybe
perhaps pretend
""".split())
# ...and one that opens with an auxiliary, in a sentence that ends in "?", is a question.
_QUESTION_OPENERS = frozenset("""
is are am was were do does did can could will would should shall may might have has had
""".split())
_ABBREVIATIONS = frozenset("mr mrs ms dr prof st jr sr vs etc approx".split())
_TOKEN_RE = re.compile(r"[^\W_]+(?:['.@+:/_-][^\W_]+)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;:]\s|\s[—–-]+\s|[—–])\s*")
_RULE_RE = re.compile(r"\b(?:always|never|from now on|do not|don't|dont|every time|whenever)\b",
                      re.IGNORECASE)
_REPORT_ID_RE = re.compile(r"^(?:Remembered|Already remembered as) #(\d+)")
# Verbs that take many values at once ("I like tea", "I like hiking"): sharing only one of
# these says nothing about whether two facts are about the same thing, and a replacement may
# drop one without the user having named it.
_MANY_VALUED = frozenset("like love hate prefer enjoy want need use dislike".split())
SIMILAR_MAX = 2

# A message this long, or with this many lines, is treated as pasted, not typed — the same
# thresholds at which the prompt compacts a paste into a [paste #N] chip (tui/ui/prompt.py:
# `n_lines < _PASTE_TAG_LINES and len(data) < _PASTE_TAG_CHARS` inserts verbatim).
MAX_TYPED_CHARS = 600
MAX_TYPED_LINES = 3
MAX_FACT_CHARS = 200


def enabled() -> bool:
    return bool(get_config().get("memory.auto_learn", True))


# ── words ─────────────────────────────────────────────────────────────────────────────────────

def _normalise(text) -> str:
    """Lowercase, straight apostrophes, and a contraction's "not" written out — "don't" and
    "do not" are the same statement, and the negation must be visible to the polarity check."""
    text = str(text or "").lower().replace("’", "'")
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    return re.sub(r"(\w)n't\b", r"\1 not", text)


def content_words(text) -> list:
    """The words of substance in `text`, as written, in order, once each: every token that is
    not glue. No word is too short to count — a fact chunked into two-letter pieces must not
    read as having nothing to check."""
    out, seen = [], set()
    for tok in _TOKEN_RE.findall(_normalise(text)):
        if tok in GLUE or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _lemmas(tok: str) -> set:
    """The forms one word may be matched under. Plural and tense endings are undone without
    dropping a final "e", so "lives" meets "live" and "moved" meets "move", but "hats" never
    meets "hates". Numbers, times and addresses match exactly."""
    if tok.endswith("'s"):
        tok = tok[:-2]
    out = {tok}
    if not tok.isalpha():
        return out
    n = len(tok)
    if tok.endswith("ies") and n > 4:
        out.add(tok[:-3] + "y")
    elif tok.endswith("es") and n > 4 and tok[:-2].endswith(("s", "x", "z", "ch", "sh")):
        out.add(tok[:-2])
    elif tok.endswith("s") and not tok.endswith("ss") and n > 3:
        out.add(tok[:-1])
    for suffix in ("ing", "ed"):
        if tok.endswith(suffix) and n > len(suffix) + 2:
            base = tok[: -len(suffix)]
            out |= {base, base + "e"}
            if len(base) > 2 and base[-1] == base[-2]:
                out.add(base[:-1])          # stopped → stop
            if suffix == "ed" and base.endswith("i"):
                out.add(base[:-1] + "y")    # tried → try
    return out


def _pool(words) -> set:
    out: set = set()
    for w in words:
        out |= _lemmas(w)
    return out


def _in(word: str, pool: set) -> bool:
    return bool(_lemmas(word) & pool)


def uncovered(fact, typed) -> list:
    """The fact's content words that appear in none of the `typed` texts, as written."""
    have: set = set()
    for t in typed:
        have |= _pool(content_words(t))
    return [w for w in content_words(fact) if not _in(w, have)]


def typed_texts(state) -> list:
    """What the user typed in this conversation (turn requests and steer notes, the steer
    prefix removed), minus anything long enough to have been pasted."""
    out = []
    for text in provenance.of(state).typed:
        if text.startswith(STEER_PREFIX):
            text = text[len(STEER_PREFIX):]
        if len(text) < MAX_TYPED_CHARS and text.count("\n") + 1 < MAX_TYPED_LINES:
            out.append(text)
    return out


# ── statements ────────────────────────────────────────────────────────────────────────────────

def _sentences(text: str) -> list:
    """`text` cut into sentences, an abbreviation's full stop ("Dr. Núñez") not ending one."""
    out: list = []
    for piece in _SENTENCE_END_RE.split(str(text or "")):
        piece = piece.strip()
        if not piece:
            continue
        last = out[-1].rstrip(".").rsplit(None, 1)[-1].lower() if out else ""
        if out and out[-1].endswith(".") and last in _ABBREVIATIONS:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _stated_clauses(sentence: str) -> list:
    """The clauses of one sentence that STATE something: not the clause a "?" ends, not one
    that opens with a question or supposing word, and — in a sentence that ends in "?" — not
    one that opens with an auxiliary ("Is Petra my manager, or Sam?")."""
    asked = sentence.rstrip().endswith("?")
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
    out = []
    for i, clause in enumerate(clauses):
        first = (_TOKEN_RE.findall(_normalise(clause)) or [""])[0]
        if asked and (i == len(clauses) - 1 or first in _QUESTION_OPENERS):
            continue
        if first in _UNSTATED_OPENERS:
            continue
        out.append(clause)
    return out


def _not_stated(words: list, typed: list) -> "str | None":
    """Why `words` (all of them typed somewhere) do not amount to something the user SAID, or
    None when one sentence states them with the same polarity."""
    reason = None
    meaning = [w for w in words if w not in NEGATIONS]
    fact_negates = len(meaning) < len(words)
    for message in reversed(typed):
        for sentence in _sentences(message):
            if not all(_in(w, _pool(content_words(sentence))) for w in words):
                continue
            stated = _stated_clauses(sentence)
            if not all(_in(w, _pool(content_words(" ".join(stated)))) for w in words):
                reason = reason or "you asked or supposed it — you did not state it"
                continue
            source = [c for c in stated
                      if any(_in(w, _pool(content_words(c))) for w in meaning)]
            source_negates = any(w in NEGATIONS for c in source for w in content_words(c))
            if source_negates != fact_negates:
                reason = "its 'not' is not where you put it"
                continue
            return None
    return reason or "its words do not come from one sentence you typed"


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
    from stores.memory_registry import LAYERS, entry, normalize_layer

    layer = str(args.get("layer") or "user")
    if normalize_layer(layer) not in LAYERS:
        return f"the layer {layer!r} is not one of memory's own"
    if provenance.of(state).untrusted:
        return ("content from outside — a web page, mail, a file or an attachment — has "
                "entered this conversation, so only you can confirm what is remembered")
    typed = typed_texts(state)
    # The free-text arguments that are written beside the fact must be the user's words too.
    extras = [str(args.get(key) or "") for key, generic in
              (("category", GENERIC_CATEGORIES), ("sensitivity", GENERIC_SENSITIVITY))
              if str(args.get(key) or "").strip().lower() not in generic]
    words = content_words(" ".join([fact, *extras]))
    if not content_words(fact):
        return "it holds no word of substance to check against what you typed"
    missing = uncovered(" ".join([fact, *extras]), typed)
    if missing:
        shown = ", ".join(repr(w) for w in missing[:4])
        return f"{shown} {'is' if len(missing) == 1 else 'are'} not in anything you typed"
    unstated = _not_stated(words, typed)
    if unstated:
        return unstated
    rid = _replaced_id(args.get("replaces")) if args.get("replaces") not in (None, "", 0) else 0
    if rid > 0:
        old = entry(rid)
        # Everything the replacement drops from the old fact ("paris" when Berlin replaces
        # Paris) must be something the user named; a pure refinement drops nothing. A verb
        # of taste or a negation may go unnamed — every other word may not, or one shared
        # word would let a model retire an unrelated fact.
        mentioned = _pool(w for t in typed for w in content_words(t))
        kept = _pool(content_words(fact))
        dropped = [w for w in (content_words(old["text"]) if old else [])
                   if not _in(w, kept) and w not in NEGATIONS and not _in(w, _MANY_VALUED)]
        unnamed = [w for w in dropped if not _in(w, mentioned)]
        if unnamed:
            return (f"it would replace #{rid} ({old['text']!r}), and nothing you typed "
                    f"mentions {unnamed[0]!r}")
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


def similar(fact: str, layer: str = "user", *, exclude=()) -> list:
    """The stored facts in `layer` that a new `fact` may contradict or repeat: they share at
    least half of the smaller fact's content words, and not only a many-valued verb. Most
    shared first, then newest; at most SIMILAR_MAX. The same text is the dedup path
    (add_memory), not a neighbour."""
    from stores.memory_registry import entries, normalize_layer

    new = content_words(fact)
    if not new:
        return []
    text = " ".join(str(fact or "").split()).lower()
    skip = {int(i) for i in exclude if i}
    found = []
    for e in entries(normalize_layer(layer)):
        if e.get("id") in skip or e["text"].lower() == text:
            continue
        old = content_words(e["text"])
        shared = [w for w in new if _in(w, _pool(old))]
        if (any(not _in(w, _MANY_VALUED) for w in shared)
                and len(shared) * 2 >= min(len(new), len(old))):
            found.append((len(shared), e.get("id") or 0, e))
    found.sort(key=lambda t: (-t[0], -t[1]))
    return [e for _n, _id, e in found[:SIMILAR_MAX]]


def similar_names(near) -> str:
    """`#1 "I live in Paris", #5 "…"` — the neighbours as every surface names them."""
    from textutil import clip

    return ", ".join(f'#{e["id"]} "{clip(str(e["text"]), 60)}"' for e in near)


def similar_note(near) -> str:
    """The line under a write that landed beside a related fact, with the way to resolve it."""
    ids = [e["id"] for e in near]
    tail = (f"/memory forget {ids[0]} if that is no longer true" if len(ids) == 1
            else "/memory forget <n> if one is no longer true")
    return f"similar: {similar_names(near)} — {tail}"


def fact_id(report: str) -> "int | None":
    m = _REPORT_ID_RE.match(str(report or ""))
    return int(m.group(1)) if m else None

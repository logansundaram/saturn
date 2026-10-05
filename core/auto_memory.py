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
                           · the user did not decline a `remember` earlier in this turn: a
                             "no" at the gate is not answered by rewording the fact;
                           · the fact has at least one word of substance, and every one of its
                             words (and of a non-generic category or sensitivity mark) that is
                             not glue appears in ONE SENTENCE the user typed — a turn request
                             or a steer note, on a line that carried no paste
                             (core.state.PASTED_KEY, set where the prompt sees the paste) and
                             shorter than a paste chip — after light inflection matching.
                             Words gathered from several sentences are the model's
                             composition, not the user's statement;
                           · the user STATED it: the words do not come from a question ("Is
                             Petra my manager?", "is Petra my manager", "tell me if Petra is
                             my manager") or a sentence that supposes or sets a condition
                             ("what if I were vegetarian?", "When I travel, I'm vegetarian");
                           · it says WHOLE CLAUSES: the fact's words are exactly the words of
                             one or more clauses of that sentence, nothing left out but a
                             lead-in ("remember that…") or a title. "My brother is
                             vegetarian" never yields "User is vegetarian", nor "I used to
                             be vegetarian", nor "I'm vegetarian on weekdays". A "not" is a
                             word like any other, so it stays where the user put it, and so
                             is the tense ("I was vegetarian", "I worked at Acme" are not
                             what the user is or does now);
                           · it leaves no clause behind that changes what it says: a list
                             item or conjunct comes with the clause it hangs off unless that
                             clause is plainly the user's own ("My brother is tall and
                             vegetarian" is not "User is vegetarian"; "I don't have a car, a
                             dog…" is not "User has a dog"), and a clause that qualifies,
                             takes back or reports the statement cannot be dropped ("On
                             weekdays, I'm vegetarian", "I'm vegetarian, not really", "My
                             sister said, I'm vegetarian") — an aside can ("By the way, …");
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

What the check proves, and what it does not. It proves the user typed the fact's words as
whole stated clauses of one sentence. It does not prove the fact means what they meant: a
restatement that keeps every word can still reorder them ("I hate cilantro but my sister
loves sushi" → "User's sister hates cilantro; user loves sushi"), and a clause is cut at
commas and "and / but / because", not by grammar — which clause qualifies another, and which
verb is in the past, are read off closed word lists and word endings ("I sold the car and the
dog" still yields "User has a dog": the item's verb is not checked, only whose it is). That is
what the line after the answer is for — every fact saved this way is shown with its undo.
Where the check cannot tell, it asks: "Have two kids" opens like a question and faces the
gate, as does "User works at Acme" after "I work at Acme and at Globex".
"""

from __future__ import annotations

import re

from langchain.messages import ToolMessage

from config import get_config
from core import provenance
from core.state import STEER_PREFIX, this_turn

# Words a restatement may add without the user having typed them: they name the user or
# carry grammar, never a fact. Polarity and sentiment words are deliberately NOT here, and
# neither are the words an instruction is made of ("should", "would", "just", "please",
# "remember"): a fact must not be able to say anything with glue alone.
GLUE = frozenset("""
user user's users the a an is are am be being has have do does
i i'm im i've i'd i'll me my mine myself we we're our ours us you your yours they them their
theirs he him his she her hers it its it's this that these those there here of to in on at by
for with as from into about and or but so also too now currently called named saturn assistant
""".split())
# The past forms of be / have / do are NOT glue: "I was vegetarian" does not say what the user
# is. A replacement may drop one without the user having named it (as with a negation).
_PAST_FORMS = frozenset("was were been had did".split())
# Categories the remember tool's description suggests: labels, not facts.
GENERIC_CATEGORIES = frozenset("""
general preference preferences identity project projects person people rule rules fact facts
personal work family health money diet food location contact contacts schedule
""".split())
# Sensitivity marks the tool's description names: marks, not facts.
GENERIC_SENSITIVITY = frozenset("health money private sensitive medical financial".split())
# Words that turn a statement around. A fact carries one exactly when its source clause does.
# ("cannot" and every "…n't" are written out as "… not" before any word is read: _normalise.)
NEGATIONS = frozenset("not no never none nothing nobody neither nor without".split())
# A clause that opens with one of these is asked, not stated.
_ASKED_OPENERS = frozenset("what who whom whose where why how which".split())
# One that opens with one of these supposes or sets a condition — for its whole sentence:
# "When I travel, I'm vegetarian" does not state "I'm vegetarian".
_SUPPOSED_OPENERS = frozenset("""
if whether when unless suppose supposing imagine maybe perhaps pretend
""".split())
# One that opens with an auxiliary is a question ("Is Petra my manager", with or without
# its "?") — unless it is the imperative "do not …".
_QUESTION_OPENERS = frozenset("""
is are am was were do does did can could will would should shall may might have has had
""".split())
# Words that open a clause without being part of what it says: an address to Saturn or a
# filler. How a clause opens is read past them ("Hey is Petra my manager"), and a fact may
# leave them out ("Remember that I work at Acme" → "User works at Acme").
_LEAD_INS = frozenset("""
remember note fyi btw please hey hi hello ok okay oh well yes yeah yep actually just anyway
""".split())
# ...and the joining words it is read past too. "so" only in a question: "so is Sam" states.
_CONNECTIVES = frozenset("and but or also now then".split())
_SUBJECTS = frozenset("i you we he she they it".split())
# A clause holding none of these (nor an auxiliary) has no subject or verb of its own: it is
# a list item or an afterthought ("a dog", "not vegan") hanging off the clause before it.
_OWN_CLAUSE = _SUBJECTS | frozenset("""
i'm im i've i'd i'll we're we've you're he's she's they're it's there my your our his her their
""".split())
# A clause that opens with one of these, and names nobody else, is plainly about the user: what
# hangs off it ("… and a dog", "… and live in Berlin") is the user's too.
_FIRST_PERSON = frozenset("i i'm im i've i'd i'll we we're we've".split())
_SOMEONE_ELSE = frozenset("""
you he she they it you're he's she's they're it's my your our his her their
""".split())
# A clause that ENDS in one of these has its object elsewhere in the sentence: it reports or
# hedges what another clause says ("My sister said, I'm vegetarian", "I'm vegetarian, I
# think"). The user quoting themselves ("Like I said, …") is still the user stating it.
_SAYING = frozenset("""
said say says saying told tell tells claim claims claimed heard hear hears read reads wrote
writes
""".split())
_SUPPOSING = frozenset("""
think thinks thought believe believes believed guess guessed suppose supposed assume assumed
hope hoped wish wished doubt doubted wonder wondered joke joked joking dreamed dreamt imagine
imagined pretend pretended reckon reckons suspect suspects suspected
""".split())
# A clause with no subject or verb of its own that only talks to Saturn: droppable.
_ASIDES = frozenset("""
way no nope nah never mind nevermind right sure thanks thank sorry honestly seriously frankly
basically obviously look listen record quick side good morning afternoon evening
""".split())
# One that opens with one of these, or is made only of the words below (with or without a
# "not"), says WHEN or HOW FAR the clause before it holds: "…, on weekdays", "…, not really".
_QUALIFIER_OPENERS = frozenset("""
on in at until till since before after during except besides for with without by from to as
like per only especially apart aside
""".split())
_HEDGES = frozenset("""
really anymore any more longer mostly mainly usually sometimes often always yet exactly quite
strictly entirely totally technically lately nowadays almost nearly partly generally normally
typically occasionally rarely seldom probably possibly apparently supposedly allegedly
hypothetically theoretically temporarily formerly previously once ever kind sort much though
""".split())
_ABBREVIATIONS = frozenset("mr mrs ms dr prof st jr sr vs etc approx".split())
_TITLES = frozenset("mr mrs ms dr prof jr sr".split())
_TOKEN_RE = re.compile(r"[^\W_]+(?:['.@+:/_-][^\W_]+)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;:]\s|\s[—–-]+\s|[—–])\s*")
_CONJUNCTION_SPLIT_RE = re.compile(r"\s+(?:and|but|because)\s+", re.IGNORECASE)
_ASKED = "it reads as something you asked or supposed, not something you stated"
_NOT_MOVED = "its 'not' is not where you put it"
_REPORTED = "it reads as something someone said or supposed, not something you stated"
_DECLINED = "you declined a remember earlier in this turn, so this one asks too"
_PASTED = ("its words are on a line you pasted, recalled or typed ahead, not one you typed "
           "at the prompt")
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
    text = re.sub(r"\bcan(?:'t|not)\b", "can not", text)
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


def _plain(tok: str) -> set:
    """`tok` and its form without a possessive or plural ending — never without a tense."""
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
    return out


def _lemmas(tok: str) -> set:
    """The forms one word may be matched under. Plural and tense endings are undone without
    dropping a final "e", so "lives" meets "live" and "moved" meets "move", but "hats" never
    meets "hates". Numbers, times and addresses match exactly. (Matching under a tense ending
    finds the word; whether the tense was kept is `_tense_shift`.)"""
    out = _plain(tok)
    if tok.endswith("'s"):
        tok = tok[:-2]
    if not tok.isalpha():
        return out
    n = len(tok)
    for suffix in ("ing", "ed"):
        if tok.endswith(suffix) and n > len(suffix) + 2:
            base = tok[: -len(suffix)]
            out |= {base, base + "e"}
            if len(base) > 2 and base[-1] == base[-2]:
                out.add(base[:-1])          # stopped → stop
            if suffix == "ed" and base.endswith("i"):
                out.add(base[:-1] + "y")    # tried → try
    return out


def _tense_shift(fact_words: list, said_words: list) -> "str | None":
    """Why a fact whose words were all said is still in another tense, or None: one of its
    words meets what was said only under an "-ed" ending that one of the two lacks ("works" /
    "worked"). An "-ing" form is not a tense: "I'm living in Berlin" says the user lives there."""
    for w in fact_words:
        near = [s for s in said_words if _lemmas(w) & _lemmas(s)]
        if not near or any(_plain(w) & _plain(s) for s in near):
            continue
        if all(w.endswith("ed") != s.endswith("ed") for s in near):
            return f"it says {w!r} where you said {near[0]!r}"
    return None


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


def typed_texts(state, *, pasted: bool = False, prov=None) -> list:
    """What the user typed in this conversation (turn requests and steer notes, the steer
    prefix removed), minus every line not known to be typed by hand — one that carried a
    paste, was recalled from history or was typed ahead, recorded where the line was read
    (provenance.by_hand) — and anything long enough that the prompt would have chipped it.
    `pasted=True` keeps those lines: only to say why a fact is asking. `prov` is the
    conversation's provenance when the caller already read it."""
    p = prov or provenance.of(state)
    out = []
    for text in (p.typed if pasted else p.by_hand):
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


def _opener(tokens: list, asked: bool) -> list:
    """`tokens` from the word a clause really opens with: past lead-ins and joining words
    ("Hey is Petra…", "and is it Sam") and, in a question, a leading "so"."""
    skip = _LEAD_INS | _CONNECTIVES | ({"so"} if asked else set())
    i = 0
    while i < len(tokens) and tokens[i] in skip:
        i += 1
    return tokens[i:]


def _required(tokens: list) -> list:
    """The content words of a clause that a fact drawing on it must keep: all of them, but a
    lead-in before its first word of substance and a title anywhere."""
    out, leading = [], True
    for tok in tokens:
        if tok in GLUE:
            continue
        if leading and tok in _LEAD_INS:
            continue
        leading = False
        if tok not in _TITLES and tok not in out:
            out.append(tok)
    return out


def _user_led(opens: list) -> bool:
    """Whether a clause (its tokens past any lead-in) is plainly about the user: it opens with
    "I" / "we" and names nobody else who could be the subject of what hangs off it — no other
    pronoun or possessive ("I think my brother is tall"), no saying or supposing verb."""
    return bool(opens) and opens[0] in _FIRST_PERSON and not any(
        t in _SOMEONE_ELSE or t in _SAYING or t in _SUPPOSING for t in opens[1:])


def _reports(clause: dict) -> bool:
    """Whether an own clause reports or hedges another: it ends in a supposing verb ("…, I
    think"), or in a saying verb whose subject is not the user ("My sister said, …" — but not
    "Like I said, …")."""
    last = clause["required"][-1] if clause["required"] else ""
    if last in _SUPPOSING:
        return True
    if last not in _SAYING:
        return False
    tokens = clause["tokens"]
    at = len(tokens) - 1 - tokens[::-1].index(last)
    return not (at > 0 and tokens[at - 1] in _FIRST_PERSON)


def _clauses(sentence: str) -> list:
    """One sentence as its clauses, in order — cut at commas, semicolons, colons and dashes,
    then at "and / but / because". Each is {words, required, stated, head, …}:

      stated   it STATES something. Not the clause a "?" ends; not one that opens with a
               question word; not one that opens with an auxiliary — in a sentence that ends
               in "?", or as the first clause after a comma ("is Petra my manager"), the
               imperative "do not …" excepted. Nothing in a sentence that holds a clause
               opening with a supposing word. A clause with no subject or verb of its own is
               stated only when the clause it hangs off is.
      head     for such a clause, the clause it hangs off (the nearest before it with a
               subject or verb of its own), else None.
      leads    for such a clause with no head ("On weekdays, …"), the clause it leads into
               (the nearest after it with a subject or verb of its own), else None.
      own      it has a subject or verb of its own.   tokens  its words as typed.
      opens    the word it opens with, past lead-ins and joining words.
      user_led it is plainly about the user (`_user_led`)."""
    asked = sentence.rstrip().endswith("?")
    parts = []      # (text, first of its comma segment)
    for segment in _CLAUSE_SPLIT_RE.split(sentence):
        pieces = [p for p in _CONJUNCTION_SPLIT_RE.split(segment) if p.strip()]
        parts += [(piece, i == 0) for i, piece in enumerate(pieces)]
    out: list = []
    supposed = False
    owner = None    # the last clause with a subject or verb of its own
    for n, (text, first) in enumerate(parts):
        tokens = _TOKEN_RE.findall(_normalise(text))
        opens = _opener(tokens, asked)
        word = opens[0] if opens else ""
        stated = True
        if asked and n == len(parts) - 1:
            stated = False
        elif word in _ASKED_OPENERS:
            stated = False
        elif word in _SUPPOSED_OPENERS:
            stated, supposed = False, True
        elif word in _QUESTION_OPENERS and (asked or first):
            imperative = (word == "do" and opens[1:2] == ["not"]
                          and not (opens[2:3] and opens[2] in _SUBJECTS))
            stated = imperative
        own = any(t in _OWN_CLAUSE or t in _QUESTION_OPENERS for t in tokens)
        clause = {"words": content_words(text), "required": _required(tokens),
                  "stated": stated, "head": None if own else owner, "leads": None,
                  "own": own, "tokens": tokens, "opens": word,
                  "user_led": own and _user_led(opens)}
        if not own and owner is not None and not owner["stated"]:
            clause["stated"] = False
        if own:
            owner = clause
        out.append(clause)
    if supposed:
        for clause in out:
            clause["stated"] = False
    following = None
    for clause in reversed(out):
        if clause["own"]:
            following = clause
        elif clause["head"] is None:
            clause["leads"] = following
    return out


def _left_out(words: list, fact: set) -> str:
    left = list(dict.fromkeys(w for w in words if not _in(w, fact)))
    if not left:
        return "it leaves out part of what you said"
    return f"it leaves out {', '.join(repr(w) for w in left[:4])} from what you said"


def _dropped_problem(clause: dict, taken, fact: set) -> "str | None":
    """Why a stated clause the fact did NOT take cannot be left behind, or None. A clause with
    a subject and verb of its own says its own thing, and may go — unless it reports or
    hedges the rest (`_reports`). One without hangs off a neighbour: when the fact took that
    neighbour, the clause may go only as an aside ("By the way, …", "…, thanks") or, after
    it, as another item of the same kind ("…, not vegan", "…, coffee"). Before it ("On
    weekdays, …"), or after it as a qualifier ("…, on weekdays", "…, mostly", "…, not
    really"), it changes what the neighbour says."""
    if clause["own"]:
        return _REPORTED if _reports(clause) else None
    neighbour = clause["head"] or clause["leads"]
    if neighbour is not None and not taken(neighbour):
        return None
    required = clause["required"]
    rest = [w for w in required if w not in NEGATIONS]
    negated = len(rest) < len(required)
    if clause["head"] is None:
        # Leading, or a sentence with no subject or verb at all: only an aside may go.
        if all(w in _ASIDES or w in _LEAD_INS for w in required):
            return None
    elif not negated and all(w in _ASIDES or w in _LEAD_INS for w in rest):
        return None
    elif clause["opens"] not in _QUALIFIER_OPENERS and not all(w in _HEDGES for w in rest):
        return None
    return _NOT_MOVED if negated else _left_out(required, fact)


def _sentence_problem(sentence: str, fact_words: list, extra_words: list) -> "str | None":
    """Why `sentence` (which holds all the words) does not STATE the fact, or None when the
    fact's words are exactly those of one or more of its stated clauses."""
    stated = [c for c in _clauses(sentence) if c["stated"] and c["words"]]
    said = _pool(w for c in stated for w in c["words"])
    if not all(_in(w, said) for w in [*fact_words, *extra_words]):
        return _ASKED
    fact = _pool(fact_words)
    chosen = [c for c in stated if all(_in(w, fact) for w in c["required"])]
    covered = _pool(w for c in chosen for w in c["words"])
    missing = [w for w in fact_words if not _in(w, covered)]
    if missing:
        # The clauses the fact draws on without taking all of, and what it left behind.
        meaning = [w for w in fact_words if w not in NEGATIONS]
        touched = [c for c in stated if not any(c is k for k in chosen)
                   and any(_in(w, _pool(c["words"])) for w in meaning)]
        left = [w for c in touched for w in c["required"] if not _in(w, fact)]
        left = list(dict.fromkeys(left))
        if any(w in NEGATIONS for w in left) or all(w in NEGATIONS for w in missing):
            return _NOT_MOVED
        if any(w in _SUPPOSED_OPENERS or w in _ASKED_OPENERS for w in left):
            return _ASKED
        if left:
            shown = ", ".join(repr(w) for w in left[:4])
            return f"it leaves out {shown} from what you said"
        return "its words do not come from one sentence you typed"
    def taken(clause) -> bool:
        return any(clause is k for k in chosen)

    # A list item or conjunct comes with the clause it hangs off — "I don't have a car, a dog"
    # never yields "User has a dog", nor "My brother is tall and vegetarian" "User is
    # vegetarian" — unless that clause is plainly the user's own ("I have a cat and a dog").
    for c in chosen:
        head = c["head"]
        if head is None or taken(head):
            continue
        if any(w in NEGATIONS for w in head["words"]):
            return _NOT_MOVED
        if not head["user_led"]:
            return _left_out(head["required"], fact)
    shifted = _tense_shift(fact_words, [w for c in chosen for w in c["words"]])
    if shifted:
        return shifted
    # ...and nothing the fact left behind may change what it says.
    for c in stated:
        if not taken(c):
            why = _dropped_problem(c, taken, fact)
            if why:
                return why
    return None


def _not_stated(fact_words: list, extra_words: list, typed: list) -> "str | None":
    """Why the fact's words (all of them typed somewhere) do not amount to something the user
    SAID, or None when one sentence states them as whole clauses. `extra_words` (a category
    or sensitivity mark) need only be stated in that sentence."""
    reason = None
    for message in reversed(typed):
        for sentence in _sentences(message):
            have = _pool(content_words(sentence))
            if not all(_in(w, have) for w in [*fact_words, *extra_words]):
                continue
            why = _sentence_problem(sentence, fact_words, extra_words)
            if why is None:
                return None
            reason = reason or why
    return reason or "its words do not come from one sentence you typed"


def _declined_this_turn(state) -> bool:
    """Whether the user said no to a `remember` at the gate earlier in this turn."""
    for m in this_turn(list(state.get("messages") or [])):
        status = (getattr(m, "additional_kwargs", None) or {}).get("saturn_status")
        if isinstance(m, ToolMessage) and m.name == "remember" and status == "skipped":
            return True
    return False


def _replaced_id(raw) -> int:
    try:
        return int(str(raw).strip().lstrip("#"))
    except (TypeError, ValueError):
        return 0


def why_not(call: dict, state, prov=None) -> "str | None":
    """Why this `remember` call may not skip the gate, or None (the module docstring has the
    rule). `prov` is `provenance.of(state)` when the caller already read it — the gate asks
    about every call of a batch, and the conversation is read once."""
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
    prov = prov or provenance.of(state)
    if prov.untrusted:
        return ("content from outside — a web page, mail, a file or an attachment — has "
                "entered this conversation, so only you can confirm what is remembered")
    # A "no" at the gate is a no to remembering, not to one wording of it: the reworded
    # retry asks again (the agent's declined-repeat guard only catches the identical call).
    if _declined_this_turn(state):
        return _DECLINED
    typed = typed_texts(state, prov=prov)
    # The free-text arguments that are written beside the fact must be the user's words too.
    extras = [str(args.get(key) or "") for key, generic in
              (("category", GENERIC_CATEGORIES), ("sensitivity", GENERIC_SENSITIVITY))
              if str(args.get(key) or "").strip().lower() not in generic]
    fact_words = content_words(fact)
    if not fact_words:
        return "it holds no word of substance to check against what you typed"
    extra_words = [w for w in content_words(" ".join(extras)) if w not in fact_words]
    missing = uncovered(" ".join([fact, *extras]), typed)
    if missing:
        if not uncovered(" ".join([fact, *extras]), typed_texts(state, pasted=True, prov=prov)):
            return _PASTED
        shown = ", ".join(repr(w) for w in missing[:4])
        return f"{shown} {'is' if len(missing) == 1 else 'are'} not in anything you typed"
    unstated = _not_stated(fact_words, extra_words, typed)
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
                   if not _in(w, kept) and w not in NEGATIONS and w not in _PAST_FORMS
                   and not _in(w, _MANY_VALUED)]
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

"""
The first-run interview — five questions whose answers become memory facts (docs/pivot.md #5).

Saturn's second turn should already know who it is talking to. After the first launch's
/models pick, the REPL asks what to call you, what you do, the people you mention most, what
you want help with, and what Saturn should never do — and writes each answer through
`memory_registry.add_memory` with by=user: typing the answer IS the user action. No model call:
it runs before a model is pulled, costs nothing, and cannot paraphrase your facts into
something you did not say. The only transformations are a fixed template per question
("Call me …", "What I do: …") and a stated split of the two many-fact answers (people, rules).

Every fact the interview writes carries the category `setup-<question>`: that is how
`/memory setup` finds the current answer on a re-run (Enter keeps it; a new answer supersedes a
one-fact question with replaces=, and adds to a many-fact one). The hyphenated category is ONE
match token that no request contains, so it never changes what loads. The same fact is stamped
`src=setup:<question>`, which is what `/memory why` reads.

  run_interview(ask=, emit=)   the five prompts (a scripted `ask` in tests, ui.ask in the app)
  offer_at_launch(...)          the REPL's once-only decision: interview / one-line hint / nothing
  rule_layer()                  where "never" answers go — a layer loaded EVERY turn

Imports only config, diag and the memory registry, so the REPL and /memory can both use it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from stores import memory_registry as mr

# What `ask` returns on Ctrl-C / Ctrl-D: the caller passes it as ui.ask's on_interrupt, so an
# interrupt means "stop the interview" — never the empty reply that means "skip this one".
INTERRUPT = "\x03"
_LEAVE = ("q", "quit")
_SKIP = ("-", "skip")
# An always-loaded fact costs memory.context_cap (4,000 chars) on every turn: one line, not a page.
MAX_FACT_CHARS = 300


@dataclass(frozen=True)
class Question:
    key: str        # the category suffix: the facts are stored as [setup-<key>]
    prompt: str
    layer: str      # a memory layer, or "rules" — resolved by rule_layer()
    template: str   # one fact per answer piece; {a} is the piece
    many: bool      # several facts per answer (added to on a re-run) vs one (replaced)
    hint: str = ""

    @property
    def category(self) -> str:
        return f"setup-{self.key}"


QUESTIONS = (
    Question("name", "What should I call you?", "user", "Call me {a}", False),
    Question("work", "What do you do? (work, studies, what fills your week)", "user",
             "What I do: {a}", False),
    Question("people", "Who are the people you mention most, and who are they to you?",
             "entities", "{a}", True, hint="e.g. Petra, my manager; Sam, my partner"),
    Question("help", "What do you most want help with?", "user", "What I want help with: {a}",
             False),
    Question("never", "Is there anything I should never do?", "rules", "{a}", True,
             hint="e.g. never schedule anything before 10am; don't email Petra without showing me"),
)

_RULE_START = re.compile(r"^(?:never|don'?t|do\s+not|no|avoid|stop)\b", re.IGNORECASE)
_FIRST_NAME = re.compile(r"[^\W\d_][\w'-]{0,39}")


def rule_layer() -> str:
    """Where a "never" answer goes. A standing rule must be in front of the model on EVERY turn:
    "never schedule anything before 10am" shares no token with "book a dentist appointment", so
    a by-match layer would silently not apply it. `negative` loads by match today, so rules go
    to `user` (the call `auto_memory.rule_layer` makes for a remembered rule); the registry is
    asked rather than assumed, so they follow `negative` if it ever loads every turn."""
    return "negative" if mr.is_always_loaded("negative") else "user"


def layer_for(q: Question) -> str:
    return rule_layer() if q.layer == "rules" else q.layer


def pieces(q: Question, reply: str) -> list[str]:
    """The facts one answer becomes, in order. A one-fact question keeps the whole answer.
    `people` splits on `;`, or — with no `;` — on commas only when every piece starts with a
    capital (names: "Petra (my manager), Sam (partner)"; "Petra, my manager" stays whole).
    `never` splits on `;` only (commas inside a rule are common), and a piece that does not
    already read as a rule is prefixed "Never "."""
    reply = " ".join(str(reply or "").split())
    if not reply:
        return []
    if not q.many:
        parts = [reply]
    elif ";" in reply:
        parts = reply.split(";")
    elif q.key == "people" and "," in reply:
        commas = [p.strip() for p in reply.split(",")]
        parts = commas if all(p[:1].isupper() for p in commas) else [reply]
    else:
        parts = [reply]
    out = []
    for part in parts:
        p = part.strip().rstrip(".").strip()
        if not p:
            continue
        if q.key == "never":
            p = p[0].upper() + p[1:] if _RULE_START.match(p) else f"Never {p}"
        out.append(q.template.format(a=p))
    return out


def system_first_name() -> "str | None":
    """The first word of the account's full name (macOS keeps it in the passwd GECOS field) —
    one local call, no Apple event, so it cannot raise a permission prompt. None when there is
    nothing name-like there."""
    try:
        import pwd

        full = pwd.getpwuid(os.getuid()).pw_gecos or ""
    except Exception:
        return None
    words = full.split(",")[0].split()
    first = words[0] if words else ""
    return first if _FIRST_NAME.fullmatch(first) else None


def current(q: Question) -> list[dict]:
    """The facts an earlier interview wrote for this question, in id order."""
    return [e for e in mr.entries() if e.get("category") == q.category]

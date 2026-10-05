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

from config import get_config
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


INTRO = ("A few questions so I know who I'm working with. Each answer is saved to your memory "
         "as you give it — Enter skips a question, q stops.")


def _prompt(i: int, q: Question, have: list, default: "str | None") -> str:
    if have:
        keep = " [Enter keeps these; type to add]" if q.many else " [Enter keeps it]"
    else:
        keep = f" [Enter = {default}]" if default else ""
    return f"[{i}/{len(QUESTIONS)}] {q.prompt}{keep} » "


def _refusal(q: Question, texts: list[str]) -> "str | None":
    """Why an answer cannot be stored as given, or None. Asked BEFORE any write, so a refused
    answer is asked again instead of ending the interview on the registry's SecretRefused."""
    for text in texts:
        if len(text) > MAX_FACT_CHARS:
            return (f"that is {len(text)} characters — a memory fact should be one line "
                    f"(under {MAX_FACT_CHARS}). Shorten it, or press Enter to skip.")
        what = mr.secret_problem(f"{q.category} {text}")
        if what:
            return (f"that looks like it holds {what} — Saturn never writes a secret to memory "
                    "(the file is plain text and is read into every prompt). Leave it out, or "
                    "press Enter to skip.")
    return None


def run_interview(*, ask, emit=print, default_name: "str | None" = None) -> dict:
    """Ask the five questions in order. `ask(prompt) -> str` reads one line (ui.ask with
    on_interrupt=INTERRUPT in the app; a scripted callable in tests); `emit(line)` prints. Each
    answer is written through add_memory (by=user) the moment it is given, so leaving early keeps
    everything answered so far. On a re-run each question shows its current answer: Enter keeps
    it, a one-fact question is superseded (replaces=), a many-fact one is added to. Returns
    {"saved": [add_memory's reports], "left": True when the user stopped early}."""
    saved: list[str] = []
    left = False
    emit(f"  {INTRO}")
    for i, q in enumerate(QUESTIONS, 1):
        have = current(q)
        default = default_name if (q.key == "name" and not have) else None
        if have:
            emit("    now: " + "; ".join(e["text"] for e in have))
        elif q.hint:
            emit(f"    {q.hint}")
        while True:
            raw = ask(_prompt(i, q, have, default))
            if raw == INTERRUPT:
                left = True
                break
            reply = " ".join(str(raw or "").split())
            if reply.lower() in _LEAVE:
                left = True
                break
            if reply.lower() in _SKIP:
                reply = ""
            elif not reply and default:
                reply = default
            texts = pieces(q, reply)
            refusal = _refusal(q, texts)
            if refusal is None:
                break
            emit(f"    {refusal}")
            default = None  # after a refusal, Enter means skip
        if left:
            break
        for text in texts:
            replaces = None
            if have and not q.many:
                if text.lower() == have[-1]["text"].lower():
                    emit("    unchanged")
                    continue
                replaces = have[-1]["id"]
            report = mr.add_memory(text, q.category, layer=layer_for(q), replaces=replaces,
                                   by="user", src=f"setup:{q.key}")
            saved.append(report)
            emit(f"    {report}")
    emit(f"  saved {len(saved)} fact(s) · see or change them: /memory · ask again: /memory setup")
    emit(f"  file: {get_config().path('memory')}")
    return {"saved": saved, "left": left}

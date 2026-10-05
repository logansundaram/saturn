# First-Run Interview Implementation Plan

> **Built 2026-10-05** (Tasks 1–6; Task 7, the manual dogfood, is not run) — with the
> amendments I1–I3 from `../specs/2026-10-04-know-the-user-design.md` §2, which this text
> predates: the launch OFFERS the interview, the first run asks `name` / `work` / `never`,
> `/memory setup` asks all five, and a write names its similar facts and refuses a secret.
> That spec's "As built — the interview" is the record of what the code does.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a fresh install, right after `/models` picks the tier, Saturn asks five questions (what to call you, what you do, your people, what you want help with, what it should never do), writes each answer to memory as a `by=user` fact, and never asks again; `/memory setup` re-runs it.

**Architecture:** A new leaf-ish module `core/memory_setup.py` holds the question table, the deterministic answer → fact rules, and `run_interview(ask=, emit=)` with the same injected-`ask` shape as `core/memory_review.run_review`. Every fact goes through the one registry write, `stores/memory_registry.add_memory`, tagged with the category `setup-<question>` so a re-run can find and supersede it. The REPL calls `offer_at_launch(...)` once after the first-run `/models` block; a marker file in the database directory makes it once-only. No model call anywhere.

**Tech Stack:** Python 3.11+, the existing memory registry (one markdown file), `tui.ui.ask` for input, pytest (offline).

**Spec:** `docs/pivot.md` item 5 ("A first-run interview, not a model picker"), `docs/advantages.md` §4 ("Memory — the thing local-first unlocks"), and the Design section below.

---

## Design

### The problem

Saturn starts every install knowing nothing about the person using it. The first launch asks for a model tier (`/models`), then the user's first real request ("reply to Petra about Thursday") meets an agent that does not know who Petra is, what the user does, or what it must never do. Memory exists — six layers in one markdown file, `user` and `commitments` loaded every turn — but it fills only through a gated `remember`, `/memory add`, or a review accept. `pivot.md` #5: "The agent's second turn should already know who it is talking to."

### Approaches considered

1. **A deterministic interview (chosen).** Five fixed prompts in the REPL. Each answer is written nearly verbatim through a fixed per-question template ("Call me …", "What I do: …"); the two many-fact answers (people, rules) are split by a stated rule. *For:* works before any model is pulled (the first launch may still be downloading one), costs zero model calls and zero latency, cannot mis-paraphrase the user's own facts, trivially testable offline, and the user sees exactly what was saved. *Against:* no follow-up questions; a rambling answer lands as one fact.
2. **A model-led conversational interview.** The agent interviews the user and calls `remember`. *For:* natural follow-ups ("Who is Petra to you?"). *Against:* needs a pulled, warm model at the exact moment the first launch is busiest; costs several calls on the weakest tier most users start on; puts paraphrase between the user and their own facts; every `remember` faces the gate, so the "interview" becomes five approval prompts — or the gate must be loosened for it, which is the wrong precedent.
3. **A form in a config file.** Seed `~/.saturn/about-me.md` with headings and tell the user to fill it in. *For:* zero code. *Against:* nobody does it; it duplicates the memory file; it does not reach the memory layers that load every turn without more code anyway.

A later, optional "polish" pass (the model proposes better phrasings of the interview's facts into the existing review queue) is listed under "Not in this plan".

### The chosen design, precisely

**Questions and where answers land.** One table, `QUESTIONS`, in `core/memory_setup.py`:

| key | prompt | layer | fact written | facts per answer |
|---|---|---|---|---|
| `name` | What should I call you? | `user` | `Call me {a}` | one |
| `work` | What do you do? (work, studies, what fills your week) | `user` | `What I do: {a}` | one |
| `people` | Who are the people you mention most, and who are they to you? | `entities` | `{a}` | many |
| `help` | What do you most want help with? | `user` | `What I want help with: {a}` | one |
| `never` | Is there anything I should never do? | `rule_layer()` | `{a}` as a rule | many |

Facts are first-person, the convention the existing `/memory add` examples use ("I prefer answers in metric units").

**Splitting rules (deterministic, stated in code and tests).** Input comes from `ui.ask`, one line per answer.
- A one-fact question keeps the whole answer (whitespace collapsed, a trailing period dropped).
- `people`: split on `;`. With no `;`, split on `,` **only if every comma-separated piece starts with a capital letter** — so `Petra (my manager), Sam (partner), Mom in Lisbon` becomes three facts, while `Petra, my manager` stays one. With neither, the answer is one fact.
- `never`: split on `;` only (commas inside a rule are common: "never email Petra, my manager, without asking"). A piece that already reads as a rule (starts with never / don't / do not / no / avoid / stop) is kept with its first letter capitalized; any other piece is prefixed `Never `.
- A fact longer than `MAX_FACT_CHARS` (300) is refused and the question asked again: an always-loaded fact eats `memory.context_cap` (4,000 characters) every turn, and silently truncating the user's words would be worse.

**Where rules go, so they load every turn.** Only `user`, `commitments` and the last five `memo` entries load every turn (`stores/memory_registry._ALWAYS_LAYERS`); `negative` loads only when it shares a token with the request. "Never schedule anything before 10am" shares nothing with "book a dentist appointment", so a rule stored in `negative` would silently not apply — the failure behind the reported OpenClaw incident in which a "confirm before acting" instruction was lost to context compaction and the agent deleted an inbox (sfstandard.com/2026/02/25/openclaw-goes-rogue/). A safety-relevant instruction must live where it is loaded every turn. The sibling auto-memory plan (`2026-10-01-auto-memory-from-user-statements.md`) lands first and owns the decision of whether `negative` becomes an always-loaded layer. This plan therefore asks the registry, at run time, through a one-line public predicate `memory_registry.is_always_loaded(layer)`: rules go to `negative` when it loads every turn, and to `user` otherwise. A test pins the property that matters either way: a rule saved by the interview appears in the always-loaded half of the grounding for an unrelated request.

**Controls.** Enter skips a question (or takes the offered default / keeps the current answer); `-` or `skip` skips even when a default is offered; `q` / `quit` / Ctrl-C / Ctrl-D leaves the interview, keeping everything already answered (each answer is written as it is given). Ctrl-C reaches the interview as `INTERRUPT` (`"\x03"`) through `ui.ask(..., on_interrupt=INTERRUPT)` — the same convention `core/memory_review` uses, so an interrupt is never confused with an empty "skip" reply. At the end the interview prints the count saved, the memory file path, and the two commands to see or redo it.

**Re-runs (`/memory setup`).** Every fact the interview writes carries the category `setup-<key>`. The registry already parses and keeps `[category]` (`_CATEGORY_RE`, `_clean_category`), and the hyphenated value is ONE match token (`_tokens` keeps inner hyphens) that no request contains, so it never changes what loads. (The metadata token cannot carry a new `src=` key: `_parse_bullet` drops unknown keys on the next write.) On a re-run each question shows its current answer (`now: …`). For a one-fact question, Enter keeps it, the same text is reported `unchanged` (no write — the id stays), and a new answer supersedes it through `add_memory(..., replaces=<id>)`. For a many-fact question, Enter keeps what is there and a typed answer adds to it; removing one is `/memory forget <n>`, which already exists.

**Once only, and never in the wrong place.** A marker `database/.interview_done` (beside the existing first-run sentinel `database/.setup_done`, so deleting the database resets both) records that the offer was made. `launch_action()` decides: marker present → nothing; memory empty → the interview; memory already has facts → one dim line pointing at `/memory setup` (an existing install upgrading is never interviewed mid-launch). The marker is written in a `finally`, so a skip, a Ctrl-C, or an exception inside the interview still counts as offered — a crash must not turn into a question at every launch. Off a TTY (piped stdin) nothing is asked and the marker is NOT written, so the first real terminal session still gets the offer. Headless (`-p` / `-q`) never reaches this code: it lives in `app/repl.run_repl` only.

**The default name.** `pwd.getpwuid(os.getuid()).pw_gecos` is the account's full name on macOS — one local call, no Apple event, no permission prompt. Its first word is offered as `[Enter = Logan]`. Contacts' "me" card is not used: reading it is an Apple event that can raise the Contacts permission prompt during the first launch.

**Sensitive answers.** No sensitivity prompt: five quick questions should stay quick, and these answers are the user's own facts kept on their own machine. `sens=` facts matter only when inference is remote; `/memory add --sens <mark> <fact>` exists for that, and the `/memory --help` text says so.

**Cost on the chat turn.** Zero calls. The interview's facts land in the always-loaded half of the grounding (about 5–8 short lines, ~100 tokens), which changes the stable prefix once — one re-prime, exactly what any memory write to `user` already costs. The interview itself runs while the warm-up thread loads the model weights (`_health_check` → `start_warm_up` runs just before it), so on a fresh install it hides part of the first-launch load instead of adding to it.

### Assumptions made without asking

- The five questions are pivot.md's five, in its order; wording is mine.
- An install that upgrades with an EMPTY memory file is interviewed on its next launch (it has never told Saturn anything); one with facts gets the hint line instead. Both are "offered" exactly once.
- `q` / `quit` typed alone means "stop the interview". A person whose name is literally "Q" can type "Q." or use `/memory add`.
- Answers are stored as first-person sentences through fixed templates; the people answer is stored verbatim per person.
- Rules go to `negative` only if the registry says `negative` is always loaded (the auto-memory plan's call); otherwise to `user`.
- The once-only marker lives in the database directory, like `.setup_done`, not in `config.yaml` — it is install state, not a setting, and `config.persist()` is for user-visible keys.
- The `[setup-…]` category is visible in `/memory` listings (`(2026-10-02) [setup-name] Call me Logan`). That is acceptable — it tells the user where the fact came from.

---

## Global Constraints

- Tests fully offline: no Ollama, no network, no embedder, no Apple events. `pwd` is local and allowed, but tests that reach the default-name path monkeypatch `memory_setup.system_first_name`.
- Any test touching configured paths uses the `isolated_paths` fixture (it redirects `paths.database` and `paths.memory` into `tmp_path`).
- No model call anywhere in the interview, the launch offer, or `/memory setup`.
- `diag.log()`, never `print()`, for diagnostics in `core/`; user-facing lines go through the injected `emit` / `note` callables (the app passes `commands._framework._print` and `ui.note`).
- Every command accepts `--help` (the framework handles it; the `/memory` details text must document `setup`).
- Facts are written only through `stores/memory_registry.add_memory` with `by="user"` — the user typing an answer is the user action. Ids come from the registry's `<!-- next-id -->` high-water mark; this plan never assigns ids itself.
- User-visible changes go under `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog).
- Commit messages: `area: what changed`, lowercase, ending with the `Co-Authored-By` line the session's attribution reminder specifies.

## Review Focus

The five inputs most likely to bite a real user that the happy-path tests would not exercise, each pinned by a test in the owning task:

1. **Ctrl-C at the very first question on a fresh install** → nothing saved, no traceback, and the interview is never offered again. Pinned by `test_ctrl_c_at_the_first_question_saves_nothing` (Task 3) and `test_ctrl_c_at_launch_still_marks_done` (Task 4).
2. **A standing rule must apply to a request that shares no word with it** ("never schedule anything before 10am" vs "book a dentist appointment") → the rule is in the always-loaded half of the grounding. Pinned by `test_a_rule_loads_for_an_unrelated_request` (Task 3).
3. **Re-running `/memory setup` and pressing Enter at every question** → the memory file is byte-identical (no id churn, no duplicate facts). Pinned by `test_rerun_with_enter_everywhere_changes_nothing` (Task 3).
4. **A pasted paragraph as an answer** → refused and asked again, never stored as a giant always-loaded fact. Pinned by `test_a_too_long_answer_is_asked_again` (Task 3).
5. **An existing user upgrading, or a launch with piped stdin** → no questions at launch; the upgrade gets one hint line, and the piped launch neither asks nor marks the offer as made. Pinned by `test_existing_memory_gets_the_hint_not_the_questions` and `test_no_terminal_no_questions_and_no_marker` (Task 4).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `stores/memory_registry.py` | Modify | Add `is_always_loaded(layer) -> bool`, the public face of `_ALWAYS_LAYERS`. |
| `core/memory_setup.py` | Create | The question table, the answer → fact rules, `run_interview`, the once-only launch decision. Imports only `config`, `diag`, `stores.memory_registry`. |
| `app/repl.py` | Modify | Call `memory_setup.offer_at_launch(...)` right after the first-run `/models` block. |
| `commands/knowledge.py` | Modify | `/memory setup` subcommand; usage and `--help` details. |
| `tests/test_memory_setup.py` | Create | Every test in this plan. |
| `CHANGELOG.md`, `CLAUDE.md`, `README.md`, `docs/pivot.md`, `docs/ARCHITECTURE.md` | Modify | Say what is now true. |

## Interfaces

- **Consumes (existing, unchanged):** `stores.memory_registry.add_memory(fact: str, category: str = "general", *, layer: str = "user", replaces=None, by: str = "user", run_id=None, sensitivity=None, due=None) -> str` — **the one registry write this plan and the auto-memory plan both use.** Also `memory_registry.entries(layer=None) -> list[dict]` (each dict has `id`, `layer`, `category`, `text`, `by`, …), `memory_registry.select_for_context(query, *, cap=None, local_inference=None) -> dict`, `memory_registry.memory_context_split(query) -> (always: str, matched: str, ids: list[int])`, `tui.ui.ask(prompt_text, *, on_interrupt="") -> str`, `tui.ui.note(text)`, `commands._framework._print(line)`, `commands._utils._stdin_is_tty() -> bool`.
- **Produces:**
  - `memory_registry.is_always_loaded(layer: str) -> bool` (Task 1). If the auto-memory plan has already landed a public predicate or constant with the same meaning, use that instead of adding this one, and change `memory_setup.rule_layer` to call it.
  - **Reconciled with the auto-memory plan (written after this one, lands first):** that plan adds `src=None` to `add_memory` and a `src=` metadata key the registry keeps, and moves standing rules to the `user` layer without changing which layers load. So: (1) `is_always_loaded("negative")` stays False and `rule_layer()` resolves to `"user"` — the tests here pass unchanged; (2) when `add_memory` accepts `src`, Task 3 also passes `src=f"setup:{q.key}"` on every write (the `[setup-<key>]` category stays the re-run key; `src` is what `/memory why` reads); (3) `core.memory_setup.rule_layer()` and `core.auto_memory.rule_layer(fact, layer)` are different functions in different modules — keep both names module-qualified at call sites.
  - `core.memory_setup`: `INTERRUPT: str = "\x03"`, `MAX_FACT_CHARS: int = 300`, `Question` (frozen dataclass: `key`, `prompt`, `layer`, `template`, `many`, `hint`; property `category -> "setup-<key>"`), `QUESTIONS: tuple[Question, ...]`, `rule_layer() -> str`, `layer_for(q: Question) -> str`, `pieces(q: Question, reply: str) -> list[str]`, `system_first_name() -> str | None`, `current(q: Question) -> list[dict]` (Task 2); `run_interview(*, ask, emit=print, default_name: str | None = None) -> dict` returning `{"saved": list[str], "left": bool}` (Task 3); `marker_path() -> Path`, `launch_action() -> str` (`"interview"` / `"hint"` / `""`), `mark_done() -> None`, `HINT: str`, `offer_at_launch(*, ask, emit, note, interactive: bool) -> str` (Task 4).

---

### Task 1: `is_always_loaded` — the registry says which layers load every turn

**Files:**
- Modify: `stores/memory_registry.py` (add a function after `normalize_layer`)
- Test: `tests/test_memory_setup.py` (create)

**Interfaces:**
- Consumes: `normalize_layer(name) -> str`, `_ALWAYS_LAYERS`.
- Produces: `is_always_loaded(layer: str) -> bool`.

- [ ] **Step 0: Check whether the sibling plan already provides this**

Run: `grep -n "def is_always_loaded\|_ALWAYS_LAYERS = " stores/memory_registry.py`
If `is_always_loaded` exists, skip Steps 1–5 and go to Task 2. If `_ALWAYS_LAYERS` now includes `"negative"`, note it: Task 3's rule facts will land in `negative`, and every test below is written to pass either way.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_setup.py`:

```python
"""The first-run interview (core/memory_setup.py) and the registry predicate it relies on.
Offline: `ask` is scripted, the store is isolated, the default name is stubbed."""

from config import get_config
from stores import memory_registry as mr


def _scripted(answers):
    """An `ask` that replays `answers`, then answers Enter ("") forever."""
    it = iter(answers)

    def ask(_prompt, **_kw):
        return next(it, "")

    return ask


def _quiet(_line):
    pass


def test_is_always_loaded_names_the_every_turn_layers():
    assert mr.is_always_loaded("user") and mr.is_always_loaded("commitments")
    assert mr.is_always_loaded("preferences")  # an alias folds to user first
    assert not mr.is_always_loaded("entities")
    assert not mr.is_always_loaded("memo")  # only its recent digest rides; the layer is by-match
    assert not mr.is_always_loaded("health")  # a hand-made heading is a by-match section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: FAIL — `AttributeError: module 'stores.memory_registry' has no attribute 'is_always_loaded'`

- [ ] **Step 3: Write minimal implementation**

In `stores/memory_registry.py`, directly after `def normalize_layer(...)`:

```python
def is_always_loaded(layer) -> bool:
    """Whether every fact in `layer` rides into EVERY turn's grounding (up to the cap), rather
    than only when it shares a token with the request. A standing rule must live in such a layer
    — "never schedule anything before 10am" shares no word with "book a dentist appointment".
    The memo layer is not: only its recent digest rides along."""
    return normalize_layer(layer) in _ALWAYS_LAYERS
```

`_ALWAYS_LAYERS` is defined above `normalize_layer`, so no reordering is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py tests/test_memory_registry.py tests/test_memory_layers.py -q`
Expected: all pass (`1` new test plus the existing ones).

- [ ] **Step 5: Commit**

```bash
git add stores/memory_registry.py tests/test_memory_setup.py
git commit -m "memory: is_always_loaded — the registry names the every-turn layers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The question table and the answer → fact rules

**Files:**
- Create: `core/memory_setup.py`
- Test: `tests/test_memory_setup.py`

**Interfaces:**
- Consumes: `memory_registry.is_always_loaded`, `memory_registry.entries`.
- Produces: `Question`, `QUESTIONS`, `rule_layer()`, `layer_for(q)`, `pieces(q, reply)`, `system_first_name()`, `current(q)`, `INTERRUPT`, `MAX_FACT_CHARS`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_memory_setup.py`, add this import under `from stores import memory_registry as mr`:

```python
from core import memory_setup as ms
```

Then append:

```python
def _q(key):
    return next(q for q in ms.QUESTIONS if q.key == key)


def test_the_five_questions_in_order():
    assert [q.key for q in ms.QUESTIONS] == ["name", "work", "people", "help", "never"]
    assert _q("people").category == "setup-people"


def test_pieces_one_fact_question_keeps_the_answer_whole():
    assert ms.pieces(_q("name"), "  Logan  ") == ["Call me Logan"]
    assert ms.pieces(_q("work"), "product, at a small startup.") == [
        "What I do: product, at a small startup"]
    assert ms.pieces(_q("help"), "") == []


def test_pieces_people_split_on_semicolons():
    assert ms.pieces(_q("people"), "Petra, my manager; Sam, my partner;") == [
        "Petra, my manager", "Sam, my partner"]


def test_pieces_people_split_on_commas_only_between_names():
    assert ms.pieces(_q("people"), "Petra (my manager), Sam (partner), Mom in Lisbon") == [
        "Petra (my manager)", "Sam (partner)", "Mom in Lisbon"]
    # "my manager" is not a name: the comma is part of one description
    assert ms.pieces(_q("people"), "Petra, my manager") == ["Petra, my manager"]


def test_pieces_rules_read_as_rules():
    assert ms.pieces(_q("never"), "schedule anything before 10am; don't email Petra, ever") == [
        "Never schedule anything before 10am", "Don't email Petra, ever"]
    assert ms.pieces(_q("never"), "never book flights") == ["Never book flights"]


def test_rule_layer_is_loaded_every_turn():
    assert mr.is_always_loaded(ms.rule_layer())
    assert ms.layer_for(_q("never")) == ms.rule_layer()
    assert ms.layer_for(_q("people")) == "entities"


def test_system_first_name_is_the_first_word_of_the_full_name(monkeypatch):
    import pwd

    fake = type("P", (), {"pw_gecos": "Jean-Luc Picard,,,"})()
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: fake)
    assert ms.system_first_name() == "Jean-Luc"


def test_system_first_name_refuses_junk(monkeypatch):
    import pwd

    for gecos in ("", "   ", "1234", "_daemon"):
        fake = type("P", (), {"pw_gecos": gecos})()
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid, f=fake: f)
        assert ms.system_first_name() is None, gecos
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'core.memory_setup'`

- [ ] **Step 3: Write minimal implementation**

Create `core/memory_setup.py`:

```python
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
match token that no request contains, so it never changes what loads.

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
    a by-match layer would silently not apply it. `negative` is a rule's natural home once it
    loads every turn (the auto-memory plan owns that change); until then rules go to `user`,
    which always loads."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add core/memory_setup.py tests/test_memory_setup.py
git commit -m "memory: the first-run interview's questions and answer rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `run_interview` — five prompts, each answer written as it is given

**Files:**
- Modify: `core/memory_setup.py` (append)
- Test: `tests/test_memory_setup.py`

**Interfaces:**
- Consumes: everything from Task 2; `memory_registry.add_memory`, `config.get_config().path("memory")`.
- Produces: `run_interview(*, ask, emit=print, default_name=None) -> {"saved": list[str], "left": bool}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory_setup.py`:

```python
_FIVE = [
    "Logan",
    "product manager at a small startup",
    "Petra, my manager; Sam, my partner",
    "keeping on top of mail",
    "never schedule anything before 10am; don't email Petra without showing me",
]


def test_five_answers_land_in_their_layers(isolated_paths):
    lines = []
    out = ms.run_interview(ask=_scripted(_FIVE), emit=lines.append)
    rules = ms.rule_layer()
    got = {(e["layer"], e["category"], e["text"], e["by"]) for e in mr.entries()}
    assert got == {
        ("user", "setup-name", "Call me Logan", "user"),
        ("user", "setup-work", "What I do: product manager at a small startup", "user"),
        ("entities", "setup-people", "Petra, my manager", "user"),
        ("entities", "setup-people", "Sam, my partner", "user"),
        ("user", "setup-help", "What I want help with: keeping on top of mail", "user"),
        (rules, "setup-never", "Never schedule anything before 10am", "user"),
        (rules, "setup-never", "Don't email Petra without showing me", "user"),
    }
    assert len(out["saved"]) == 7 and out["left"] is False
    assert str(get_config().path("memory")) in "\n".join(lines)  # where the facts live


def test_enter_skips_and_the_default_name_is_offered(isolated_paths):
    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return ""

    ms.run_interview(ask=ask, emit=_quiet, default_name="Logan")
    assert "[Enter = Logan]" in asked[0]
    assert len(asked) == 5
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]  # every other question skipped


def test_dash_skips_even_with_a_default(isolated_paths):
    ms.run_interview(ask=_scripted(["-"]), emit=_quiet, default_name="Logan")
    assert mr.entries() == []


def test_q_stops_and_keeps_what_was_answered(isolated_paths):
    out = ms.run_interview(ask=_scripted(["Logan", "q", "never asked"]), emit=_quiet)
    assert out["left"] is True
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]


def test_ctrl_c_at_the_first_question_saves_nothing(isolated_paths):
    out = ms.run_interview(ask=_scripted([ms.INTERRUPT]), emit=_quiet, default_name="Logan")
    assert out == {"saved": [], "left": True}
    assert mr.entries() == []


def test_a_too_long_answer_is_asked_again(isolated_paths):
    lines = []
    ms.run_interview(ask=_scripted(["x" * 400, "Logan"]), emit=lines.append)
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]
    assert any("characters" in line for line in lines)


def test_a_rule_loads_for_an_unrelated_request(isolated_paths, monkeypatch):
    monkeypatch.setattr(mr, "_inference_is_local", lambda: True)
    ms.run_interview(ask=_scripted(["", "", "", "", "never schedule anything before 10am"]),
                     emit=_quiet)
    always, _matched, _ids = mr.memory_context_split("book a dentist appointment for Tuesday")
    assert "Never schedule anything before 10am" in always  # the every-turn half, not by match


def test_rerun_with_enter_everywhere_changes_nothing(isolated_paths):
    ms.run_interview(ask=_scripted(_FIVE), emit=_quiet)
    path = get_config().path("memory")
    before = path.read_bytes()
    out = ms.run_interview(ask=_scripted([]), emit=_quiet)
    assert out["saved"] == []
    assert path.read_bytes() == before


def test_rerun_supersedes_a_one_fact_answer(isolated_paths):
    ms.run_interview(ask=_scripted(["Logan"]), emit=_quiet)
    lines = []
    out = ms.run_interview(ask=_scripted(["Lo"]), emit=lines.append)
    assert [e["text"] for e in mr.entries()] == ["Call me Lo"]
    assert "replaces #1" in out["saved"][0]
    assert any("now: Call me Logan" in line for line in lines)  # the old answer was shown


def test_rerun_adds_people_and_keeps_an_unchanged_name(isolated_paths):
    ms.run_interview(ask=_scripted(["Logan", "", "Petra, my manager"]), emit=_quiet)
    name_id = ms.current(_q("name"))[0]["id"]
    ms.run_interview(ask=_scripted(["Logan", "", "Jonah, an old friend"]), emit=_quiet)
    assert [e["id"] for e in ms.current(_q("name"))] == [name_id]  # same text: no write, same id
    assert [e["text"] for e in ms.current(_q("people"))] == [
        "Petra, my manager", "Jonah, an old friend"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: the 10 new tests FAIL with `AttributeError: module 'core.memory_setup' has no attribute 'run_interview'`; the 9 earlier tests pass.

- [ ] **Step 3: Write minimal implementation**

Append to `core/memory_setup.py` (and add `from config import get_config` to the imports at the top, below `from dataclasses import dataclass`):

```python
INTRO = ("A few questions so I know who I'm working with. Each answer is saved to your memory "
         "as you give it — Enter skips a question, q stops.")


def _prompt(i: int, q: Question, have: list, default: "str | None") -> str:
    if have:
        keep = " [Enter keeps these; type to add]" if q.many else " [Enter keeps it]"
    else:
        keep = f" [Enter = {default}]" if default else ""
    return f"[{i}/{len(QUESTIONS)}] {q.prompt}{keep} » "


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
            long_one = next((t for t in texts if len(t) > MAX_FACT_CHARS), None)
            if long_one is None:
                break
            emit(f"    that is {len(long_one)} characters — a memory fact should be one line "
                 f"(under {MAX_FACT_CHARS}). Shorten it, or press Enter to skip.")
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
                                   by="user")
            saved.append(report)
            emit(f"    {report}")
    emit(f"  saved {len(saved)} fact(s) · see or change them: /memory · ask again: /memory setup")
    emit(f"  file: {get_config().path('memory')}")
    return {"saved": saved, "left": left}
```

Why `texts` is safe to read after the loop: the loop always assigns it before `break`, except on the two `left = True` paths, which skip the write via `if left: break`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: `19 passed`

- [ ] **Step 5: Run the memory suites to check nothing else moved**

Run: `.venv/bin/python -m pytest tests/test_memory_registry.py tests/test_memory_layers.py tests/test_memory_review.py tests/test_onboarding.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add core/memory_setup.py tests/test_memory_setup.py
git commit -m "memory: run_interview — five answers written as by=user facts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Once per install — the launch offer and the REPL wiring

**Files:**
- Modify: `core/memory_setup.py` (append; add `import diag` and `from pathlib import Path` to the imports)
- Modify: `app/repl.py` (after the `if _first_run:` block, before the "Memory candidates left over" block)
- Test: `tests/test_memory_setup.py`

**Interfaces:**
- Consumes: `run_interview`, `system_first_name` (Tasks 2–3); `memory_registry.entries`; `config.get_config().path("database")`.
- Produces: `marker_path()`, `launch_action()`, `mark_done()`, `HINT`, `offer_at_launch(*, ask, emit, note, interactive) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory_setup.py`:

```python
def _recorder():
    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return ""

    return asked, ask


def test_fresh_install_is_interviewed_once(isolated_paths, monkeypatch):
    monkeypatch.setattr(ms, "system_first_name", lambda: None)
    asked, ask = _recorder()
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=True) == "interview"
    assert len(asked) == 5 and ms.marker_path().exists()
    asked.clear()
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=True) == ""
    assert asked == []


def test_existing_memory_gets_the_hint_not_the_questions(isolated_paths):
    mr.add_memory("I like tea")
    asked, ask = _recorder()
    notes = []
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=notes.append, interactive=True) == "hint"
    assert asked == []
    assert notes and "/memory setup" in notes[0]
    assert ms.marker_path().exists()  # the hint is said once
    notes.clear()
    ms.offer_at_launch(ask=ask, emit=_quiet, note=notes.append, interactive=True)
    assert notes == []


def test_no_terminal_no_questions_and_no_marker(isolated_paths):
    asked, ask = _recorder()
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=False) == ""
    assert asked == []
    assert not ms.marker_path().exists()  # the first real terminal session still gets it


def test_ctrl_c_at_launch_still_marks_done(isolated_paths, monkeypatch):
    monkeypatch.setattr(ms, "system_first_name", lambda: "Logan")
    ms.offer_at_launch(ask=_scripted([ms.INTERRUPT]), emit=_quiet, note=_quiet, interactive=True)
    assert mr.entries() == []
    assert ms.marker_path().exists()


def test_a_crash_inside_the_interview_still_marks_done(isolated_paths, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ms, "run_interview", boom)
    try:
        ms.offer_at_launch(ask=_scripted([]), emit=_quiet, note=_quiet, interactive=True)
    except RuntimeError:
        pass
    assert ms.marker_path().exists()  # a crash must not become a question at every launch


def test_repl_offers_the_interview_after_models():
    import inspect

    from app import repl

    src = inspect.getsource(repl.run_repl)
    assert "memory_setup.offer_at_launch(" in src
    assert src.index('commands.dispatch("/models", cmd_ctx)') < src.index(
        "memory_setup.offer_at_launch(")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: the 6 new tests FAIL (`AttributeError: … 'offer_at_launch'` / `'marker_path'`, and the source assertion for the REPL); the 19 earlier tests pass.

- [ ] **Step 3: Write the launch offer**

Add `import diag` and `from pathlib import Path` to the imports of `core/memory_setup.py`, then append:

```python
HINT = ("/memory setup asks five quick questions — what to call you, your work, your people, "
        "what you want help with, what I should never do. Your memory already has facts, so it "
        "won't run on its own.")


def marker_path() -> Path:
    """Install state, not a setting: beside the first-run sentinel (database/.setup_done), so
    deleting the database resets both."""
    return get_config().path("database") / ".interview_done"


def launch_action() -> str:
    """What this launch does about the interview: "" once it has been offered; "interview" when
    memory is empty (a fresh install, or one that never learned anything); "hint" when memory
    already has facts — an existing install is never interviewed mid-launch."""
    if marker_path().exists():
        return ""
    return "hint" if mr.entries() else "interview"


def mark_done() -> None:
    try:
        path = marker_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError as exc:
        diag.log(f"memory_setup: could not write {marker_path()}: {exc}")


def offer_at_launch(*, ask, emit, note, interactive: bool) -> str:
    """The REPL's once-only offer, after the first-run /models pick. Off a terminal nothing is
    asked AND nothing is marked, so the first real session still gets it. Otherwise the offer is
    marked made whatever happens — a skip, a Ctrl-C, even a crash inside the interview — so it is
    never a question at every launch. Returns the action taken ("" / "interview" / "hint")."""
    action = launch_action()
    if not action or not interactive:
        return ""
    try:
        if action == "interview":
            run_interview(ask=ask, emit=emit, default_name=system_first_name())
        else:
            note(HINT)
    finally:
        mark_done()
    return action
```

- [ ] **Step 4: Wire it into the REPL**

In `app/repl.py`, find this block (the end of the first-run section):

```python
        _health_check()
        try:
            _setup_sentinel.parent.mkdir(parents=True, exist_ok=True)
            _setup_sentinel.touch()
        except Exception as exc:
            diag.log(f"first-run sentinel write failed: {exc}")
```

Immediately after it (outside the `if _first_run:` block, at the same indentation as `if _first_run:`), add:

```python
    # The first-run interview (core/memory_setup): once per install, after the tier is chosen —
    # five questions whose answers become memory facts, or one line about /memory setup for an
    # install that already has facts. On a fresh install it runs while the warm-up thread loads
    # the weights. Non-fatal: a failure here must never stop the REPL.
    try:
        from commands._framework import _print
        from commands._utils import _stdin_is_tty
        from core import memory_setup

        memory_setup.offer_at_launch(
            ask=lambda p: ui.ask(p, on_interrupt=memory_setup.INTERRUPT),
            emit=_print, note=ui.note, interactive=_stdin_is_tty(),
        )
    except Exception as exc:
        ui.warn(f"memory setup skipped: {exc}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: `25 passed`

- [ ] **Step 6: Commit**

```bash
git add core/memory_setup.py app/repl.py tests/test_memory_setup.py
git commit -m "repl: offer the first-run interview once, after /models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `/memory setup` — the re-run

**Files:**
- Modify: `commands/knowledge.py` (`_memory`, its `@command` usage + details, the local `usage` string)
- Test: `tests/test_memory_setup.py`

**Interfaces:**
- Consumes: `memory_setup.run_interview`, `memory_setup.mark_done`, `memory_setup.system_first_name`, `memory_setup.INTERRUPT`; `commands._utils._stdin_is_tty`; `tui.ui.ask`.
- Produces: the `/memory setup` (alias `/memory interview`) subcommand.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory_setup.py`:

```python
_TTY = type("T", (), {"isatty": staticmethod(lambda: True)})()
_NOT_TTY = type("T", (), {"isatty": staticmethod(lambda: False)})()


def test_memory_setup_needs_a_terminal(isolated_paths, ctx, monkeypatch, capsys):
    import commands

    monkeypatch.setattr("sys.stdin", _NOT_TTY)
    commands.dispatch("/memory setup", ctx)
    assert "interactive terminal" in capsys.readouterr().out
    assert mr.entries() == []


def test_memory_setup_runs_the_interview(isolated_paths, ctx, monkeypatch, capsys):
    import commands
    from tui import ui

    monkeypatch.setattr("sys.stdin", _TTY)
    monkeypatch.setattr(ms, "system_first_name", lambda: None)
    monkeypatch.setattr(ui, "ask", _scripted(["Logan"]))
    commands.dispatch("/memory setup", ctx)
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]
    assert ms.marker_path().exists()  # a later launch does not offer it again
    assert "Remembered #1" in capsys.readouterr().out


def test_memory_help_lists_setup(ctx, capsys):
    import commands

    commands.dispatch("/memory --help", ctx)
    assert "/memory setup" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py -q`
Expected: the 3 new tests FAIL (`unknown subcommand 'setup'` in the output; `/memory setup` absent from the help); the 25 earlier tests pass.

- [ ] **Step 3: Implement the subcommand**

In `commands/knowledge.py`, inside `_memory`, add this branch directly before `if sub == "review":`:

```python
    if sub in ("setup", "interview"):
        from commands._utils import _stdin_is_tty
        from core import memory_setup

        if not _stdin_is_tty():
            _print("  /memory setup asks five questions — run it in an interactive terminal.")
            return
        memory_setup.run_interview(
            ask=lambda p: ui.ask(p, on_interrupt=memory_setup.INTERRUPT),
            emit=_print, default_name=memory_setup.system_first_name(),
        )
        memory_setup.mark_done()
        return
```

- [ ] **Step 4: Document it**

In the `@command("memory", ...)` decorator, replace the `usage=` value with:

```python
    usage="/memory [list [layer] | add [--layer L] [--replaces n] [--sens mark] <fact> | "
          "edit <n> <text> | forget <n> | why <n> | setup | review [--no-llm] | stale]",
```

In its `details=` text, insert after the `/memory why <n>` entry (before `/memory review`):

```
  /memory setup              the five-question interview the first launch runs: what to call
                             you, what you do, your people, what you want help with, what I
                             should never do. Each answer is saved as you give it (Enter keeps
                             or skips, q stops); a re-run shows the current answers and a new
                             one replaces it (people and rules are added to). Answers are not
                             marked sensitive — /memory add --sens <mark> <fact> for that.
```

And in the local `usage = (...)` string at the top of `_memory`, replace `| why <n> | review [--no-llm] | stale]` with `| why <n> | setup | review [--no-llm] | stale]`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_setup.py tests/test_help.py tests/test_command_grammar.py -q`
Expected: all pass (`28` in `test_memory_setup.py`).

- [ ] **Step 6: Commit**

```bash
git add commands/knowledge.py tests/test_memory_setup.py
git commit -m "memory: /memory setup re-runs the first-run interview

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Docs say what is now true

**Files:**
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `README.md`, `docs/pivot.md`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: CHANGELOG**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md`, add as the first bullet:

```markdown
- **Saturn asks who you are.** After the first launch picks your model, five quick questions —
  what to call you, what you do, the people you mention most, what you want help with, and
  anything Saturn should never do — are saved straight to your memory, so the next answer
  already knows you. Enter skips a question, `q` stops, and nothing is guessed: each answer is
  stored in your words (the people answer splits into one fact per person, "never…" answers
  into one rule each, and rules load on every turn). It runs once; `/memory setup` asks again,
  showing your current answers. An existing install with memories gets one line about
  `/memory setup` instead of the questions.
```

- [ ] **Step 2: CLAUDE.md**

In the `### Memory` section, replace the sentence

```
Never write a fact without a user action (a gated `remember`, `/memory add`, or a review
accept).
```

with

```
Never write a fact without a user action (a gated `remember`, `/memory add`, a review
accept, or an answer typed into the first-run interview — `core/memory_setup.py`, re-run as
`/memory setup`, once per install via `database/.interview_done`).
```

If the auto-memory plan has already reworded this sentence, keep its wording and append the interview as one more user action in the same parenthesis.

- [ ] **Step 3: README**

In `README.md`, after the paragraph that begins "The installer defaults to the lightweight **`4b`** size class" (it ends "`/models` anytime, or set `SATURN_TIER=9b` …"), add a paragraph:

```markdown
Right after the model pick, Saturn asks five quick questions — what to call you, what you do,
the people you mention most, what you want help with, and anything it should never do — and
saves your answers to its memory (a markdown file you can read and edit; `/memory` shows it).
Enter skips any question; `/memory setup` asks again later.
```

- [ ] **Step 4: pivot.md and ARCHITECTURE.md**

In `docs/pivot.md`, change the heading `### 5. A first-run interview, not a model picker (1 day)` to:

```markdown
### 5. A first-run interview, not a model picker (1 day) — shipped 2026-10-0X (`core/memory_setup.py`; deterministic, no model call; `/memory setup` re-runs it; rules land in a layer loaded every turn)
```

(replace `0X` with the day it lands).

In `docs/ARCHITECTURE.md`, add a row to the `core/` table directly after the `memory_review.py` row:

```markdown
| `memory_setup.py` | The first-run interview: five fixed questions, answers written through `add_memory` as `by=user` facts tagged `[setup-<question>]` (so `/memory setup` can show and supersede them); rules go to a layer loaded every turn. Offered once per install from `app/repl.py`; no model call. |
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (the previous count plus the 28 tests in `tests/test_memory_setup.py`).

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md CLAUDE.md README.md docs/pivot.md docs/ARCHITECTURE.md
git commit -m "docs: the first-run interview and /memory setup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Dogfood it (manual — a real terminal)

- [ ] **Step 1: Fresh-install path without touching real data**

Run Saturn against a throwaway data folder so the real `database/` is untouched. For a wheel install: `SATURN_HOME=$(mktemp -d) saturn`. For the repo checkout (data under `database/`), copy the checkout to a scratch folder and launch from there. Expect: `/models` runs, then the five questions; `[Enter = <first name>]` on the first; answers echoed as `Remembered #N (layer): …`; the outro names the memory file.

- [ ] **Step 2: The second turn knows you**

In that session, run `docs/dogfood.md` §16 "A first session", step 4: "What do you know about me so far?" — it should name what you typed. Then "Book a dentist appointment next week" with a "never before 10am" rule saved — the gate prompt's proposed time should respect it (on the 9b; a 4b miss is a model limit, per the dogfood notes).

- [ ] **Step 3: Relaunch and re-run**

Relaunch: no questions. Run `/memory setup`, press Enter at every question: `saved 0 fact(s)` and `/memory` unchanged. Ctrl-C at the first question of another `/memory setup`: no traceback.

---

## Not in this plan

- **A model "polish" pass**: the model proposes tidier phrasings of the interview's facts into the existing review queue (`core/memory_review`), accepted one by one. Only worth it if dogfooding shows the templated facts read badly to the model.
- **Follow-up questions** ("Who is Petra to you?" when the people answer named someone without a role). Needs a model or a parser; the people answer's hint asks for the role instead.
- **Moving rules between layers**: if the auto-memory plan makes `negative` always-loaded after this plan has run, rules an earlier interview wrote to `user` stay there (they still load every turn). A one-off migration is not worth its code.
- **Marking an existing fact sensitive** (`/memory sens <n> <mark>`). The interview does not ask; today the route is forget + `/memory add --sens`.

## Merge notes for sibling plans

- `2026-10-01-auto-memory-from-user-statements.md` lands first. It owns whether `negative` is always loaded; this plan reads that through `memory_registry.is_always_loaded` (Task 1 adds it only if the sibling did not). Both plans write through `add_memory(..., by="user")`. Both edit CLAUDE.md's "Never write a fact without a user action" sentence — Task 6 Step 2 says how to merge.
- `2026-10-01-launch-brief.md` also adds startup output in `app/repl.py`. The interview runs first (it is part of first-run setup, right after `/models`); a brief, if enabled, prints after it. On a fresh install there is nothing to brief, so they do not compete.

---

## Self-Review (done while writing)

**1. Spec coverage.** pivot #5: five questions after the tier is chosen (Tasks 3–4; wiring after `/models` pinned by `test_repl_offers_the_interview_after_models`); answers to `user` / `entities` / the rule layer (Task 3, `test_five_answers_land_in_their_layers`); skippable (Enter, `-`, `q`, Ctrl-C — Task 3 tests); re-runnable as `/memory setup` (Task 5); the second turn knows the user (Task 7 manual). Brief's extras: no model call (no import of `core.llms` anywhere in the module); light parsing rules stated in code and tests (Task 2); saved facts shown with ids, the `/memory` pointer and the file path (Task 3 outro, asserted in `test_five_answers_land_in_their_layers`); re-run shows the current answer and supersedes with `replaces=` (Task 3); once only, skippable in one keystroke, never headless or off-TTY (Task 4); upgrade gets a hint (Task 4); default name without a permission prompt (Task 2); the shared registry write named (Interfaces); `sens=` decided (Design; Task 5 help text); rules load every turn with the fallback to `user` (Tasks 1–3); docs (Task 6). No gap found.

**2. Placeholder scan.** Every code step shows complete code. The one fill-in is the ship date in pivot.md (`0X`), which can only be known at landing and is called out as such.

**3. Type consistency.** `run_interview(*, ask, emit=print, default_name=None) -> {"saved": list[str], "left": bool}` in Tasks 3, 4, 5. `offer_at_launch(*, ask, emit, note, interactive)` in Task 4's code, tests and the REPL call. `Question.category` (`setup-<key>`) used by `current` (Task 2) and the tests (Tasks 3, 5). `INTERRUPT` passed as `on_interrupt` in both the REPL (Task 4) and `/memory setup` (Task 5). `is_always_loaded(layer) -> bool` (Task 1) is the only registry addition.

**4. Review Focus.** All five lines have tests written out in Tasks 3 and 4, plus `test_a_crash_inside_the_interview_still_marks_done` for the `finally`. Test counts: Task 1 → 1, Task 2 → +8 (9), Task 3 → +10 (19), Task 4 → +6 (25), Task 5 → +3 (28).

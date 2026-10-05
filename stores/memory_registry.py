"""
Persistent memory — the durable, layered store behind `remember` / `recall`, `/memory`, and the
grounding node's memory block.

Session memory rides in the checkpointed message thread; the knowledge base lives in the RAG
store; this module is the layer that survives a restart. It is deliberately ONE flat markdown
file (`paths.memory`), not a database: human-readable, hand-editable, atomic to write, and it
shows up in the workspace like everything else. The design is the manifest of layers, not the
file count:

  layer         holds                                        loaded into context
  ─────────────────────────────────────────────────────────────────────────────────────────
  user          identity, preferences, constraints, style     always, whole (capped)
  commitments   open to-dos / reminders / unfinished work     always (the open set)
  memo          dated episodic notes: decisions, open threads  the recent digest always;
                                                              older ones by match
  agent         operating knowledge about THIS machine/user:  by match against the request
                tool spellings, commands always denied,
                "PDFs need ingest", sites whose extract fails
  entities      the nouns of the user's life + their aliases  by match (alias hit)
  negative      approaches rejected, questions not to re-ask  by match

Every fact carries a metadata token `{#id by=user|inferred run=<run_id> used=<date> n=<count>
sens=<mark> due=<date> src=<how>}` at the end of its bullet: learned-on (the date prefix), the
run it came from (provenance to /trace why), whether the user said it or the review inferred
it, last-used (the expiry signal for never-matched facts), confirmed-count (an inferred fact
graduates to trusted when the user re-states it), a sensitivity mark (a sensitive fact is
withheld from a prompt bound for a remote inference host), a due date for commitments, and how
a by=user fact arrived without a prompt (src=said for auto-learn, src=setup:<question> for the
first-run interview). Missing tokens are
tolerated — a hand-written bullet is a user-layer fact with defaults — and the next write fills
them in. A file written before the layers existed (bullets, no `## layer` headings) reads as the
user layer and is migrated on its next write.

The `remember` / `recall` tools and `/memory` are thin wrappers over `add_memory` /
`search_memory` / `edit_memory` / `remove_memory`; the grounding node calls `memory_context_split`
with the current request so selection stays auditable (`/trace context` shows exactly what
loaded). No FACT is written without a caller that the user drove (a tool call that faced the
gate or whose every word the user typed — core/auto_memory — a slash command, the first-run
interview, or the review screen's accept); the one read-path write is `mark_used`,
which stamps last-used on the facts a turn loaded and changes nothing else.

Hand-editing: bullets and `## layer` headings are the file. Prose outside them (the header, a
note) is not preserved by the next write, and a heading that is not one of the six layers keeps
its bullets as its own by-match section.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path

from config import get_config
from textutil import visible_text

LAYERS = ("user", "commitments", "memo", "agent", "entities", "negative")

# Spellings the tool caller / a hand edit may use for a layer, folded to the canonical name.
_LAYER_ALIASES = {
    "commitment": "commitments", "todo": "commitments", "todos": "commitments",
    "reminder": "commitments", "reminders": "commitments", "pending": "commitments",
    "entity": "entities", "people": "entities", "person": "entities", "project": "entities",
    "projects": "entities", "place": "entities", "places": "entities",
    "note": "memo", "notes": "memo", "memos": "memo", "episodic": "memo", "history": "memo",
    "operating": "agent", "procedure": "agent", "procedures": "agent", "tooling": "agent",
    "dont": "negative", "don't": "negative", "avoid": "negative", "rejected": "negative",
    "preference": "user", "preferences": "user", "identity": "user", "profile": "user",
    "general": "user",
}

# Layers loaded WHOLE every turn (up to the cap), and the size of the memo digest that rides
# along with them. Everything else is selected by match against the request.
_ALWAYS_LAYERS = ("user", "commitments")
_MEMO_DIGEST = 5

_DEFAULT_CONTEXT_CAP = 4000   # chars — the same order as SATURN.md's 6000 (nodes/ground.py)
_DEFAULT_STALE_DAYS = 90      # a by-match fact unused this long is flagged stale in /memory


def _atomic_write(path: Path, text: str) -> None:
    """Write durable memory crash-safely: write a sibling temp file, then os.replace (atomic on
    Windows + POSIX). A crash mid-write leaves the original intact rather than truncating the
    user's irreplaceable facts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_HEADER = (
    "# Saturn — persistent memory\n\n"
    "What Saturn remembers across sessions, by layer. One fact per bullet; safe to hand-edit.\n"
    "user and commitments load into context every turn; the recent memo entries do too; agent,\n"
    "entities and negative load only when they match the request. The trailing {#id …} token is\n"
    "per-fact metadata (id, who said it, source run, last used, confirmations, sensitivity,\n"
    "due date) — keep it or delete it; Saturn refills what it needs. Only bullets and `## layer`\n"
    "headings survive the next write: notes written between them are dropped.\n\n"
)

_LAYER_BLURB = {
    "user": "identity, preferences, constraints, answer style — the user said so",
    "commitments": "open items: to-dos, reminders, work the agent did not finish (with a due date)",
    "memo": "dated notes: what happened, decisions made, open threads — the index over /trace",
    "agent": "operating knowledge about this machine and this user (tools, spellings, what fails)",
    "entities": "people, projects, places, documents, and the user's shorthand for them",
    "negative": "what not to do: approaches rejected, suggestions declined, questions not to re-ask",
}


def _memory_path() -> Path:
    return get_config().path("memory")


def _read_raw() -> str:
    path = _memory_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def normalize_layer(name) -> str:
    """Fold a layer spelling to a canonical layer. An unrecognized but word-like name (a hand
    heading `## health`, a model's `layer="tools"`) becomes its own BY-MATCH section rather than
    being promoted into `user` — the always-loaded layer must never grow from a typo. A name
    that slugs to nothing (`???`) falls back to `user` so the fact is not lost."""
    key = " ".join(str(name or "").lower().split()).strip().rstrip(":")
    if key in LAYERS:
        return key
    if key in _LAYER_ALIASES:
        return _LAYER_ALIASES[key]
    slug = re.sub(r"[^a-z0-9_-]+", "-", key).strip("-")
    return slug or "user"


def is_always_loaded(layer) -> bool:
    """Whether every fact in `layer` rides into EVERY turn's grounding (up to the cap), rather
    than only when it shares a token with the request. A standing rule must live in such a layer
    — "never schedule anything before 10am" shares no word with "book a dentist appointment".
    The memo layer is not: only its recent digest rides along."""
    return normalize_layer(layer) in _ALWAYS_LAYERS


# ── parsing ───────────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\((\d{4}-\d{2}-\d{2})\)\s*")
_CATEGORY_RE = re.compile(r"^\[([^\]]*)\]\s*")
# The trailing metadata token: `{#12 by=user run=7 used=2026-09-02 n=2 sens=health due=...}`.
_META_RE = re.compile(r"\s*\{#(\d+)((?:\s+[a-z]+=[^\s}]+)*)\s*\}\s*$")
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
# The id high-water mark, written under the header: ids are never reused after a deletion, so
# `replaces=#n` and `/trace why` provenance can't silently point at a different fact later.
# A hand edit that drops the line falls back to max(id)+1.
_NEXT_ID_RE = re.compile(r"^<!--\s*next-id:\s*(\d+)\s*-->\s*$")


def _new_entry(text: str, *, layer: str = "user", category: str = "general", by: str = "user",
               run_id=None, sensitivity=None, due=None, src=None, day: str | None = None) -> dict:
    return {
        "id": None,
        "layer": layer,
        "date": day or str(date.today()),
        "category": category or "general",
        "text": text,
        "by": "inferred" if str(by).lower().startswith("infer") else "user",
        "run": int(run_id) if isinstance(run_id, int) and run_id > 0 else None,
        "used": None,
        "n": 1,
        "sens": sensitivity or None,
        "due": due or None,
        "src": src or None,
    }


def _parse_bullet(line: str, layer: str) -> dict | None:
    body = line[2:].strip()
    if not body:
        return None
    entry = _new_entry("", layer=layer, day="")
    m = _META_RE.search(body)
    if m:
        entry["id"] = int(m.group(1))
        for pair in m.group(2).split():
            k, _, v = pair.partition("=")
            if k == "by":
                entry["by"] = "inferred" if v.startswith("infer") else "user"
            elif k == "run" and v.isdigit():
                entry["run"] = int(v)
            elif k == "used":
                entry["used"] = v
            elif k == "n" and v.isdigit():
                entry["n"] = max(1, int(v))
            elif k == "sens":
                entry["sens"] = v
            elif k == "due":
                entry["due"] = v
            elif k == "src":
                entry["src"] = v
        body = body[: m.start()].rstrip()
    d = _DATE_RE.match(body)
    if d:
        entry["date"] = d.group(1)
        body = body[d.end():]
    else:
        # A hand-written bullet: `date` is already today (_new_entry) and the next write
        # stamps it. Until then the model is shown no day for it (_context_line) — "today,
        # every day" would outrank every dated fact and change the cached stable prefix at
        # each midnight.
        entry["undated"] = True
    c = _CATEGORY_RE.match(body)
    if c:
        entry["category"] = " ".join(c.group(1).split()) or "general"
        body = body[c.end():]
    entry["text"] = " ".join(body.split())
    if not entry["date"]:
        entry["date"] = str(date.today())
    return entry if entry["text"] else None


def _parse_state(text: str) -> "tuple[list[dict], int]":
    """`(entries, next_id)`: every fact in the file as an entry dict, in ID order (insertion
    order — the file itself is grouped by layer), plus the id high-water mark. Lines that are
    neither a `## layer` heading, a `- ` bullet, nor the next-id marker are ignored (the header
    prose, blank lines, layer blurbs, hand-written notes). Bullets before the first heading
    belong to the user layer (the pre-layer file format)."""
    entries: list[dict] = []
    layer = "user"
    next_id = 0
    for line in text.splitlines():
        h = _HEADING_RE.match(line)
        if h:
            layer = normalize_layer(h.group(1))
            continue
        n = _NEXT_ID_RE.match(line)
        if n:
            next_id = int(n.group(1))
            continue
        if line.startswith("- "):
            e = _parse_bullet(line, layer)
            if e is not None:
                entries.append(e)
    ids = [e["id"] for e in entries if e.get("id")]
    next_id = max(next_id, (max(ids) + 1) if ids else 1)
    entries.sort(key=lambda e: (e.get("id") is None, e.get("id") or 0))
    # Id-less bullets (a pre-layer file, a hand-written line) get their ids HERE, on read, in
    # file order from the high-water mark — deterministic, so `/memory remove 2` on a legacy
    # file addresses the same fact the listing showed. The file itself is untouched until the
    # next write persists them (a read never writes).
    next_id = _assign_ids(entries, next_id)
    return entries, next_id


def _parse(text: str) -> list[dict]:
    return _parse_state(text)[0]


def _read_state() -> "tuple[list[dict], int]":
    return _parse_state(_read_raw())


def _entries() -> list[dict]:
    return _read_state()[0]


def _display(e: dict) -> str:
    """The `(date) [category] text` line — what /memory lists and what the pre-layer tests pin."""
    tag = f"[{e['category']}] " if e.get("category") and e["category"] != "general" else ""
    return f"({e['date']}) {tag}{e['text']}"


def display(e: dict, *, due: bool = True) -> str:
    """`_display` plus the due date of a commitment — the one rendering every listing (/memory,
    the `recall` tool, the forget confirmation) shares."""
    line = _display(e)
    return f"{line}  (due {e['due']})" if due and e.get("due") else line


def _meta_token(e: dict) -> str:
    parts = [f"#{e['id']}", f"by={e['by']}"]
    if e.get("run"):
        parts.append(f"run={e['run']}")
    if e.get("used"):
        parts.append(f"used={e['used']}")
    if int(e.get("n") or 1) > 1:
        parts.append(f"n={int(e['n'])}")
    if e.get("sens"):
        parts.append(f"sens={_token_safe(e['sens'])}")
    if e.get("due"):
        parts.append(f"due={_token_safe(e['due'])}")
    if e.get("src"):
        parts.append(f"src={_token_safe(e['src'])}")
    return "{" + " ".join(parts) + "}"


def _token_safe(v) -> str:
    """A metadata value must not contain whitespace or `}` — either breaks the token parse."""
    return re.sub(r"[\s}]+", "-", str(v)).strip("-") or "x"


def _render(entries: list[dict], next_id: int) -> str:
    """Serialize back to the file: header, the next-id marker, then one `## layer` section per
    layer (the six standard ones always, so a hand-editor sees the shape; any other heading a
    hand edit or a `layer=` argument introduced keeps its bullets under its own by-match
    section — see normalize_layer). Non-bullet, non-heading lines are not preserved."""
    by_layer: dict[str, list[dict]] = {layer: [] for layer in LAYERS}
    for e in entries:
        by_layer.setdefault(e["layer"], []).append(e)
    out = [_HEADER, f"<!-- next-id: {next_id} -->\n\n"]
    for layer, items in by_layer.items():
        out.append(f"## {layer}\n")
        blurb = _LAYER_BLURB.get(layer)
        if blurb:
            out.append(f"<!-- {blurb} -->\n")
        for e in items:
            out.append(f"- {_display(e)} {_meta_token(e)}\n")
        out.append("\n")
    return "".join(out).rstrip("\n") + "\n"


def _assign_ids(entries: list[dict], next_id: int) -> int:
    """Give every id-less entry the next free id from the high-water mark (ids are never
    reused, so `/trace why` provenance and `replaces=` references stay meaningful). Returns the
    new high-water mark."""
    used = {e["id"] for e in entries if e.get("id")}
    for e in entries:
        if not e.get("id"):
            while next_id in used:
                next_id += 1
            e["id"] = next_id
            used.add(next_id)
            next_id += 1
    return next_id


def _write(entries: list[dict], next_id: int) -> None:
    next_id = _assign_ids(entries, next_id)
    _atomic_write(_memory_path(), _render(entries, next_id))


# ── writes ────────────────────────────────────────────────────────────────────────────────────

def _clean_text(fact) -> str:
    # Normalize BEFORE the empty check and the dedup comparison — this is the ONE write boundary
    # (the `remember` tool, /memory add, and the review screen all land here). The file's contract
    # is one fact per bullet: a model-supplied multi-line fact written verbatim would leave
    # continuation lines the parser never sees. Collapsing ALL whitespace runs keeps dedup
    # comparing the same form a reflowed duplicate arrives in. A stray `{#…}` token inside the
    # text would be parsed back as metadata, so its braces are softened. The fact is stored as
    # the approval gate would show it (`textutil.visible_text`): /memory add and the review
    # screen print it on a console that shows controls but not a tag, zero-width or bidi
    # character, and text nobody saw must not ride every later turn's context.
    text = " ".join(visible_text(str(fact or "")).split())
    return text.replace("{#", "(#").replace("}", ")") if "{#" in text else text


def _clean_category(category) -> str:
    # The category rides inside the bullet's "[category] " prefix: a newline breaks the bullet
    # line and a "]" would end the tag early. Terminal controls and unseen characters become
    # symbols, as in the fact itself (`_clean_text`). A category that sanitizes to nothing
    # falls back to the untagged default.
    return (" ".join(visible_text(str(category or "")).split()).replace("]", "").strip()
            or "general")


# ── secrets are never saved ───────────────────────────────────────────────────────────────────
# The memory file is plain text, is read into every prompt, and leaves the machine under a
# remote OLLAMA_HOST. So no Saturn path writes a credential into it — not the model's
# `remember`, not a review accept, not /memory add or /memory edit (add_memory and edit_memory
# are the two writers of fact text, and both ask here; the category is screened with the fact).
# Deterministic, recognisable shapes only — a secret written some other way gets through, so
# this is a net, not a promise. Health and money are the user's own facts and are handled by
# `sens=`, not refused. Editing the file by hand is the user's business.
_CARD_RE = re.compile(r"(?<![\w-])\d(?:[ .-]?\d){12,18}(?![\w-])")
_SSN_RE = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"
                     r"|\b(?:ssn|social security)\b\D{0,24}\d{3}[ -]?\d{2}[ -]?\d{4}(?!\d)",
                     re.IGNORECASE)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_API_KEY_RE = re.compile(r"(?<![A-Za-z0-9])(?:sk[-_][A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
                         r"|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}"
                         r"|xox[baprs]-[A-Za-z0-9-]{10,})")
# "password is hunter2", "password: hunter2", "wifi password for home is hunter2", "the
# password I use is hunter2" — the value follows is/:/= within a few words, whatever they are;
# or "wifi password hunter2" — it follows directly and does not look like a word (a digit or
# a symbol in it). Only a noun right after the word makes it talk ABOUT passwords: "password
# manager is Bitwarden", "the password policy is strict", "passcode rotation is quarterly".
# The list is what is let through, so a phrasing it does not know is refused, not saved.
_ABOUT = (r"(?!\s+(?:manager|managers|policy|policies|rotation|reset|resets|length|rule|rules"
          r"|requirement|requirements|hint|hints|field|prompt|change|changes|expiry|expiration"
          r"|strength|generator|vault|app|protection|sharing|reuse|complexity|storage"
          r"|management|history|recovery|reminder|screen|page|form|box|dialog|check|checker"
          r"|link|email|habit|habits|hygiene|security)\b)")
_WHICH_ONE = _ABOUT + r"(?:\s+[\w'’.-]+){0,4}?"
_IS = r"\s*(?:\b(?:is|was)\b\s*[:=]?|[:=])\s*"
_PASSWORD_SAID_RE = re.compile(
    r"\b(?:password|passcode|passphrase)\b" + _WHICH_ONE + _IS + r"[\"']?([^\s\"']+)",
    re.IGNORECASE)
_PASSWORD_BARE_RE = re.compile(
    r"\b(?:password|passcode|passphrase)\s+[\"']?((?=\S*[\d!@#$%^&*+=_])[^\s\"']{4,})",
    re.IGNORECASE)
# "my pin (number) for the garage is 4821", "the pin I use is 4821"; in capitals also bare and
# as "PIN code" — in lower case a "pin code" is a postal code ("pin code is 94110") unless it
# is the pin code OF something that has one ("my phone's pin code is 482193").
_HAS_A_PIN = (r"(?:phone|iphone|ipad|tablet|laptop|computer|card|debit|credit|bank|atm|sim"
              r"|door|garage|gate|lock|alarm|safe|locker|voicemail|account)")
_PIN_RE = re.compile(r"\bpin\b(?:\s+number)?(?!\s+code\b)" + _WHICH_ONE + _IS + r"\d{4,8}\b"
                     r"|\b" + _HAS_A_PIN + r"(?:['’]s)?\s+pin\s+code\b" + _WHICH_ONE + _IS
                     + r"\d{4,8}\b"
                     r"|(?-i:\bPIN)(?:\s+code)?(?:" + _IS + r"|\s+)\d{4,8}\b", re.IGNORECASE)
# What may follow "the password is …" without being the password itself.
_NOT_A_PASSWORD = frozenset("""
in on at the a an my our stored saved kept written not same different too weak strong long
short there here what where with for under inside changed expired wrong correct unknown
required needed optional mandatory set reset protected empty blank missing safe secure
""".split())
# ...unless it opens a passphrase: "correct horse battery staple" is four words or more and
# none of the rest is one of these or of the words above, where a remark about the password
# ("required for the wifi", "stored in my vault") is made of them.
_PASSPHRASE_MIN_WORDS = 4
_REMARK_WORDS = _NOT_A_PASSWORD | frozenset("""
is are was be been it this that these those and or but to of by as than from now again
always never only also every each last next day week month year today yesterday i we you
enough already honestly really very quite pretty so just still far much more less anyway
though actually probably maybe
""".split())
_VALUE_END_RE = re.compile(r"[.;,!?\n\"]")


class SecretRefused(ValueError):
    """A fact was refused because it holds a credential. `str(exc)` is the sentence to show."""


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def secret_problem(text) -> "str | None":
    """What kind of credential `text` holds ("a card number", "a password", …), or None."""
    text = str(text or "")
    if _PRIVATE_KEY_RE.search(text):
        return "a private key"
    if _API_KEY_RE.search(text):
        return "an API key"
    for m in _CARD_RE.finditer(text):
        if _luhn(re.sub(r"\D", "", m.group(0))):
            return "a card number"
    if _SSN_RE.search(text):
        return "a Social Security number"
    if _PIN_RE.search(text):
        return "a PIN"
    if _PASSWORD_BARE_RE.search(text):
        return "a password"
    for m in _PASSWORD_SAID_RE.finditer(text):
        if m.group(1).lower().strip(".,;:") not in _NOT_A_PASSWORD:
            return "a password"
        if _VALUE_END_RE.search(m.group(1)[-1:]):
            continue        # "The password is wrong. Call Petra…" — its sentence ends there
        rest = _VALUE_END_RE.split(text[m.end(1):], maxsplit=1)[0].lower().split()
        if (len(rest) + 1 >= _PASSPHRASE_MIN_WORDS
                and not any(w.strip("':") in _REMARK_WORDS for w in rest)):
            return "a password"
    return None


def _refuse_secret(text: str) -> None:
    what = secret_problem(text)
    if what:
        raise SecretRefused(f"not saved — this looks like it holds {what}. Saturn never writes "
                            "a secret to memory: the file is plain text and is read into every "
                            "prompt.")


def add_memory(fact: str, category: str = "general", *, layer: str = "user", replaces=None,
               by: str = "user", run_id=None, sensitivity=None, due=None, src=None) -> str:
    """Append a durable fact to `layer`. A fact already stored (same text, any layer) is not
    duplicated — its confirmed-count rises, and an inferred fact the user now states outright
    graduates to trusted. `replaces=<id>` supersedes an earlier fact instead of sitting beside it
    ("I moved to Berlin" replaces "I live in Paris"). `src` says how a by=user fact arrived
    without a prompt — `said` (auto-learn: the user typed it in conversation, core/auto_memory)
    or `setup:<question>` (the first-run interview); None for the gate, /memory add and the
    review. Returns a one-line report."""
    fact = _clean_text(fact)
    if not fact:
        return "Nothing to remember — the fact was empty."
    category = _clean_category(category)
    _refuse_secret(f"{category} {fact}")     # the category is written into the file too
    layer = normalize_layer(layer)

    entries, next_id = _read_state()
    # Retire the superseded fact FIRST, so a correction whose text already exists elsewhere
    # ("I live in Berlin" stored at a review, now stated with replaces=<the Paris fact>) still
    # removes the old one instead of returning early from the dedup below.
    replaced = None
    if replaces is not None:
        try:
            rid = int(str(replaces).strip().lstrip("#"))
        except (TypeError, ValueError):
            rid = 0
        if rid > 0:
            for e in entries:
                if e.get("id") == rid:
                    replaced = e
                    break
            if replaced is not None:
                entries.remove(replaced)
    replaced_note = (f" — replaces #{replaced['id']} {replaced['text']!r}" if replaced is not None
                     else f" (no fact #{replaces} to replace — stored alongside)"
                     if replaces is not None else "")

    # Dedup on the bare fact text, case-insensitive — and by equality, not substring: a short new
    # fact must not be swallowed just because it appears inside a longer stored line.
    for e in entries:
        if fact.lower() == e["text"].lower():
            e["n"] = int(e.get("n") or 1) + 1
            graduated = e["by"] == "inferred" and by == "user"
            if graduated:
                e["by"] = "user"
            if sensitivity and not e.get("sens"):
                e["sens"] = sensitivity
            if src and not e.get("src"):
                e["src"] = src
            _write(entries, next_id)
            note = " — now confirmed by you" if graduated else f" (confirmed ×{e['n']})"
            return f"Already remembered as #{e['id']}{note}: {fact!r}{replaced_note}"

    new = _new_entry(fact, layer=layer, category=category, by=by, run_id=run_id,
                     sensitivity=sensitivity, due=due, src=src)
    if replaced is not None and replaced["layer"] != "user" and layer == "user":
        new["layer"] = replaced["layer"]  # a replacement stays in the layer it corrects
    entries.append(new)
    _write(entries, next_id)
    return f"Remembered #{new['id']} ({new['layer']}): {fact!r}{replaced_note}"


def edit_memory(fact_id: int, new_text: str) -> str | None:
    """Rewrite the text of fact `fact_id` in place (id, layer, provenance kept; confirmed-count
    reset, since the words changed). Returns the old text, or None if no such id / empty text."""
    text = _clean_text(new_text)
    if not text:
        return None
    _refuse_secret(text)
    entries, next_id = _read_state()
    for e in entries:
        if e.get("id") == fact_id:
            old = e["text"]
            e["text"] = text
            e["n"] = 1
            _write(entries, next_id)
            return old
    return None


def remove_memory(fact_id: int) -> str | None:
    """Delete the fact with id `fact_id` (the number /memory shows). Returns the removed
    fact's display line, or None if no such id. Rewrites the file atomically with the standard
    header — hand-written non-bullet lines are not preserved (the file's contract is one fact
    per bullet; see _HEADER)."""
    entries, next_id = _read_state()
    for e in entries:
        if e.get("id") == fact_id:
            entries.remove(e)
            _write(entries, next_id)
            return display(e)
    return None


def mark_used(ids, *, day: str | None = None) -> int:
    """Stamp last-used on the given facts (the by-match facts the grounding node actually loaded
    this turn). One atomic write, and only when a stamp changes — an unchanged day is a no-op, so
    a busy session doesn't rewrite the file every turn. Returns the number of facts stamped."""
    wanted = {int(str(i).lstrip("#")) for i in ids or () if str(i).lstrip("#").isdigit()}
    if not wanted:
        return 0
    today = day or str(date.today())
    entries, next_id = _read_state()
    changed = 0
    for e in entries:
        if e.get("id") in wanted and e.get("used") != today:
            e["used"] = today
            changed += 1
    if changed:
        _write(entries, next_id)
    return changed


# ── reads ─────────────────────────────────────────────────────────────────────────────────────

def entries(layer: str | None = None) -> list[dict]:
    """Every stored fact as a dict (id, layer, date, category, text, by, run, used, n, sens,
    due), in id order; `layer` filters. A hand-written or pre-layer bullet carries a provisional
    id (assigned on read, persisted by the next write)."""
    items = _entries()
    if layer:
        layer = normalize_layer(layer)
        items = [e for e in items if e["layer"] == layer]
    return items


def entry(fact_id: int) -> dict | None:
    for e in _entries():
        if e.get("id") == fact_id:
            return e
    return None


def search_memory(query: str = "", *, local_inference: bool | None = None) -> list[str]:
    """Stored facts matching `query` (case-insensitive substring over the text, category and
    layer), rendered as `#id [layer] (date) text`. An empty query returns everything — the store
    is small by design, so a full dump is reasonable. The same sensitivity rule as the grounding
    selection applies: when inference is not local, `sens=` facts are withheld and a trailing
    line says how many — `recall` is a model-facing read and must not be the leak around it."""
    query = " ".join((query or "").split()).lower()
    local = _inference_is_local() if local_inference is None else local_inference
    out = []
    withheld = 0
    for e in _entries():
        hay = f"{e['layer']} {e['category']} {e['text']}".lower()
        if not query or query in hay:
            if not local and e.get("sens"):
                withheld += 1
                continue
            out.append(f"#{e['id']} [{e['layer']}] {display(e)}")
    if withheld:
        out.append(f"({withheld} sensitive fact(s) withheld — inference is not local)")
    return out


def context_cap() -> int:
    try:
        return max(200, int(get_config().get("memory.context_cap", _DEFAULT_CONTEXT_CAP)))
    except Exception:
        return _DEFAULT_CONTEXT_CAP


def stale_days() -> int:
    try:
        return max(1, int(get_config().get("memory.stale_days", _DEFAULT_STALE_DAYS)))
    except Exception:
        return _DEFAULT_STALE_DAYS


def is_stale(e: dict, *, today: date | None = None) -> bool:
    """A by-match fact that has not matched a request in `memory.stale_days` (measured from its
    last use, or from the day it was learned if it never matched). Always-loaded layers are never
    stale — they load regardless."""
    if e["layer"] in _ALWAYS_LAYERS:
        return False
    today = today or date.today()
    anchor = e.get("used") or e.get("date")
    try:
        then = datetime.strptime(str(anchor), "%Y-%m-%d").date()
    except ValueError:
        return False
    return (today - then).days > stale_days()


_STOPWORDS = frozenset("""
the and for with that this from what when where which who how are was were will would
can could should into onto about after before over under then than them they there here
have has had does did done not but all any some our your their its his her you
please make give find show tell let get use using run take need want like just also
""".split())


def _tokens(text: str) -> set[str]:
    """Match tokens: lowercase alphanumeric words of 3+ chars (an inner apostrophe or hyphen is
    part of the word — "petra's", "read-only"; a wrapping quote is not), minus stopwords."""
    return {t for t in re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", str(text or "").lower())
            if len(t) >= 3 and t not in _STOPWORDS}


def match_score(e: dict, query_tokens: set[str]) -> int:
    """How many request tokens the fact shares (text + category). Zero = not loaded."""
    if not query_tokens:
        return 0
    return len(query_tokens & _tokens(f"{e['text']} {e['category']}"))


def _inference_is_local() -> bool:
    """trust.egress.ollama_is_local, failing toward NOT local like egress itself does — an
    unanswerable question must never earn a sensitive fact a ride off the machine."""
    try:
        from trust.egress import ollama_is_local
        return bool(ollama_is_local())
    except Exception:
        return False


def select_for_context(query: str = "", *, cap: int | None = None,
                       local_inference: bool | None = None) -> dict:
    """Pick what rides into this turn's grounding context, under one cap:

      always   the user layer, the open commitments, and the last `_MEMO_DIGEST` memo entries
      matched  agent / entities / negative / older memo facts sharing tokens with the request,
               best match first

    Sensitive facts (`sens=`) are withheld when inference is not local (a remote OLLAMA_HOST) —
    a health or money fact must never ride a prompt bound off the machine. Returns
    `{"always": [...], "matched": [...], "digest": [...], "omitted": int,
    "sensitive_withheld": int}` (`digest` is the memo subset of `always`); the grounding node
    renders it and stamps last-used on the matched + digest ids (selected_ids)."""
    cap = context_cap() if cap is None else cap
    local = _inference_is_local() if local_inference is None else local_inference
    items = _entries()
    withheld = 0
    if not local:
        kept = [e for e in items if not e.get("sens")]
        withheld = len(items) - len(kept)
        items = kept

    always: list[dict] = [e for e in items if e["layer"] in _ALWAYS_LAYERS]
    memos = [e for e in items if e["layer"] == "memo"]
    digest = memos[-_MEMO_DIGEST:] if memos else []
    always += digest
    digest_ids = {id(e) for e in digest}

    qtok = _tokens(query)
    candidates = []
    for e in items:
        if e["layer"] in _ALWAYS_LAYERS or id(e) in digest_ids:
            continue
        score = match_score(e, qtok)
        if score > 0:
            candidates.append((score, e.get("id") or 0, e))
    candidates.sort(key=lambda t: (-t[0], -t[1]))

    budget = cap
    chosen_always: list[dict] = []
    omitted = 0
    for e in always:
        cost = len(_context_line(e)) + 1
        if cost <= budget:
            chosen_always.append(e)
            budget -= cost
        else:
            omitted += 1
    chosen_matched: list[dict] = []
    for _score, _id, e in candidates:
        cost = len(_context_line(e)) + 1
        if cost <= budget:
            chosen_matched.append(e)
            budget -= cost
        else:
            omitted += 1
    return {"always": chosen_always, "matched": chosen_matched,
            "digest": [e for e in chosen_always if id(e) in digest_ids],
            "omitted": omitted, "sensitive_withheld": withheld}


def _context_line(e: dict) -> str:
    """One fact as the model sees it: the id (so `remember(..., replaces=<id>)` can supersede
    it), the day it was written (the one deterministic signal for which of two facts is
    current — a small model otherwise follows whichever stored value it reads; spec
    2026-10-04-know-the-user A1), the layer tag for the by-match facts (so a `negative` fact
    reads as a prohibition), `[inferred]` on a fact the user accepted but did not state, and the
    due date of a commitment."""
    bits = [f"- #{e['id'] or '?'}"]
    if e.get("date") and not e.get("undated"):
        bits.append(f"({e['date']})")
    if e["layer"] not in ("user",):
        bits.append(f"[{e['layer']}]")
    if e.get("by") == "inferred":
        bits.append("[inferred]")
    bits.append(e["text"])
    if e.get("due"):
        bits.append(f"(due {e['due']})")
    return " ".join(bits)


def memory_context_split(query: str = "") -> "tuple[str, str, list[int]]":
    """One selection for one turn, in two blocks: `(always, matched, matched_ids)`. `always` is
    the query-INDEPENDENT half (the user layer, open commitments, the memo digest — byte-stable
    across turns while the store is unchanged, so it rides the grounding's stable half and the
    daemon's prompt cache); `matched` is the by-match half plus the trailer naming what did NOT
    load (so a fact that silently didn't ride is never a mystery), which changes with the
    request. Either is "" when empty. `matched_ids` are the facts the caller stamps as used
    (see selected_ids)."""
    sel = select_for_context(query)
    always = [_context_line(e) for e in sel["always"]]
    matched = [_context_line(e) for e in sel["matched"]]
    if not always and not matched:
        return "", "", []
    trailer = []
    if sel["omitted"]:
        trailer.append(f"{sel['omitted']} fact(s) not loaded")
    if sel["sensitive_withheld"]:
        trailer.append(f"{sel['sensitive_withheld']} sensitive fact(s) withheld (remote inference)")
    if trailer:
        matched.append("(" + "; ".join(trailer) + " — `recall` searches everything else stored)")
    return "\n".join(always), "\n".join(matched), selected_ids(sel)


def selected_ids(sel: dict) -> list[int]:
    """The ids whose last-used the grounding node stamps: the by-match facts AND the memo digest.
    The digest rides every turn but memo is a by-match layer for staleness, so without the stamp
    a note that loads daily would be flagged stale. user/commitments are never stale and are not
    stamped."""
    seen = set()
    out = []
    for e in list(sel.get("matched") or []) + list(sel.get("digest") or []):
        if e.get("id") and e["id"] not in seen:
            seen.add(e["id"])
            out.append(e["id"])
    return out

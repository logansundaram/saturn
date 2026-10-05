"""
Grounding node: loads what is NOT already available to the model —
  - the standing instructions (~/.saturn/SATURN.md, then the workspace's SATURN.md),
  - the knowledge-base manifest (so the agent knows what documents it can search), and
  - persistent memory (stores/memory_registry): the user layer + open commitments + the recent
    memo digest always, agent/entities/negative facts by match against the request, all under
    one cap.

It deliberately does NOT include the tool inventory (the native tool bind carries the catalog;
duplicating it hurts small models) or the chat history (`messages` goes to the model directly).
Built once per turn; tool results flow through `messages`, never this grounding string.

The block is built in TWO halves: `context_stable` — instructions, the manifest, the
query-independent memory layers — is byte-identical across turns while nothing on disk changed,
and `context_dynamic` — the date, memory's by-match selection, attachments — changes every turn.
The prompt sends the stable half as its own message right after the system prompt and the
dynamic half with the request, and the idle prime (core/prime.py) re-sends exactly
`stable_grounding()` between turns so the daemon holds a checkpoint at that message boundary: the
next turn's agent call then prefills only what is new (docs/OPTIMIZATIONS.md, "the prefix
cache"). `context` stays the joined block for every reader that wants the whole thing.
"""

import time
from datetime import datetime
from pathlib import Path

import diag

from core.state import AgentState
from stores.memory_registry import memory_context_split, mark_used
from stores.document_registry import read_documents_manifest

# Standing instructions (the CLAUDE.md equivalent), loaded into context EVERY turn:
#   ~/.saturn/SATURN.md       global — tone, standing rules; hand-written, follows the user
#                             everywhere ($SATURN_HOME overrides the directory);
#   <workspace>/SATURN.md     per-workspace — conventions, goals; drafted by /init. Where the two
#                             conflict the workspace file wins, and the prompt says so.
# Each is capped so a runaway file can't eat the context window.
INSTRUCTIONS_FILE = "SATURN.md"
_INSTRUCTIONS_CAP = 6000


def global_instructions_path() -> Path:
    """Where the global standing instructions live: `$SATURN_HOME/SATURN.md`, else
    `~/.saturn/SATURN.md` (config.saturn_home)."""
    from config import saturn_home

    return saturn_home() / INSTRUCTIONS_FILE


def _read_capped(path: Path) -> str:
    if not path.is_file():
        return ""
    # errors="replace": a hand-edited file with a stray non-UTF-8 byte must not fail every turn
    # at the first node.
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > _INSTRUCTIONS_CAP:
        text = text[:_INSTRUCTIONS_CAP] + f"\n… ({path.name} truncated — keep it concise)"
    return text


def _read_instructions() -> str:
    """The workspace instructions; "" when there are none."""
    from core import workspace as _ws

    return _read_capped(_ws.root() / INSTRUCTIONS_FILE)


def _read_global_instructions() -> str:
    try:
        return _read_capped(global_instructions_path())
    except Exception as exc:  # an unreadable home must not fail the first node of every turn
        diag.log(f"grounding_node : global SATURN.md unreadable: {exc}")
        return ""


def _working_folder_section() -> str:
    """Where Saturn is working (core/workspace): the launch folder and the session's /add-dir
    folders. In the STABLE half — the root is fixed for the session, so the prefix cache holds;
    /add-dir and /rm-dir miss it once, like editing SATURN.md."""
    from core import workspace as _ws

    lines = [f"You are working in {_ws.display(_ws.root())}. Relative paths resolve here."]
    extra = _ws.extra()
    if extra:
        lines.append("Also reachable this session (added with /add-dir): "
                     + ", ".join(_ws.display(p) for p in extra))
    lines.append("Any other folder needs the user to run /add-dir <folder> first.")
    return "### Working folder\n" + "\n".join(lines)


def now_section(now: "datetime | None" = None) -> str:
    """Today's date, weekday and the time. In the DYNAMIC half — it
    changes every turn, and the dynamic half rides only the current request, never the history,
    so the cached prefix is untouched. With it, "Thursday" and "what's the date" resolve on the
    first pass instead of a current_time round trip."""
    now = now or datetime.now().astimezone()
    offset = now.strftime("%z")
    return ("### Now\n"
            f"{now.strftime('%A')} {now.strftime('%Y-%m-%d')} ({now.day} {now.strftime('%B %Y')}), "
            f"{now.strftime('%H:%M')} local time (UTC{offset[:3]}:{offset[3:]})")


def stable_grounding(memory_always: "str | None" = None) -> str:
    """The query-independent half of the grounding block — what the idle prime re-sends between
    turns. Byte-identical to the `context_stable` the next turn's grounding_node builds unless
    the knowledge base, the instructions files or the always-loaded memory layers changed
    in between (in which case the prime simply misses and the turn prefills it).
    `memory_always` is the always-loaded memory block when the caller already selected it
    (grounding_node, which needs the by-match half of the same selection); None reads it here."""
    sections = ["## Grounding context", _working_folder_section()]

    global_instructions = _read_global_instructions()
    if global_instructions:
        sections.append(
            "### Standing instructions (~/.saturn/SATURN.md — the user's standing guidance "
            "everywhere; follow it)\n" + global_instructions
        )
    instructions = _read_instructions()
    if instructions:
        sections.append(
            f"### Workspace instructions ({INSTRUCTIONS_FILE} — the user's standing guidance for this "
            "workspace; follow it, and where it conflicts with the standing instructions above "
            "it wins)\n" + instructions
        )

    docs_manifest = read_documents_manifest().strip()
    sections.append(
        "### Knowledge base (searchable via `search_knowledge_base`)\n"
        + (docs_manifest or "No ingested documents yet.")
    )

    # The always-loaded memory layers (user, commitments, the memo digest) are query-independent
    # — the stable half. The by-match facts land in the dynamic half below.
    always = memory_context_split("")[0] if memory_always is None else memory_always
    if always:
        sections.append(
            "### Persistent memory (what the user told me and the day they said it; a later "
            "day outranks an earlier one, and what the user says in this conversation or a "
            "tool returns now outranks all of it; [inferred] = my conclusion, which the user "
            "accepted; #id lets `remember(..., replaces=<id>)` correct a fact)\n" + always
        )
    return "\n\n".join(sections)


def grounding_node(state: AgentState) -> dict:
    start = time.perf_counter()

    # ONE memory selection per turn: its always half is query-independent and rides the
    # stable block, its by-match half the dynamic one below.
    always, matched, matched_ids = memory_context_split(state.get("current_query", ""))
    stable = stable_grounding(always)
    sections = [now_section()]

    # A skill the user ran this turn by typing /<name> (core/skills): their own procedure for
    # THIS request — the dynamic half, so the stable prefix is untouched and the next turn,
    # which resets `skill`, does not carry it.
    skill = state.get("skill", "")
    if skill:
        sections.append(skill)

    # Selected against THIS request (memory_registry.select_for_context): agent/entities/
    # negative facts only when they share tokens with the query, plus the trailer naming what
    # didn't load — under one cap with the always half above. /trace context shows the exact
    # block, so selection stays auditable. The by-match facts and the memo digest that loaded
    # get their last-used stamped (the expiry signal /memory flags stale on) — the one
    # read-path write, and it touches no fact text; best-effort — a stamp failure must never
    # fail the first node of every turn.
    if matched:
        sections.append(
            "### Memory facts matched to this request (same store; #id as above)\n" + matched
        )
    if matched_ids:
        try:
            mark_used(matched_ids)
        except Exception as exc:
            diag.log(f"grounding_node : memory last-used stamp failed: {exc}")

    # Files the user attached to THIS message with `@path` (resolved + read by mentions.expand in the
    # REPL loop, stashed on state). Folded in here so the agent sees the file contents inline in the
    # request's dynamic grounding. Empty on a turn with no resolvable mentions.
    attachments = state.get("attachments", "")
    if attachments:
        sections.append(attachments)

    dynamic = "\n\n".join(sections)
    context = stable + ("\n\n" + dynamic if dynamic else "")
    diag.log(f"grounding_node : {time.perf_counter() - start:.4f}s")
    out = {"context": context, "context_stable": stable, "context_dynamic": dynamic}
    # An attachment is outside content, and `attachments` is reset at the next turn — so the
    # conversation's own record is set here, the first node of the turn that carried it
    # (core/provenance.of reads it; auto-learn stays off from then on).
    if state.get("attachments"):
        out["outside_seen"] = True
    return out

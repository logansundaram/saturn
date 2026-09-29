import os
import time
from pathlib import Path

import diag

from core.state import AgentState
from config import get_config
from textutil import clip
from stores.memory_registry import memory_context_split, mark_used
from stores.document_registry import read_documents_manifest

"""
Grounding node (re-scoped from the old context_builder).

Its ONLY job is to load the things that are NOT already available to the model:
  - the knowledge-base manifest (so the agent knows what documents it can search),
  - the standing instructions (~/.saturn/SATURN.md, then the workspace's SATURN.md), and
  - persistent memory (stores/memory_registry): the user layer + open commitments + the recent
    memo digest always, agent/entities/negative facts by match against the request, all under
    one cap. (The old user_profile.md / agent_profile.md files were folded into the user and
    agent layers 2026-09-02 — nothing ever wrote them.)

It deliberately does NOT include:
  - the tool inventory  -> the agent's native tool bind carries the catalog; duplicating it
                           here hurts small models.
  - the chat history    -> `messages` is already passed to the model directly (the v2 loop
                           sees the real conversation, so the old recap section is gone).

Built once per turn (the manifest and memory are static within a turn). Dynamic information —
tool results — flows through `messages`, never this frozen grounding string.

The block is built in TWO halves (2026-09-04): `context_stable` — instructions, the manifest, the
query-independent memory layers — is byte-identical across turns while nothing on disk
changed, and `context_dynamic` — memory's by-match selection, attachments — changes every
turn. Every node's prompt sends the stable half as its own message
right after the system prompt and the dynamic half after it, and the idle prime (core/prime.py)
re-sends exactly `stable_grounding()` between turns so the daemon holds a checkpoint at that
message boundary: the next turn's agent call then prefills only what is new
(docs/OPTIMIZATIONS.md, "the prefix cache"). `context` stays the joined block for every reader that
wants the whole thing (/trace context, older checkpoints).
"""

# Standing instructions (the CLAUDE.md equivalent), loaded into context EVERY turn so the user
# can durably steer the agent without re-typing it. Two files (2026-09-28):
#   ~/.saturn/SATURN.md            global — tone, standing rules, "always metric", "never draft
#                                  to my boss without asking"; hand-written, follows the user
#                                  everywhere ($SATURN_HOME overrides the directory);
#   <workspace>/SATURN.md          per-workspace — conventions, goals, what matters here; drafted
#                                  by /init (which still writes the old name, SATURDAY.md — read
#                                  when no SATURN.md exists). Where the two conflict the
#                                  workspace file wins, and the prompt says so.
# Each is capped so a runaway file can't eat the context window.
_INSTRUCTIONS_FILES = ("SATURN.md", "SATURDAY.md")
_GLOBAL_INSTRUCTIONS_ENV = "SATURN_HOME"
_INSTRUCTIONS_CAP = 6000


def global_instructions_path() -> Path:
    """Where the global standing instructions live: `$SATURN_HOME/SATURN.md`, else
    `~/.saturn/SATURN.md`."""
    home = os.environ.get(_GLOBAL_INSTRUCTIONS_ENV) or (Path.home() / ".saturn")
    return Path(home).expanduser() / _INSTRUCTIONS_FILES[0]


def _read_capped(path: Path) -> str:
    if not path.is_file():
        return ""
    # errors="replace": a hand-edited file with a stray non-UTF-8 byte must not fail every turn
    # at the first node.
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > _INSTRUCTIONS_CAP:
        text = text[:_INSTRUCTIONS_CAP] + f"\n… ({path.name} truncated — keep it concise)"
    return text


def _read_instructions() -> "tuple[str, str]":
    """The workspace instructions as (file name, text) — SATURN.md first, the old SATURDAY.md
    when only it exists; ("", "") when neither does."""
    workspace = get_config().path("workspace")
    for name in _INSTRUCTIONS_FILES:
        text = _read_capped(workspace / name)
        if text:
            return name, text
    return "", ""


def _read_global_instructions() -> str:
    try:
        return _read_capped(global_instructions_path())
    except Exception as exc:  # an unreadable home must not fail the first node of every turn
        diag.log(f"grounding_node : global SATURN.md unreadable: {exc}")
        return ""

def stable_grounding() -> str:
    """The query-independent half of the grounding block — what the idle prime re-sends between
    turns. Byte-identical to the `context_stable` the next turn's grounding_node builds unless
    the knowledge base, the instructions files or the always-loaded memory layers changed
    in between (in which case the prime simply misses and the turn prefills it, as before)."""
    sections = ["## Grounding context"]

    global_instructions = _read_global_instructions()
    if global_instructions:
        sections.append(
            "### Standing instructions (~/.saturn/SATURN.md — the user's standing guidance "
            "everywhere; follow it)\n" + global_instructions
        )
    name, instructions = _read_instructions()
    if instructions:
        sections.append(
            f"### Workspace instructions ({name} — the user's standing guidance for this "
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
    always, _matched, _ids = memory_context_split("")
    if always:
        sections.append(
            "### Persistent memory (what the user asked me to remember and what I learned; "
            "#id lets `remember(..., replaces=<id>)` correct a fact)\n" + always
        )
    return "\n\n".join(sections)


def grounding_node(state: AgentState) -> dict:
    start = time.perf_counter()

    stable = stable_grounding()
    sections = []

    # Selected against THIS request (memory_registry.select_for_context): agent/entities/
    # negative facts only when they share tokens with the query, plus the trailer naming what
    # didn't load — under one cap with the always half above. /trace context shows the exact
    # block, so selection stays auditable. The by-match facts and the memo digest that loaded
    # get their last-used stamped (the expiry signal /memory flags stale on) — the one
    # read-path write, and it touches no fact text; best-effort — a stamp failure must never
    # fail the first node of every turn.
    _always, matched, matched_ids = memory_context_split(state.get("current_query", ""))
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
    # REPL loop, stashed on state). Folded in here so the planner/agent/synthesize — which read this
    # context, not the raw messages — all see the file contents inline. Empty on a turn with no
    # resolvable mentions.
    attachments = state.get("attachments", "")
    if attachments:
        sections.append(attachments)

    dynamic = "\n\n".join(sections)
    context = stable + ("\n\n" + dynamic if dynamic else "")
    diag.log(f"grounding_node : {time.perf_counter() - start:.4f}s")
    return {"context": context, "context_stable": stable, "context_dynamic": dynamic}

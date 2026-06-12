"""
Conversation-lifecycle commands — every verb that edits what the model carries forward, in one
module (the /help "conversation" theme; consolidated from one-file-per-command 2026-06-11):

  /clear    start over (fresh state + clean screen)
  /compact  fold older turns into one LLM-written summary
  /rewind   drop the last exchange (files untouched — /undo owns those)
  /retry    regenerate the last answer, or re-run the whole last turn
  /resume   session persistence (autosave + named sessions)

`drop_last_turn` is the shared engine of /rewind and /retry full.
"""

from commands._framework import command, _print
from commands._session import (
    _autosave_file,
    _read_session,
    _session_file,
    _session_payload,
    _sessions_dir,
    _swap_to_messages,
    clear_autosave,
    write_autosave,
)
from commands._utils import is_list_verb, is_remove_verb
from textutil import clip


# ── /clear ───────────────────────────────────────────────────────────────────────────────────
@command(
    "clear",
    "Start a fresh conversation: reset state + clear the screen.",
    aliases=("cls", "reset", "new"),
    details="""
The "new conversation" button. Drops the in-memory conversation — the message history and every
per-turn field (plan, iteration, accumulators) — AND clears the visible terminal, then reprints
the session header. One command for a clean slate.

What is NOT touched: config, model/tier bindings, the RAG corpus, the durable memory store
(remember/recall), and the on-disk trace. The trace survives, so /trace and /trace calls still
show past runs after a clear.

Pass --screen (-s) to ONLY repaint the terminal, leaving the conversation intact.

Aliases /reset and /new are the same fresh-start; /cls too.

Examples:
  /clear            new conversation + clean screen
  /clear --screen   just repaint the terminal, keep the conversation
""",
)
def _clear(ctx, args):
    import subprocess
    import sys

    screen_only = bool(args) and args[0].lower() in ("--screen", "-s", "screen")

    if not screen_only:
        ctx.state = ctx.make_initial_state()

    subprocess.run("cls" if sys.platform == "win32" else "clear", shell=True, check=False)

    if screen_only:
        return

    _reprint_banner(ctx)
    _print("  new conversation — fresh state, no message history.")


def _reprint_banner(ctx) -> None:
    """Repaint the startup session header after a clear, so the fresh slate looks like a new launch.
    Best-effort: a failure here must never undo the reset that already happened."""
    try:
        from config import get_config
        from core.llms import model_id
        from tools.registry import tool as _tools
        from stores.rag import iter_documents
        from tui import ui

        cfg = get_config()
        n_docs = sum(1 for _ in iter_documents())
        ui.banner(f"{cfg.active_tier}:{model_id('tool_caller')}", len(_tools), n_docs, ctx.db_path)
    except Exception:
        pass


# ── /compact ─────────────────────────────────────────────────────────────────────────────────
@command(
    "compact",
    "Summarize older turns into one brief to free up the context window.",
    aliases=("summarize",),
    usage="/compact",
    details="""
Folds the older turns of this conversation into a single dense, LLM-written summary (via the
`utility` model), keeping the most recent turn verbatim so follow-ups still resolve. Use it when a
long session is filling the context window and you'd rather keep going than /reset.

This is the heavier sibling of the automatic per-turn compaction: every turn already collapses old
turns to their Q&A mechanically (no LLM), and the agent ALSO auto-compacts on its own when the
context fills past runtime.compact_threshold (default 85%). /compact triggers the LLM summary on
demand, now.

Lossy by nature — it trades transcript detail for space. Facts, decisions, your stated
preferences, and open threads are preserved; verbatim phrasing and tool-output detail are not. The
durable trace (/trace, /trace calls) is untouched. To clear the conversation entirely instead,
use /reset.

Example:
  /compact
""",
)
def _compact(ctx, args):
    from core.compaction import summarize_messages

    msgs = ctx.state.get("messages", [])
    if len(msgs) < 2:
        _print("  nothing to compact yet — have a few turns first.")
        return
    _print("  compacting older turns (one LLM summary call)…")
    new_msgs, stats = summarize_messages(msgs)
    if stats["summarized_turns"] <= 0:
        _print("  only the most recent turn is present — nothing older to summarize.")
        return
    if stats["after"] >= stats["before"]:
        _print("  summary did not shrink the history — left it unchanged (see logging/diag.log).")
        return
    ctx.state["messages"] = new_msgs
    _print(
        f"  compacted {stats['summarized_turns']} earlier turn(s): "
        f"{stats['before']} → {stats['after']} messages. Most recent turn kept verbatim."
    )


# ── /rewind ──────────────────────────────────────────────────────────────────────────────────
# The conversational complement of /undo: /undo reverts what the last turn did to FILES, /rewind
# reverts what it did to the CONVERSATION (a derailed answer polluting context, a question you
# wish you hadn't asked). Repeatable — each /rewind walks one more turn back. Files are
# deliberately NOT touched (point /undo at those), and the trace keeps its record: rewinding
# edits what the model sees next turn, never the audit trail.

# What one user message gets echoed back as in the confirmation line.
_PREVIEW_CHARS = 70


def drop_last_turn(ctx) -> "str | None":
    """Remove the most recent turn (the last HumanMessage and everything after it) from the
    carried messages, clear the per-turn scratch that referred to it, and re-autosave so a crash
    can't resurrect what was just rewound. Returns the dropped user query, or None when there was
    no turn to drop. Shared with /retry full (rewind + re-run)."""
    from langchain.messages import HumanMessage

    from core.compaction import is_summary
    from core.state import is_steer_message

    msgs = ctx.state.get("messages", [])
    # A turn starts at a REAL user message: a standalone mid-turn steer note belongs to the turn
    # it corrected (slicing there would leave the question + half a scratchpad behind), and a
    # compaction summary is carried history, not a turn.
    human_idxs = [
        i
        for i, m in enumerate(msgs)
        if isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m)
    ]
    if not human_idxs:
        return None
    boundary = human_idxs[-1]
    dropped_query = str(msgs[boundary].content)
    ctx.state["messages"] = msgs[:boundary]

    # The per-turn scratch (plan, accumulators, current_query) described the turn that just got
    # dropped — clear it so nothing (notably /retry's re-synthesize) can act on a rewound turn.
    fresh = ctx.state
    fresh["current_query"] = ""
    fresh["plan"] = []
    fresh["iteration"] = 0
    fresh["agent_nudges"] = 0
    fresh["replans"] = 0
    fresh["tools_called"] = []
    fresh["tool_results"] = []
    fresh["documents_retrieved"] = []
    fresh["tool_events"] = []
    # Gate decisions belong to the dropped turn too — a lingering record would violate the
    # "gate_events empty means the human was never asked" invariant /glass and exports rely on.
    fresh["gate_events"] = []

    # write_autosave skips an empty conversation by design (a fresh launch + quit must not wipe
    # the previous session) — but rewinding the ONLY turn empties it deliberately, so clear the
    # slot explicitly or a crash/quit + /resume would resurrect the turn that was just rewound.
    if ctx.state["messages"]:
        write_autosave(ctx.state)
    else:
        clear_autosave()
    return dropped_query


@command(
    "rewind",
    "Drop the last exchange from the conversation (files untouched).",
    usage="/rewind",
    details="""
Removes the most recent turn — your last message and everything the agent said and gathered in
response — from the conversation the model carries forward, so a derailed or unwanted exchange
stops polluting the context. Repeat to walk further back, one turn per /rewind.

Scope (deliberate):
  - conversation only — file changes are NOT reverted; /undo does that (the two compose:
    /rewind + /undo fully unwinds a bad turn).
  - the trace is NOT rewritten — /trace still shows the full record of what actually ran.
    Rewinding changes what the model sees next turn, never the audit trail.
  - the autosave slot is updated immediately, so quitting after a /rewind stays rewound.

See also: /retry (regenerate the last answer), /clear (drop the whole conversation).
""",
)
def _rewind(ctx, args):
    n_before = len(ctx.state.get("messages", []))
    dropped = drop_last_turn(ctx)
    if dropped is None:
        _print("  nothing to rewind — the conversation is empty.")
        return
    n_dropped = n_before - len(ctx.state.get("messages", []))
    _print(f'  rewound the last turn — dropped {n_dropped} message(s) ("{clip(dropped, _PREVIEW_CHARS)}").')
    remaining = sum(
        1 for m in ctx.state.get("messages", []) if m.__class__.__name__ == "HumanMessage"
    )
    if remaining:
        _print(f"  {remaining} earlier turn(s) remain; /rewind again to keep walking back.")
    else:
        _print("  the conversation is now empty.")
    _print("  (files were not touched — /undo reverts those.)")


# ── /retry ───────────────────────────────────────────────────────────────────────────────────
# Two depths, matching the two ways a turn disappoints:
#
#   bare /retry    the gathering was fine but the ANSWER was weak — re-run only the synthesize
#                  step over the same gathered results (tool results, retrieved documents,
#                  draft), replacing the last answer. No tools re-run, no gate, one LLM call.
#   /retry full    the whole TURN went wrong — rewind it (drop_last_turn above) and re-run the
#                  same question as a fresh agent turn, plan and tools and all.
#
# The bare form leans on the fact that the loop's per-turn scratch (tool_results,
# documents_retrieved, current_query, plan) persists on state until the NEXT turn starts —
# exactly the inputs synthesize_node needs. A /clear, /rewind, or /resume in between clears that
# scratch, so /retry then points the user at /retry full.


@command(
    "retry",
    "Regenerate the last answer (or re-run the whole last turn).",
    aliases=("regenerate",),
    usage="/retry [full]",
    details="""
  /retry        regenerate just the final answer from what the last turn already gathered —
                the same tool results, retrieved documents, and draft are synthesized again
                (one LLM call, no tools re-run, nothing re-gated) and the new answer replaces
                the old one in the conversation. Use when the research was right but the
                write-up wasn't.
  /retry full   rewind the last turn (like /rewind) and re-run its question from scratch as a
                fresh agent turn — new plan, new tool calls, the usual gates. Use when the turn
                itself went sideways.

Notes:
  - the bare form needs the last turn's working state, which survives until the next query;
    after /clear, /rewind, or /resume only /retry full is available.
  - the regenerated answer replaces the previous one in the conversation and the autosave; the
    trace keeps the original turn's record (the regeneration is not a traced run).
  - file changes are never re-run or reverted by either form (/undo owns files).
""",
)
def _retry(ctx, args):
    if args and args[0].lower() in ("full", "--full", "-f"):
        return _retry_full(ctx)
    if args:
        _print(f"  unknown argument {args[0]!r} — usage: /retry [full]")
        return
    return _retry_synthesize(ctx)


def _last_query(ctx) -> "str | None":
    from langchain.messages import HumanMessage

    from core.compaction import is_summary
    from core.state import is_steer_message

    # Last REAL question: a standalone mid-turn steer note is a correction to a turn, not the
    # turn's query (requeueing it would re-run the correction without the question), and a
    # compaction summary is carried history.
    for m in reversed(ctx.state.get("messages", [])):
        if isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m):
            return str(m.content)
    return None


def _retry_full(ctx):
    query = _last_query(ctx)
    if not query:
        _print("  nothing to retry — the conversation is empty.")
        return
    drop_last_turn(ctx)
    # The REPL loop consumes `requeue` right after this handler returns and runs it as an
    # ordinary agent turn (agent.main) — fresh plan, fresh gates, fresh trace run.
    ctx.requeue = query
    _print("  re-running the last turn from scratch…")


def _retry_synthesize(ctx):
    from langchain.messages import AIMessage

    state = ctx.state
    if not state.get("current_query"):
        _print("  no working state from a previous turn to regenerate from "
               "(it clears on /clear, /rewind, and /resume) — try /retry full.")
        return
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
        _print("  the last turn left no final answer to regenerate — try /retry full.")
        return

    # Pop the old answer so synthesize sees the conversation exactly as it did the first time
    # (its draft-answer input is whatever now trails the messages); restore it on any failure so
    # a model hiccup can't eat the only answer the user has.
    popped = msgs.pop()
    _print("  regenerating the answer from the last turn's gathered results…")
    try:
        from nodes.synthesize import synthesize_node

        out = synthesize_node(state)
    except Exception:
        msgs.append(popped)
        raise

    new_msg = out["messages"][0]
    msgs.append(new_msg)
    state["tok_per_sec"] = out.get("tok_per_sec", 0.0)
    state["context_tokens"] = out.get("context_tokens", state.get("context_tokens", 0))
    write_autosave(state)

    from tui import ui

    ui.response(str(new_msg.content))


# ── /resume ──────────────────────────────────────────────────────────────────────────────────
@command(
    "resume",
    "Sessions: resume the autosave, or save/load/list/delete/rename named sessions.",
    aliases=("continue",),
    usage="/resume [<name> | save [name] | list | delete <name|n> | rename <old> <new>]",
    details="""
The one front door to session persistence. (The old /save and /load were folded in here —
one command, not three.)

  /resume                   restore the autosave slot — the live conversation is autosaved on
                            /quit and after every turn (per-turn db.sqlite checkpoints are
                            pruned, so this slot is what survives a quit, crash, or Ctrl-C).
                            Typically the first thing you type in a new session.
  /resume <name>            restore a named session saved earlier.
  /resume save [name]       save the current conversation under a name (timestamped if
                            omitted); a matching name overwrites. Only messages are persisted —
                            per-turn scratch (plan, iteration, tool results) is rebuilt fresh.
  /resume list              list the named sessions on disk (numbered).
  /resume delete <name|n>   delete a named session — by name or its /resume list number.
                            (any removal verb works: delete/del/rm/remove/forget/drop. The
                            autosave slot can't be deleted from here.)
  /resume rename <old> <new>  rename a named session.

Restoring rebuilds a fresh state seeded with the saved messages — config, model bindings, and
the RAG corpus are untouched. Files live under database/sessions/ (paths.sessions).

Examples:
  /resume                    continue your previous session
  /resume save research      name and keep this conversation
  /resume research           pick it back up later
  /resume delete 2           drop the second session in /resume list
  /resume rename research llm-notes
""",
)
def _resume(ctx, args):
    if args and args[0].lower() in ("save", "--save", "-s"):
        return _save_named(ctx, args[1:])
    if args and (is_list_verb(args[0]) or args[0].lower() in ("--list", "-l")):
        return _list_saved()
    if args and (is_remove_verb(args[0]) or args[0].lower() == "--delete"):
        return _delete_named(args[1:])
    if args and args[0].lower() in ("rename", "mv", "--rename"):
        return _rename_named(args[1:])
    if args:
        return _load_named(ctx, " ".join(args))

    path = _autosave_file()
    if not path.exists():
        _print("  no previous session to resume — nothing has been autosaved yet.")
        _print("  (a session autosaves on /quit and after each turn; /resume save keeps one by name.)")
        return

    messages, saved_at = _read_session(path)
    if not messages:
        _print("  the autosaved session is empty — nothing to resume.")
        return
    _swap_to_messages(ctx, messages)
    _print(f"  resumed {len(messages)} message(s) from your last session (saved {saved_at}).")
    _print("  conversation history restored — continue where you left off.")


def _save_named(ctx, args):
    import json
    from datetime import datetime

    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  nothing to save — no messages in this session yet.")
        return

    name = " ".join(args) if args else "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _session_file(name)
    existed = path.exists()
    path.write_text(json.dumps(_session_payload(messages), indent=2), encoding="utf-8")
    note = " (overwrote existing)" if existed else ""
    _print(f"  saved {len(messages)} message(s) -> {path.name}{note}")
    _print(f"  restore it with /resume {path.stem}")


def _named_sessions() -> list:
    """The named session files, sorted — the one ordering /resume list shows and the numeric
    arguments of delete resolve against, so the numbers always agree."""
    return sorted(f for f in _sessions_dir().glob("*.json") if not f.stem.startswith("_"))


def _list_saved():
    files = _named_sessions()
    if not files:
        _print("  no named sessions yet — use /resume save [name] first.")
        return
    _print("  named sessions:")
    for i, f in enumerate(files, 1):
        _print(f"    {i}. {f.stem}")
    _print("  restore one with /resume <name>; /resume delete <name|number> drops one.")


def _resolve_named(token: str):
    """A delete/rename target by name or 1-based /resume list number; None when nothing matches.
    Underscore-prefixed slots (the autosave) are unreachable: numbers index the named list only,
    and _session_file sanitizes a leading underscore out of any typed name."""
    files = _named_sessions()
    if token.isdigit() and 1 <= int(token) <= len(files):
        return files[int(token) - 1]
    path = _session_file(token)
    return path if path.exists() else None


def _delete_named(args):
    if not args:
        _print("  usage: /resume delete <name|number>   (numbers as shown by /resume list)")
        return
    token = " ".join(args)
    path = _resolve_named(token)
    if path is None:
        _print(f"  no saved session matching {token!r} (/resume list shows what's on disk).")
        return
    path.unlink()
    _print(f"  deleted session {path.stem!r}.")


def _rename_named(args):
    if len(args) != 2:
        _print("  usage: /resume rename <old> <new>")
        return
    old, new = args
    src = _resolve_named(old)
    if src is None:
        _print(f"  no saved session matching {old!r} (/resume list shows what's on disk).")
        return
    dest = _session_file(new)
    if dest.exists():
        _print(f"  a session named {dest.stem!r} already exists — pick another name "
               "or /resume delete it first.")
        return
    src.rename(dest)
    _print(f"  renamed session {src.stem!r} -> {dest.stem!r}.")


def _load_named(ctx, name: str):
    path = _session_file(name)
    if not path.exists():
        _print(f"  no saved session named {path.stem!r} (/resume list shows what's on disk).")
        return
    messages, saved_at = _read_session(path)
    # Same guard as the bare-autosave path: an empty session must not wipe the live
    # conversation — there is nothing to restore, so leave the current state alone.
    if not messages:
        _print(f"  session {path.stem!r} is empty — keeping the current conversation.")
        return
    _swap_to_messages(ctx, messages)
    _print(f"  loaded {len(messages)} message(s) from {path.name} (saved {saved_at}).")
    _print("  fresh state — conversation history restored.")

"""
Conversation-lifecycle commands — the verbs that manage what the model carries forward, in one
module (the /help "conversation" theme):

  /clear    start over (fresh state + clean screen)
  /resume   session persistence (autosave + named sessions)

(/compact, /rewind, and /retry were CUT 2026-07-07 to trim the "go back / redo" pile-up down to
/clear + /resume here and /undo for files. Auto-compaction still runs on its own — the engine
lives in core/compaction.py + app/session._maybe_autocompact, never needing a manual trigger.)
"""

from commands._framework import command, _print
from commands._session import (
    _autosave_file,
    _read_session,
    _session_file,
    _sessions_dir,
    _swap_to_messages,
    clear_autosave,
    write_session_file,
)
from commands._utils import LIST_VERBS, REMOVE_VERBS


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

The autosave slot IS dropped when a non-empty conversation is cleared — "fresh start" means the
cleared conversation is not silently restorable via /resume.

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
    # Any OTHER argument must error, never fall through to the destructive default — a typo'd
    # `--scren` asking for a repaint must not wipe the conversation (the /mcp precedent: an
    # unrecognized verb stops instead of degrading into the default action).
    if args and not screen_only:
        _print(f"  unknown argument {args[0]!r} — usage: /clear [--screen]")
        return

    if not screen_only:
        # Drop the autosave slot only when a non-empty conversation was actually discarded —
        # write_autosave's empty-guard contract (_session.py): a caller that deliberately empties
        # the conversation clears the slot, or /clear → /quit → /resume resurrects exactly what
        # the user cleared. Unconditional clearing would instead wipe the PREVIOUS session's
        # autosave when /clear is typed at a fresh launch — the case the empty-guard protects.
        had_messages = bool(ctx.state.get("messages"))
        ctx.state = ctx.make_initial_state()
        if had_messages:
            clear_autosave()

    subprocess.run("cls" if sys.platform == "win32" else "clear", shell=True, check=False)

    if screen_only:
        return

    _reprint_banner(ctx)
    _print("  new conversation — fresh state, no message history.")
    # Only true when nothing was cleared this call (an empty conversation leaves the previous
    # session's autosave intact above) — a cleared conversation's slot is gone by design.
    if _autosave_file().exists():
        _print("  (the previous session is still in the autosave — /resume restores it until "
               "your next turn overwrites the slot.)")


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
    verb = _resume_verb(args[0]) if args else None
    if verb == "save":
        return _save_named(ctx, args[1:])
    if verb == "list":
        return _list_saved()
    if verb == "remove":
        return _delete_named(args[1:])
    if verb == "rename":
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


# The /resume subcommand vocabulary — ONE table drives both the router (`_resume_verb`) and the
# reserved-stem screen below, so a subcommand cannot be added without its name being refused as
# a session name at save time (the stranded-session trap this hunk fixed: `/resume save list`
# used to succeed and the session was then only reachable by list number). Per subcommand:
# (bare spellings — these are also the reserved stems, flag spellings — safe_stem strips their
# dashes back to the bare words, so they need no separate reservation).
_RESUME_VERBS = {
    "save": (("save",), ("--save", "-s")),
    "list": (LIST_VERBS, ("--list", "-l")),
    "remove": (REMOVE_VERBS, ("--delete",)),
    "rename": (("rename", "mv"), ("--rename",)),
}


def _resume_verb(token: str) -> "str | None":
    """Which /resume subcommand a first token routes to, or None (load-by-name / bare resume)."""
    t = str(token).lower()
    for verb, (bare, flags) in _RESUME_VERBS.items():
        if t in bare or t in flags:
            return verb
    return None


# Stems the /resume router intercepts BEFORE the load-by-name branch: a session saved under one
# could never be loaded by typing its name (`/resume list` would list, not load, list.json). The
# refusal happens at CREATION (mirroring /policy allow's lone-verb reservation) and compares the
# SANITIZED stem case-insensitively — the router lowercases args[0], so `/resume save LIST`
# strands too, and safe_stem turns flag spellings like `--list` into these same words. Load /
# delete / rename RESOLUTION stays unchanged, so a pre-existing colliding file remains reachable
# (by /resume list number). Derived from the router's own table — never a second hand-kept copy.
_RESERVED_SESSION_STEMS = frozenset(
    w for bare, _flags in _RESUME_VERBS.values() for w in bare
)


def _refuse_reserved_stem(path) -> bool:
    """True (after printing the refusal) when `path`'s stem is a /resume subcommand word."""
    if path.stem.lower() in _RESERVED_SESSION_STEMS:
        _print(f"  {path.stem!r} is a /resume subcommand — a session saved under that name "
               "could never be loaded by name. Pick another name.")
        return True
    return False


def _save_named(ctx, args):
    from datetime import datetime

    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  nothing to save — no messages in this session yet.")
        return

    name = " ".join(args) if args else "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _session_file(name)
    if _refuse_reserved_stem(path):
        return
    existed = path.exists()
    write_session_file(path, messages)
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
    # Renaming ONTO a reserved subcommand word strands the session exactly like saving under
    # one — same refusal, same creation-time boundary.
    if _refuse_reserved_stem(dest):
        return
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

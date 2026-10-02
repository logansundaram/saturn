"""
Conversation-lifecycle commands — the verbs that manage what the model carries forward, in one
module (the /help "conversation" theme):

  /clear    start over (fresh state + clean screen)
  /resume   session persistence (autosave + named sessions)
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
    details="""
The "new conversation" button. Drops the in-memory conversation — the message history and every
per-turn field (plan, iteration, accumulators) — AND clears the visible terminal, then reprints
the session header. One command for a clean slate.

What is NOT touched: config, model/tier bindings, the RAG corpus, the durable memory store
(remember/recall), and the on-disk trace — /trace still shows past runs after a clear.

The autosave slot IS dropped when a non-empty conversation is cleared — "fresh start" means the
cleared conversation is not silently restorable via /resume.
""",
)
def _clear(ctx, args):
    import subprocess

    # An argument must error, never fall through to the destructive default — a typo'd flag
    # must not wipe the conversation (the /mcp precedent: an unrecognized verb stops instead of
    # degrading into the default action).
    if args:
        _print(f"  unknown argument {args[0]!r} — usage: /clear")
        return

    # Drop the autosave slot only when a non-empty conversation was actually discarded —
    # write_autosave's empty-guard contract (_session.py): a caller that deliberately empties
    # the conversation clears the slot, or /clear → /quit → /resume resurrects exactly what
    # the user cleared. Unconditional clearing would instead wipe the PREVIOUS session's
    # autosave when /clear is typed at a fresh launch — the case the empty-guard protects.
    had_messages = bool(ctx.state.get("messages"))
    ctx.state = ctx.make_initial_state()
    if had_messages:
        clear_autosave()

    subprocess.run("clear", shell=True, check=False)

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
        ui.banner(f"{cfg.active_tier}:{model_id()}", len(_tools), n_docs)
    except Exception:
        pass


# ── /resume ──────────────────────────────────────────────────────────────────────────────────
@command(
    "resume",
    "Sessions: resume the autosave, or save/load named sessions.",
    aliases=("continue",),
    usage="/resume [<name> | save [name] | list]",
    details="""
The one front door to session persistence.

  /resume                   restore the autosave slot — the live conversation is autosaved on
                            /quit and after every turn (per-turn db.sqlite checkpoints are
                            pruned, so this slot is what survives a quit, crash, or Ctrl-C).
                            Typically the first thing you type in a new session.
  /resume <name>            restore a named session saved earlier.
  /resume save [name]       save the current conversation under a name (timestamped if
                            omitted); a matching name overwrites. Only messages are persisted —
                            per-turn scratch (plan, iteration, tool results) is rebuilt fresh.
  /resume list              list the named sessions on disk.

Sessions are plain .json files under database/sessions/ (paths.sessions) — delete or rename
one there.

Restoring rebuilds a fresh state seeded with the saved messages — config, model bindings, and
the RAG corpus are untouched.

Examples:
  /resume                    continue your previous session
  /resume save research      name and keep this conversation
  /resume research           pick it back up later
""",
)
def _resume(ctx, args):
    verb = _resume_verb(args[0]) if args else None
    if verb == "save":
        return _save_named(ctx, args[1:])
    if verb == "list":
        return _list_saved()
    if verb in ("remove", "rename"):
        # Intercepted (not treated as a session name) so a habit-typed `/resume rm old` can't
        # misparse; the files are the interface.
        _print("  session delete/rename was cut — sessions are plain files; manage them in:")
        _print(f"    {_sessions_dir()}")
        return
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
# a session name at save time (a session saved as `list` would only be reachable by list
# number). Per subcommand: (bare spellings — these are also the reserved stems, flag spellings —
# safe_stem strips their dashes back to the bare words, so they need no separate reservation).
# The remove/rename verbs are routed only so the router can intercept them with the cut note
# (never a load-by-name misparse).
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
# strands too, and safe_stem turns flag spellings like `--list` into these same words. Derived
# from the router's own table — never a second hand-kept copy.
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
    """The named session files, sorted — the one ordering /resume list shows."""
    return sorted(f for f in _sessions_dir().glob("*.json") if not f.stem.startswith("_"))


def _list_saved():
    files = _named_sessions()
    if not files:
        _print("  no named sessions yet — use /resume save [name] first.")
        return
    _print("  named sessions:")
    for f in files:
        _print(f"    {f.stem}")
    _print(f"  restore one with /resume <name>; the files live in {_sessions_dir()}.")


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


# ── /copy ────────────────────────────────────────────────────────────────────────────────────

def _pbcopy(text: str) -> bool:
    """Put `text` on the clipboard (macOS `pbcopy`); False when that is not possible here."""
    import subprocess
    import sys

    if sys.platform != "darwin":
        return False
    try:
        proc = subprocess.run(["pbcopy"], input=text.encode("utf-8"), capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


@command(
    "copy",
    "Copy Saturn's last answer to the clipboard.",
    details="""
Puts the text of the most recent answer on the clipboard, without the Sources receipt or the
incidents note. macOS only (pbcopy).

The other direction is a mention, not a command: type @clipboard in a message and whatever is
on the clipboard is attached to that message, like an @file ("fix the tone of @clipboard").
Saturn never reads the clipboard on its own.
""",
)
def _copy(ctx, args):
    from nodes.agent import strip_trailers

    answer = ""
    for m in reversed((ctx.state or {}).get("messages") or []):
        if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None):
            answer = strip_trailers(str(getattr(m, "content", "") or ""))
            if answer:
                break
    if not answer:
        _print("  nothing to copy yet — there is no answer in this conversation.")
        return
    if _pbcopy(answer):
        _print(f"  copied the last answer ({len(answer)} characters) to the clipboard.")
    else:
        _print("  could not reach the clipboard (this needs macOS's pbcopy).")

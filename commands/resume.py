from commands._framework import command, _print
from commands._session import (
    _autosave_file,
    _read_session,
    _session_file,
    _session_payload,
    _sessions_dir,
    _swap_to_messages,
)


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
                            (aliases: rm, del. The autosave slot can't be deleted from here.)
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
    if args and args[0].lower() in ("list", "--list", "-l"):
        return _list_saved()
    if args and args[0].lower() in ("delete", "del", "rm", "remove", "--delete"):
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

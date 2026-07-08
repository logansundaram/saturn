"""commands/_session.py + the /resume delete/rename surface — name sanitization (the autosave
slot must be unreachable from user input), payload round-trips, and list/number resolution."""

from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage

from commands._session import (
    _autosave_file,
    _read_session,
    _session_file,
    _session_payload,
    clear_autosave,
    write_autosave,
)
from commands.conversation import (
    _delete_named,
    _load_named,
    _named_sessions,
    _rename_named,
    _resolve_named,
)


def test_session_file_sanitization(isolated_paths):
    assert _session_file("my research!.json").name == "my-research.json"
    # A leading underscore is stripped, so typed names can never collide with the reserved
    # _autosave slot.
    assert _session_file("_autosave").name == "autosave.json"
    assert _session_file("___").name == "session.json"  # nothing left -> fallback stem
    assert _session_file("..\\..\\evil").name == "evil.json"  # path bits dropped


def test_payload_roundtrip(isolated_paths):
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello back")]
    path = _session_file("roundtrip")
    import json

    path.write_text(json.dumps(_session_payload(msgs)), encoding="utf-8")
    loaded, saved_at = _read_session(path)
    assert [m.content for m in loaded] == ["hi", "hello back"]
    assert isinstance(loaded[0], HumanMessage) and isinstance(loaded[1], AIMessage)
    assert saved_at != "?"


def test_autosave_skips_empty_and_writes_nonempty(isolated_paths):
    assert not write_autosave({"messages": []})
    assert write_autosave({"messages": [HumanMessage(content="q")]})
    assert _autosave_file().exists()


def test_clear_autosave_drops_the_slot(isolated_paths):
    """For callers that deliberately emptied the conversation — write_autosave's empty guard
    would otherwise leave the just-cleared turn restorable."""
    write_autosave({"messages": [HumanMessage(content="q")]})
    assert _autosave_file().exists()
    assert clear_autosave()
    assert not _autosave_file().exists()
    assert not clear_autosave()  # already gone — best-effort, not an error


def test_load_named_empty_session_keeps_live_conversation(isolated_paths):
    """Restoring a session with no messages must not wipe the live conversation."""
    _make("hollow")
    live = [HumanMessage(content="precious context")]
    ctx = SimpleNamespace(state={"messages": live}, make_initial_state=lambda: {"messages": []})
    _load_named(ctx, "hollow")
    assert ctx.state["messages"] is live  # untouched — not swapped to a fresh empty state


def _make(name):
    _session_file(name).write_text('{"version": 1, "messages": []}', encoding="utf-8")


def test_named_sessions_excludes_autosave(isolated_paths):
    _make("alpha")
    write_autosave({"messages": [HumanMessage(content="q")]})
    assert [f.stem for f in _named_sessions()] == ["alpha"]


def test_resolve_by_number_matches_list_order(isolated_paths):
    _make("bravo")
    _make("alpha")
    files = _named_sessions()
    assert [f.stem for f in files] == ["alpha", "bravo"]  # sorted — the order /resume list shows
    assert _resolve_named("1").stem == "alpha"
    assert _resolve_named("2").stem == "bravo"
    assert _resolve_named("3") is None
    assert _resolve_named("bravo").stem == "bravo"
    assert _resolve_named("missing") is None


def test_delete_and_rename(isolated_paths):
    _make("keep")
    _make("drop")
    _delete_named(["drop"])
    assert [f.stem for f in _named_sessions()] == ["keep"]
    _rename_named(["keep", "kept"])
    assert [f.stem for f in _named_sessions()] == ["kept"]
    # Renaming onto an existing name is refused.
    _make("other")
    _rename_named(["other", "kept"])
    assert sorted(f.stem for f in _named_sessions()) == ["kept", "other"]


def test_resume_router_and_reserved_stems_share_one_table(isolated_paths):
    """Every bare word the router routes is a reserved session stem — derived from the same
    _RESUME_VERBS table, so a future subcommand cannot recreate the stranded-session trap
    (`/resume save list` succeeding, then `/resume list` listing instead of loading)."""
    from commands.conversation import (
        _RESERVED_SESSION_STEMS,
        _RESUME_VERBS,
        _refuse_reserved_stem,
        _resume_verb,
    )
    from commands._session import _session_file

    for verb, (bare, flags) in _RESUME_VERBS.items():
        for word in bare + flags:
            assert _resume_verb(word) == verb  # the router routes every spelling…
        for word in bare:
            assert word in _RESERVED_SESSION_STEMS  # …and every bare word is reserved
            assert _refuse_reserved_stem(_session_file(word))  # so saving under it refuses
    # Case-insensitive, like the router's lowercasing; non-verbs stay saveable.
    assert _refuse_reserved_stem(_session_file("LIST"))
    assert not _refuse_reserved_stem(_session_file("research"))
    assert _resume_verb("research") is None

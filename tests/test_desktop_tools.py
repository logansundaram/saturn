"""
What the user is pointing at: `read_browser_tab` (the page in the front browser window),
`finder_selection` (the files selected in Finder), the `@clipboard` mention and `/copy`.

Offline: osascript, pgrep, pbpaste and pbcopy never run (their seams are replaced).
"""

import pytest

from tools.applescript import RS, US


def _tool(name):
    from tools.registry import tools_by_name
    return tools_by_name[name]


def _err(name, args):
    from tools.toolspec import ToolError

    with pytest.raises(ToolError) as info:
        _tool(name).invoke(args)
    return f"Error: {info.value}"


@pytest.fixture
def running(monkeypatch):
    """Which apps `pgrep` would find running."""
    from tools import desktop
    apps = {"Safari"}
    monkeypatch.setattr(desktop, "_is_running", lambda app: app in apps)
    monkeypatch.setattr(desktop, "_front_to_back", lambda: [])     # the window order is unknown
    return apps


# ── read_browser_tab ─────────────────────────────────────────────────────────────────────────

def test_desktop_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("read_browser_tab") == "read_only" and "read_browser_tab" in _UNTRUSTED
    assert risk_of("finder_selection") == "read_only" and "finder_selection" not in _UNTRUSTED


def test_safari_tab_is_read_locally_without_launching_anything(mac, running):
    mac.reply(f"https://example.com/lease{US}Lease terms{US}Notice period: 60 days.\nRent: 1,900.\n")
    out = _tool("read_browser_tab").invoke({})
    assert out == {"browser": "Safari", "url": "https://example.com/lease", "title": "Lease terms",
                   "text": "Notice period: 60 days.\nRent: 1,900."}
    assert all(c[0] == "osascript" for c in mac.calls)        # never `open`: a closed browser stays closed
    s = mac.script()
    assert 'tell application "Safari"' in s and "text of t" in s and "current tab of front window" in s


def test_reading_a_tab_is_not_egress(mac, running):
    from trust import egress
    mac.reply(f"https://example.com{US}t{US}body")
    mark = egress.next_seq()
    _tool("read_browser_tab").invoke({})
    assert egress.events_since(mark) == []


def test_chrome_text_comes_from_javascript_and_degrades_to_the_url(mac, running):
    running.clear()
    running.add("Google Chrome")
    mac.reply(f"https://example.com/a{US}A page{US}")
    out = _tool("read_browser_tab").invoke({})
    assert out["browser"] == "Google Chrome" and out["url"] == "https://example.com/a" and out["text"] == ""
    assert "Allow JavaScript from Apple Events" in out["note"] and "web_extract" in out["note"]
    s = mac.script()
    assert "active tab of front window" in s and 'execute t javascript "document.body.innerText"' in s


def test_the_named_browser_wins_and_safari_is_the_default(mac, running):
    running.add("Google Chrome")
    mac.reply(f"u{US}t{US}x")
    assert _tool("read_browser_tab").invoke({})["browser"] == "Safari"
    assert _tool("read_browser_tab").invoke({"browser": "chrome"})["browser"] == "Google Chrome"


def test_the_browser_the_user_was_last_in_is_the_one_read(mac, running, monkeypatch):
    # Safari stays running long after its window is forgotten: "the page I'm on" is the
    # frontmost browser behind the terminal, not the first one in a fixed list.
    from tools import desktop
    running.add("Google Chrome")
    monkeypatch.setattr(desktop, "_front_to_back", lambda: ["Terminal", "Google Chrome", "Finder", "Safari"])
    mac.reply(f"https://example.com/a{US}A page{US}body")
    assert _tool("read_browser_tab").invoke({})["browser"] == "Google Chrome"
    assert len(mac.calls) == 1


def test_a_browser_with_no_window_passes_to_the_next_one(running, monkeypatch):
    import subprocess
    from tools import applescript
    running.add("Google Chrome")

    def fake_run(argv, timeout):
        out = "" if 'application "Safari"' in argv[-1] else f"https://example.com/a{US}A page{US}body"
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    monkeypatch.setattr(applescript, "_run", fake_run)
    assert _tool("read_browser_tab").invoke({})["browser"] == "Google Chrome"
    assert "Safari has no open window" in _err("read_browser_tab", {"browser": "safari"})


def test_window_order_is_read_from_lsappinfo(monkeypatch):
    import subprocess
    from tools import desktop
    listing = 'ASN:0x0-0x2f02f-"Google_Chrome": ASN:0x0-0x17f17f-"Terminal": ASN:0x0-0x38038-"Safari":\n'
    monkeypatch.setattr(desktop.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, listing, ""))
    assert desktop._front_to_back() == ["Google Chrome", "Terminal", "Safari"]


def test_no_browser_running_and_no_window_are_plain_errors(mac, running):
    running.clear()
    assert "no supported browser is running" in _err("read_browser_tab", {})
    assert mac.calls == []
    running.add("Safari")
    mac.reply("")
    assert "no open window" in _err("read_browser_tab", {})
    assert "not a supported browser" in _err("read_browser_tab", {"browser": "Netscape"})


def test_browser_tab_reports_non_mac_honestly(monkeypatch):
    from tools import applescript
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert "only available on macOS" in _err("read_browser_tab", {})


# ── finder_selection ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def launched(tmp_path, isolated_paths):
    from core import workspace
    r = tmp_path / "launch"
    r.mkdir()
    return workspace.set_root(r)


def test_finder_selection_returns_paths_the_file_tools_accept(mac, launched, tmp_path):
    inside = launched / "photos" / "IMG_1.jpg"
    mac.reply(f"{inside}{RS}{launched / 'photos' / 'IMG_2.jpg'}{RS}")
    out = _tool("finder_selection").invoke({})
    assert out == {"selected": ["photos/IMG_1.jpg", "photos/IMG_2.jpg"]}
    s = mac.script()
    assert 'tell application "Finder"' in s and "selection as alias list" in s


def test_finder_selection_says_which_folder_to_add_when_out_of_reach(mac, launched, tmp_path):
    outside = tmp_path / "Desktop"
    outside.mkdir()
    mac.reply(f"{outside / 'a.pdf'}{RS}{outside / 'b.pdf'}{RS}")
    out = _tool("finder_selection").invoke({})
    from core import workspace
    assert out["selected"] == [workspace.relative(outside / "a.pdf"), workspace.relative(outside / "b.pdf")]
    assert f"/add-dir {workspace.display(outside)}" in out["note"]


def test_finder_selection_empty(mac, launched):
    mac.reply("")
    assert _tool("finder_selection").invoke({}) == "Nothing is selected in Finder."


# ── @clipboard and /copy ─────────────────────────────────────────────────────────────────────

def test_clipboard_mention_attaches_what_is_on_the_clipboard(monkeypatch):
    from core import mentions
    monkeypatch.setattr(mentions, "_clipboard", lambda: "Dear Petra,\nThursday works.")
    block, attached = mentions.expand("fix the tone of @clipboard please")
    assert attached == ["clipboard"]
    assert "#### clipboard" in block and "Dear Petra,\nThursday works." in block


def test_clipboard_is_read_only_when_the_user_asks_for_it(monkeypatch):
    from core import mentions
    calls = []
    monkeypatch.setattr(mentions, "_clipboard", lambda: calls.append(1) or "secret")
    assert mentions.expand("what is on my clipboard?") == ("", [])
    assert mentions.expand("mail me@clipboard.com") == ("", [])
    assert calls == []


def test_clipboard_mention_is_clamped_and_an_empty_one_attaches_nothing(monkeypatch):
    from core import mentions
    monkeypatch.setattr(mentions, "_clipboard", lambda: "x" * (mentions._MAX_FILE_CHARS + 50))
    block, _ = mentions.expand("@clipboard")
    assert "truncated" in block and len(block) < mentions._MAX_FILE_CHARS + 400
    monkeypatch.setattr(mentions, "_clipboard", lambda: "")
    assert mentions.expand("@clipboard") == ("", [])
    monkeypatch.setattr(mentions, "_clipboard", lambda: None)       # not macOS / pbpaste failed
    assert mentions.expand("@clipboard") == ("", [])


def test_clipboard_mention_rides_along_with_file_mentions(monkeypatch, tmp_path):
    from core import mentions
    f = tmp_path / "a.txt"
    f.write_text("file text", encoding="utf-8")
    monkeypatch.setattr(mentions, "_clipboard", lambda: "clip text")
    block, attached = mentions.expand(f"compare @{f} with @clipboard")
    assert attached == [str(f), "clipboard"]
    assert "file text" in block and "clip text" in block


def test_copy_puts_the_last_answer_on_the_clipboard(ctx, capsys, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from commands import conversation as copy_cmd
    from commands._framework import dispatch

    copied = []
    monkeypatch.setattr(copy_cmd, "_pbcopy", lambda text: copied.append(text) or True)
    ctx.state["messages"] = [
        HumanMessage(content="draft a reply"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "read_mail", "args": {"id": 1}}]),
        ToolMessage(content="body", tool_call_id="c1", name="read_mail"),
        AIMessage(content="Thursday works for me.\n\nSources:\n  [1] read_mail"),
    ]
    dispatch("/copy", ctx)
    assert copied == ["Thursday works for me."]        # the prose, not the mechanical trailers
    assert "copied" in capsys.readouterr().out


def test_copy_with_nothing_to_copy_and_off_macos(ctx, capsys, monkeypatch):
    from langchain_core.messages import AIMessage
    from commands import conversation as copy_cmd
    from commands._framework import dispatch

    monkeypatch.setattr(copy_cmd, "_pbcopy", lambda text: True)
    dispatch("/copy", ctx)
    assert "nothing to copy" in capsys.readouterr().out
    ctx.state["messages"] = [AIMessage(content="an answer")]
    monkeypatch.setattr(copy_cmd, "_pbcopy", lambda text: False)
    dispatch("/copy", ctx)
    assert "could not" in capsys.readouterr().out
    dispatch("/copy --help", ctx)
    assert "/copy" in capsys.readouterr().out


def test_a_file_named_like_the_clipboard_is_not_the_clipboard(monkeypatch, tmp_path):
    from core import mentions
    calls = []
    monkeypatch.setattr(mentions, "_clipboard", lambda: calls.append(1) or "a password")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clipboard-notes.md").write_text("notes", encoding="utf-8")
    block, attached = mentions.expand("summarize @clipboard-notes.md and @clipboard.txt")
    assert attached == ["clipboard-notes.md"] and "a password" not in block
    assert calls == []


def test_clipboard_mention_survives_sentence_punctuation(monkeypatch):
    from core import mentions
    monkeypatch.setattr(mentions, "_clipboard", lambda: "clip text")
    for text in ("tidy up @clipboard.", "take @clipboard, then shorten it", "@CLIPBOARD?"):
        assert mentions.expand(text)[1] == ["clipboard"], text

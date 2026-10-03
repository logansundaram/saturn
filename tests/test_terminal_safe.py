"""Terminal-safe text (2026-10-01): untrusted bytes never reach the terminal as live escape
sequences; the gate shows bidi and zero-width characters. Every test builds escapes from
literals and renders to StringIO consoles or capsys — nothing raw reaches the real terminal."""

import json
import re
import time

import pytest

from textutil import (has_controls, is_control_picture, json_terminal_safe, visible_controls,
                      visible_controls_n, visible_format_chars)

ESC = "\x1b"
OSC52 = ESC + "]52;c;ZXZpbA==\x07"
OSC8 = ESC + "]8;;https://evil.example/" + ESC + "\\click" + ESC + "]8;;" + ESC + "\\"
ERASE_UP = ESC + "[2K" + ESC + "[1A"
C1_CSI = "\x9b2J"
_RAW = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def no_raw_controls(text: str) -> bool:
    return not _RAW.search(text)


@pytest.mark.parametrize("dirty", [OSC52, OSC8, ERASE_UP, C1_CSI, "a\x00b\x07c\x7fd"])
def test_every_control_becomes_visible(dirty):
    out = visible_controls("before " + dirty + " after")
    assert no_raw_controls(out)
    assert out.startswith("before ") and out.endswith(" after")


def test_escape_shows_as_its_control_picture():
    assert visible_controls(ESC + "[2J") == "␛[2J"
    assert visible_controls("\x07") == "␇"
    assert visible_controls("\x7f") == "␡"


def test_c1_shows_as_its_seven_bit_form():
    assert visible_controls("\x9b2J") == "␛[2J"
    assert visible_controls("\x9d8;;x") == "␛]8;;x"


def test_sgr_colour_is_removed_not_shown():
    assert visible_controls(ESC + "[1;32mok" + ESC + "[0m") == "ok"
    assert visible_controls("\x9b31mred") == "red"


def test_carriage_returns_become_lines():
    assert visible_controls("a\r\nb") == "a\nb"
    assert visible_controls("safe text\rrm -rf ~") == "safe text\nrm -rf ~"
    assert visible_controls("10%\r50%\r100%") == "10%\n50%\n100%"


@pytest.mark.parametrize("clean", [
    "plain ascii", "tabs\tand\nnewlines\n", "naïve café — 日本語 — Привет",
    "עברית ועربية", "family 👨\u200d👩\u200d👧 and flags 🇵🇹", "\u202eRTL override kept here", "",
])
def test_ordinary_text_is_untouched_and_the_same_object(clean):
    assert visible_controls(clean) is clean


def test_idempotent():
    once = visible_controls("x " + OSC52 + OSC8 + ERASE_UP + C1_CSI + "\r\x00")
    assert visible_controls(once) == once


def test_count_is_pictured_characters_only():
    out, n = visible_controls_n(ESC + "[31mred" + ESC + "[0m " + ESC + "[2K\r\n")
    assert out == "red ␛[2K\n" and n == 1


def test_none_reads_as_empty():
    assert visible_controls(None) == "" and visible_controls_n(None) == ("", 0)


def test_fast_on_a_full_observation():
    dirty = ((OSC52 + ERASE_UP + ESC + "[32mok" + ESC + "[0m\r\n") * 600)[:12000]
    start = time.perf_counter()
    for _ in range(20):
        visible_controls(dirty)
    assert (time.perf_counter() - start) / 20 < 0.005


def test_helpers():
    assert has_controls("a" + ESC) and not has_controls("a\tb\n")
    assert is_control_picture("␛") and is_control_picture("␡")
    assert not is_control_picture("e")


def test_format_chars_are_shown_by_code_point():
    assert visible_format_chars("ls \u202efdp.exe") == "ls ⟨U+202E⟩fdp.exe"
    assert visible_format_chars("pass\u200bword") == "pass⟨U+200B⟩word"
    assert visible_format_chars("plain") == "plain"
    once = visible_format_chars("a\u2066b\ufeff")
    assert visible_format_chars(once) == once


def test_json_terminal_safe_round_trips_exactly():
    payload = {"answer": "a\x1bb\x7fc\x9bd\u2028e é 🙂"}
    dumped = json_terminal_safe(json.dumps(payload, ensure_ascii=False))
    assert no_raw_controls(dumped) and "\u2028" not in dumped
    assert json.loads(dumped) == payload
    assert "é 🙂" in dumped  # ordinary non-ASCII stays readable


# ── Task 2: the source layer ──────────────────────────────────────────────────────────────────

def _run_fake_tool(monkeypatch, name, output):
    from langchain.messages import AIMessage

    import nodes.tools as tn

    class Fake:
        def invoke(self, args):
            return output

    monkeypatch.setitem(tn.tools_by_name, name, Fake())
    msg = AIMessage(content="", tool_calls=[{"name": name, "args": {"q": "x"}, "id": "c1"}])
    return tn.tool_node({"messages": [msg]})


DIRTY = "page " + OSC52 + " " + OSC8 + " " + ERASE_UP + " " + C1_CSI + " end"


def test_tool_output_is_neutralised_before_state(monkeypatch):
    delta = _run_fake_tool(monkeypatch, "calculate", DIRTY)
    content = delta["messages"][0].content
    assert no_raw_controls(content)
    assert "␛]52;c;ZXZpbA==␇" in content           # visible, not dropped
    assert all(no_raw_controls(r) for r in delta["tool_results"])
    assert no_raw_controls(delta["tool_events"][0]["result"])  # the rail's preview


def test_the_observation_says_symbols_are_not_literal(monkeypatch):
    from nodes.tools import CONTROL_NOTE

    delta = _run_fake_tool(monkeypatch, "calculate", "a" + ESC + "[2Jb")
    assert delta["messages"][0].content == "a␛[2Jb" + CONTROL_NOTE.format(n=1)


def test_clean_and_colour_only_output_carries_no_note(monkeypatch):
    clean = _run_fake_tool(monkeypatch, "calculate", "4")
    assert clean["messages"][0].content == "4"
    coloured = _run_fake_tool(monkeypatch, "calculate", ESC + "[32m4" + ESC + "[0m")
    assert coloured["messages"][0].content == "4"


def test_a_failed_call_is_neutralised_too(monkeypatch):
    from langchain.messages import AIMessage

    import nodes.tools as tn
    from tools.toolspec import ToolError

    class Boom:
        def invoke(self, args):
            raise ToolError("server said " + OSC52)

    monkeypatch.setitem(tn.tools_by_name, "calculate", Boom())
    msg = AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "c1"}])
    delta = tn.tool_node({"messages": [msg]})
    assert no_raw_controls(delta["messages"][0].content)
    assert delta["messages"][0].additional_kwargs["saturn_status"] == "error"


def test_file_attachments_are_neutralised(tmp_path, monkeypatch):
    from core import mentions

    f = tmp_path / "notes.txt"
    f.write_text("hello " + OSC52, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    block, paths = mentions.expand("look at @notes.txt")
    assert paths and no_raw_controls(block) and "␛]52" in block


def test_bang_attachment_is_neutralised_but_the_output_is_not_rewritten():
    from app import bang

    out = "ok " + ERASE_UP
    block = bang.attachment("cat x", out, 0)
    assert no_raw_controls(block) and "␛[2K" in block
    assert out == "ok " + ERASE_UP  # the REPL still prints the user's own output as their shell would


# ── Task 3: the quarantine finding ────────────────────────────────────────────────────────────

@pytest.fixture
def gate_mode(monkeypatch):
    from config import get_config
    from trust import quarantine

    quarantine.reset_turn()
    monkeypatch.setitem(get_config()._data.setdefault("runtime", {}), "quarantine", "gate")
    yield
    quarantine.reset_turn()


@pytest.mark.parametrize("seq", [OSC52, OSC8, ERASE_UP, C1_CSI, ESC + "P1$r" + ESC + "\\"])
def test_escape_sequences_in_untrusted_output_flag_and_arm_the_gate(monkeypatch, gate_mode, seq):
    from trust import quarantine

    delta = _run_fake_tool(monkeypatch, "web_extract", "article text " + seq)
    assert "terminal-escape" in delta["tool_events"][0]["quarantine"]
    assert "QUARANTINE WARNING" in delta["messages"][0].content
    assert quarantine.gate_pending()


def test_sgr_colour_in_shell_output_never_flags(monkeypatch, gate_mode):
    from trust import quarantine

    out = ESC + "[1;31mFAILED" + ESC + "[0m tests/test_x.py " + ESC + "[32m3 passed" + ESC + "[0m"
    delta = _run_fake_tool(monkeypatch, "run_shell", out)
    assert "quarantine" not in delta["tool_events"][0]
    assert not quarantine.gate_pending()
    assert delta["messages"][0].content == "FAILED tests/test_x.py 3 passed"


def test_scan_matches_raw_and_visible_forms():
    from trust import quarantine

    assert {f.kind for f in quarantine.scan("x " + OSC52)} == {"terminal-escape"}
    assert {f.kind for f in quarantine.scan("x ␛[2K")} == {"terminal-escape"}
    assert quarantine.scan("x ␛[31m colour only") == []


# ── Task 4: edit_file ─────────────────────────────────────────────────────────────────────────

def test_edit_file_explains_a_pictured_control(isolated_paths, tmp_path):
    from core import workspace
    from tools.files import edit_file
    from tools.toolspec import ToolError

    root = tmp_path / "work"
    root.mkdir()
    workspace.set_root(root)
    (root / "banner.sh").write_text("echo '" + ESC + "[1mHello" + ESC + "[0m'\n", encoding="utf-8")
    with pytest.raises(ToolError, match="control character"):
        edit_file.invoke({"file_path": "banner.sh",
                          "old_string": "echo '␛[1mHello", "new_string": "echo 'Hi"})
    assert (root / "banner.sh").read_text(encoding="utf-8").startswith("echo '" + ESC)


def test_edit_file_plain_not_found_is_unchanged(isolated_paths, tmp_path):
    from core import workspace
    from tools.files import edit_file
    from tools.toolspec import ToolError

    root = tmp_path / "work"
    root.mkdir()
    workspace.set_root(root)
    (root / "a.txt").write_text("one two", encoding="utf-8")
    with pytest.raises(ToolError, match="was not found"):
        edit_file.invoke({"file_path": "a.txt", "old_string": "three", "new_string": "four"})

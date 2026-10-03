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

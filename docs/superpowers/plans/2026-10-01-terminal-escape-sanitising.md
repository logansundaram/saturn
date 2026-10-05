# Terminal Escape Sanitising Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No byte that came from outside Saturn (a tool's output, the model's text, an old
recorded run) can reach the user's terminal as a live escape sequence, and no bidi or
zero-width character can make the approval gate show something other than what will run.

**Architecture:** Two layers over one pure function in `textutil.py`. At the **source**,
`nodes/tools.py` (and the two other ways outside text enters the prompt: `@file` / `@clipboard`
attachments and `!cmd` output) turns every control character into a visible symbol (`␛`, `␇`…)
before clamping, quarantine, state, the trace and the model — so the record is clean and replay
is safe by construction. At the **sink**, the one Rich console (`tui/ui/_base._console`), the
slash-command printer (`commands/_framework._print`), headless stdout/stderr and the diag log
apply the same function, which covers model-authored text and runs recorded before this change.
The gate additionally shows bidi overrides and zero-width characters as `⟨U+202E⟩`.

**Tech Stack:** Python 3.11+, Rich 15 (`Console._render_buffer`), stdlib `re`/`json`/`logging`,
pytest.

**Spec:** the Design section below. Background: Johann Rehberger, "Terminal DiLLMa: LLM-powered
apps can hijack your terminal via prompt injection"
(embracethered.com/blog/posts/2024/terminal-dillmas-prompt-injection-ansi-sequences/).
`docs/research.md` lists this as T1.

---

## Design

### The threat, with what was verified in this repo (2026-10-01/02)

Saturn's promise is that what you see in the rail and at the gate is what happened. A terminal
does not only print text: an `ESC` byte starts a command. `ESC ] 52 ; c ; <base64> BEL` writes
the clipboard (OSC 52, honoured by iTerm2, kitty, WezTerm and others); `ESC ] 8 ;; <url> ESC \`
makes text a hyperlink to a different address (OSC 8); `ESC [ 1 A` / `ESC [ 2 K` move the cursor
up and erase a line, so later text can visually overwrite an earlier rail line or a gate row; a
bare `CR` returns to column 0 and lets `safe text\rrm -rf ~` print over itself. The published
attack is an LLM app that relays these bytes from injected content (Rehberger, above); his
recommended fix is to encode control characters by default.

In Saturn every `untrusted=True` reader (web, mail, notes, messages, files, the shell, MCP, the
browser tab) can return such bytes, and they reach a terminal by five routes: the rail's
one-line result preview; the model echoing them into its streamed answer; `/trace` views and
`--replay` of a recorded run; slash-command output; headless stdout/stderr. Checked in this
working tree with Rich 15.0.0, rendering to an `io.StringIO` console with `force_terminal=True`:

- Rich's `Text` and `Markdown` pass `ESC ] 52 … BEL` and `ESC [ 2J` through unchanged. Rich's
  own sanitiser, `rich.control.strip_control_codes`, removes only codes 7, 8, 11, 12 and 13, and
  `Text` does not even strip `CR` (`"\r" in Text("ab\rcd").plain` is `True`).
- `textutil.clip` keeps `ESC`; `tui/ui/trace.py::_leaf` (every rail leaf) renders a string
  containing `ESC ] 52` with the sequence intact.
- `textutil.fmt_args` does **not** keep a raw `ESC`: it renders each value with `repr()`, which
  escapes every non-printable character (`'\x1b'`, `'\x9b'`, `'‮'`, `'‍'`). So the
  rail's call line (`_render_tool_events`), the `-q` progress call line and the gate's compact
  `k = repr(v)` row are already safe. The gate's **full-surface** views are not:
  `_render_shell_command`, `_render_full_args`, the write/edit diffs and `_plain_call_lines` print
  the raw string on purpose (byte-faithful, `_wrap_exact`), and so does the `e(xplain)` view.
- `commands/_framework._print` — the printer behind every slash command (`/memory`, `/trace`,
  `/policy egress`, `/mcp`…, ~360 call sites) — is the builtin `print`.
- `app/headless.py` prints the answer with `print(answer)`; `--json` uses
  `json.dumps(…, ensure_ascii=False)`, which escapes C0 (`\u001b`) but writes `DEL` and the C1
  range (`U+0080–U+009F`, which includes the 8-bit CSI `U+009B`) raw. `/trace export`
  (`commands/trace.py`, line ~188) writes files the same way.
- A Console subclass that rewrites the text of non-control segments in `_render_buffer` removes
  the sequences from `Text`, `Markdown` and a `Live` region while Rich's own cursor control for
  `Live` (control segments) still works — prototyped against an `io.StringIO` console.

Bidi and zero-width characters are a related, gate-specific problem: `U+202E` (right-to-left
override) can make a shell command or an address display in a different order than it runs, and
`U+200B` hides a character inside a word. `trust/policy.shell_prefix_rejects` already refuses
non-ASCII in a granted shell prefix; nothing makes them visible in the command the human reads.

### Approaches considered

1. **Sink only** — sanitise at every place text is printed. Covers model text and old records,
   but the model still *sees* raw escapes in observations (and can echo them into a file it
   writes, a memory, a message), the trace stores them, and every future print path must
   remember to call it.
2. **Source only** — neutralise tool output in `nodes/tools.py`. The model, state, the trace and
   replay of *new* runs are clean, and quarantine can scan the visible form. It does not cover
   the model's own text (a model can emit `\u001b` in an answer), runs recorded before this
   change, or attachments that bypass the tools node.
3. **Both, over one function (chosen).** The source layer makes the record clean; the sink layer
   is the guarantee for everything else. The cost is one `str.translate`-class pass per
   observation (~0.25 ms on 12k characters full of escapes, ~0.03 ms on clean text, measured) and
   per printed segment, and nothing on clean text beyond a regex `search`.

For the sink, three mechanisms were weighed: wrapping `_console.file` (would also eat Rich's own
styling escapes — rejected); a Rich render hook (`push_render_hook`) — but `Live` pushes its own
hook later in the list, and its region would escape ours; and overriding
`Console._render_buffer`, the one place Rich turns segments into a string (chosen). It is a
private method, so a test pins it: if a Rich upgrade renames it, the suite fails instead of the
protection silently vanishing.

### The neutraliser (one definition, `textutil.visible_controls`)

- Complete SGR sequences (`ESC [ … m`, colour and bold) are **removed**: they cannot move the
  cursor, removing them can only make text *more* visible (`SGR 8` conceal and fg==bg go away),
  and they are the common case in shell output, where showing `␛[32m` noise would bury the text.
- `CR LF` → `LF`, and a lone `CR` → `LF`: a progress bar becomes several lines and an overwrite
  trick shows both texts. Nothing is hidden.
- Every other C0 control except `TAB` and `LF` becomes its Unicode control picture
  (`U+2400 + code`: `ESC` → `␛`, `BEL` → `␇`, `NUL` → `␀`), `DEL` → `␡`, and a C1 control
  becomes `␛` plus its 7-bit equivalent (`U+009B` CSI → `␛[`, `U+009D` OSC → `␛]`), so a reader
  sees exactly what the byte was and one quarantine pattern covers both encodings.
- Text with no control character is returned as the same object (fast path). The function is
  idempotent: its output contains no character it rewrites.
- It never touches tabs, newlines, non-ASCII letters, emoji, or bidi/format characters (RTL
  languages and emoji ZWJ sequences are legitimate in answers). Those are made visible **only at
  the gate** by a second function, `visible_format_chars`.

### Decisions on the edges

- **`edit_file` after `read_file` of a file that holds control bytes.** The model sees `␛` and
  would send an `old_string` containing `␛`, which never matches the raw `ESC` in the file. The
  tool refuses with a `ToolError` that says exactly that and offers `write_file`, instead of the
  generic "not found" (reverse-mapping `␛` back to `ESC` was rejected: ambiguous when a file
  holds a literal `␛`, and a rare case does not earn a translation layer). The observation
  itself carries a one-line note when controls were pictured, so the model knows they are not
  literal text.
- **`run_shell` colour.** Commands run on a pipe, so most tools print no colour; when one does
  (forced colour, `ls --color=always`), the SGR codes are removed. `run_shell`'s environment is
  left alone (`NO_COLOR` would change command behaviour the user did not ask to change).
- **A quarantine finding, `terminal-escape`.** OSC / DCS / APC / PM sequences and cursor-moving
  or erasing CSI sequences in untrusted output have no innocent reason to be in a web page or an
  email; they arm the rail warning and the one-batch gate escalation like the other kinds. SGR
  is excluded (removed at the source, and harmless), so coloured shell output never flags.
- **`!cmd` passthrough.** The REPL prints the user's own command's output raw, as their shell
  would (their action, their terminal); the copy attached to the next message is neutralised,
  because the model reads it.
- **`--json`.** Control characters inside the answer are not rewritten; `DEL`, C1, `U+2028` and
  `U+2029` are additionally `\u`-escaped in the dumped text, so a consumer gets the exact answer
  back from `json.loads` and a human who `cat`s the file sees no raw control byte. The same goes
  for `/trace export` files.
- **`/copy`.** A pasted `ESC [ 201 ~` ends a terminal's bracketed-paste mode and the rest of the
  paste runs as typed keys ("paste-jacking"), so `/copy` puts the neutralised answer on the
  clipboard.
- **Not a terminal, out of scope:** notification text (`notify/macos.py`, shown by Notification
  Center through `osascript`, already quoted by `_flat`), Mail drafts and notes Saturn writes
  (rendered by those apps), and `write_file` content (a user may legitimately ask for a file with
  escape codes in it; the gate's diff shows it neutralised).

### Assumptions made without asking

1. Control pictures (`␛`) over `\x1b`-style escapes: they are one cell wide, unambiguous to a
   reader, and keep the line length the model and the clamp see close to the original.
2. Removing SGR rather than showing it is acceptable everywhere, including the model's view of
   shell output.
3. A lone `CR` becomes a newline (not a picture), because the honest rendering of an overwrite is
   to show both texts.
4. Bidi/zero-width characters are made visible at the gate only (shell command, full arguments,
   diffs, explain, grant prompts), not in answers or the rail. The rail's call line already
   shows them escaped through `repr`.
5. No dependency pin on Rich: the `_render_buffer` pin test is the guard.
6. `diag.py` may import `textutil` (both stay leaves: `textutil` imports nothing project-side);
   the CLAUDE.md sentence about safe leaves is updated to say so.
7. The trust benchmark gets no new probe in this plan: it needs a live model, and the offline
   end-to-end test (Task 8) asserts the property directly.

---

## Global Constraints

- `textutil.py` imports nothing project-side (it is the leaf every layer imports).
- The neutraliser is pure, idempotent (`f(f(x)) == f(x)`; a clean record replays byte-identical)
  and fast: under 5 ms for a 12,000-character observation full of escapes (measured ~0.25 ms).
- It changes no byte of ordinary text: letters in any script, emoji (including ZWJ sequences),
  tabs and newlines pass through, and clean text returns as the same object.
- No model call; a plain chat turn is unaffected (one call, same prompt bytes — no observation,
  no change).
- State, the trace and replay keep exactly what the source layer produced; no layer may
  reintroduce a raw control byte.
- Tests are offline and render to `io.StringIO` consoles or `capsys`; never print a raw escape to
  the real terminal (in tests, build them with `"\x1b"` literals).
- `diag.log()`, not `print()`, in nodes and tools.
- Tools report failure by raising `toolspec.ToolError`.
- `CHANGELOG.md`: a user-facing bullet under the existing `### Security` heading inside
  `## [Unreleased]`.
- Commit messages: `area: what changed`, lowercase, ending with the repo's attribution line.

## Review Focus

1. **Coloured shell output** (`git -c color.ui=always log`, pytest with forced colour): colours
   vanish, the text is intact, no quarantine flag, no extra gate prompt — pinned in Task 3
   (`test_sgr_colour_in_shell_output_never_flags`).
2. **Progress bars and CRLF** (`\r`-overwritten lines, Windows line endings): every overwritten
   text is visible on its own line — pinned in Task 1 (`test_carriage_returns_become_lines`).
3. **An answer in Arabic or Hebrew, or with emoji ZWJ sequences**: renders unchanged in the
   response; only the gate visualises format characters — pinned in Task 6
   (`test_answers_keep_rtl_and_zwj_text`).
4. **Replaying a run recorded before this change** (raw `ESC` in a stored tool result or
   answer): the sink renders it clean — pinned in Task 5
   (`test_rail_renders_an_old_record_with_raw_escapes_clean`).
5. **Editing a file that holds raw control bytes**: a refusal that names the cause, never a
   silent mismatch or a write of `␛` into the file — pinned in Task 4.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `textutil.py` | modify | `visible_controls`, `visible_controls_n`, `has_controls`, `is_control_picture`, `visible_format_chars`, `json_terminal_safe` — the one definition |
| `nodes/tools.py` | modify | source layer: neutralise every observation before clamp/scan/state; the control note |
| `core/mentions.py` | modify | source layer for `@file` / `@clipboard` attachment blocks |
| `app/bang.py` | modify | source layer for the `!cmd` attachment |
| `trust/quarantine.py` | modify | `terminal-escape` finding kind |
| `tools/files.py` | modify | `edit_file` refusal for pictured controls |
| `tui/ui/_base.py` | modify | `SafeConsole`; `_console` becomes one |
| `tui/ui/art.py` | modify | the startup spill goes through the neutraliser |
| `commands/_framework.py` | modify | `_print` neutralises |
| `tui/ui/approval.py` | modify | gate rows, prompts and grant notes show format characters |
| `app/headless.py` | modify | stdout answer, stderr lines, `--json` |
| `commands/trace.py` | modify | `/trace export` JSON |
| `commands/conversation.py` | modify | `/copy` |
| `stores/memory_registry.py` | modify | `_clean_text` neutralises a fact before it is written |
| `diag.py` | modify | the log formatter neutralises |
| `tests/test_terminal_safe.py` | create | every test in this plan |
| `CLAUDE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md` | modify | docs |

## Merge notes for sibling plans

- `2026-10-01-loop-guards.md` replays a repeated read-only observation from state: it gets the
  neutralised text for free. Nothing to do.
- `2026-10-01-observation-budget.md` replaces `_clamp_observation`: keep the order
  **neutralise → clamp → note → quarantine scan**. The relevance scorer then works on visible
  text, and a stub can never cut a sequence in half (there are none left).
- `2026-10-01-question-is-an-answer.md` deletes the `ask_user` branch of
  `app/headless.headless_approver`; if it lands first, Task 7's edit to that branch disappears
  with it.

---

### Task 1: The neutraliser in `textutil.py`

**Files:**
- Modify: `textutil.py` (append a section after `safe_stem`)
- Test: `tests/test_terminal_safe.py` (create)

**Interfaces:**
- Produces: `visible_controls(text) -> str`; `visible_controls_n(text) -> tuple[str, int]` (the
  int counts characters turned into pictures — not removed SGR, not CR); `has_controls(text) ->
  bool`; `is_control_picture(ch: str) -> bool`; `visible_format_chars(text) -> str`;
  `json_terminal_safe(dumped: str) -> str`. All accept `None` as `""`.

- [ ] **Step 1: Write the failing tests**

```python
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
    "עברית ועربية", "family 👨‍👩‍👧 and flags 🇵🇹", "‮RTL override kept here", "",
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
    assert visible_format_chars("ls ‮fdp.exe") == "ls ⟨U+202E⟩fdp.exe"
    assert visible_format_chars("pass​word") == "pass⟨U+200B⟩word"
    assert visible_format_chars("plain") == "plain"
    once = visible_format_chars("a⁦b﻿")
    assert visible_format_chars(once) == once


def test_json_terminal_safe_round_trips_exactly():
    payload = {"answer": "a\x1bb\x7fc\x9bd e é 🙂"}
    dumped = json_terminal_safe(json.dumps(payload, ensure_ascii=False))
    assert no_raw_controls(dumped) and " " not in dumped
    assert json.loads(dumped) == payload
    assert "é 🙂" in dumped  # ordinary non-ASCII stays readable
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q`
Expected: collection error — `ImportError: cannot import name 'has_controls' from 'textutil'`.

- [ ] **Step 3: Implement**

Append to `textutil.py`:

```python
# ── terminal-safe text ───────────────────────────────────────────────────────────────────────
# A terminal treats ESC (and the 8-bit C1 range) as the start of a command: write the clipboard
# (OSC 52), relink text (OSC 8), move the cursor and erase lines. Text from outside Saturn — tool
# output, the model, an old recorded run — passes through `visible_controls` before it can
# reach a terminal (docs/superpowers/plans/2026-10-01-terminal-escape-sanitising.md):
#   - complete SGR sequences (colour, bold) are removed: they cannot move anything, and removing
#     them only makes text more visible;
#   - CR LF and a lone CR become LF, so an overwrite shows both texts;
#   - every other C0 control but TAB and LF becomes its control picture (ESC -> ␛), DEL -> ␡,
#     and a C1 control becomes ␛ plus its 7-bit form (U+009B CSI -> ␛[).
# Text with no control character comes back as the same object; the result never contains a
# character this rewrites, so it is idempotent.
_SGR = re.compile(r"\x1b\[[0-9;:]*m|\x9b[0-9;:]*m")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _control_picture(m) -> str:
    code = ord(m.group())
    if code < 0x20:
        return chr(0x2400 + code)
    if code == 0x7F:
        return "␡"
    return "␛" + chr(code - 0x40)


def visible_controls_n(text) -> "tuple[str, int]":
    """`visible_controls`, plus how many characters became pictures (removed colour codes and
    carriage returns are not counted — nothing was turned into a symbol for them)."""
    s = "" if text is None else str(text)
    if not _CONTROL.search(s):
        return s, 0
    s = _SGR.sub("", s).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.subn(_control_picture, s)


def visible_controls(text) -> str:
    """`text` with every terminal control made inert and visible (see above)."""
    return visible_controls_n(text)[0]


def has_controls(text) -> bool:
    """Whether `text` holds a character `visible_controls` would rewrite."""
    return bool(_CONTROL.search("" if text is None else str(text)))


def is_control_picture(ch: str) -> bool:
    """Whether `ch` is one of the symbols `visible_controls` writes in place of a control."""
    return "␀" <= ch <= "␡"


# Bidi overrides and isolates, zero-width characters and the BOM: legitimate inside an answer
# (RTL scripts, emoji ZWJ sequences), but at the approval gate they can make a command or an
# address display in a different order, or hide a character. The gate shows them by code point.
_FORMAT = re.compile("[؜​-‏‪-‮⁦-⁩﻿]")


def visible_format_chars(text) -> str:
    """`text` with bidi and zero-width characters shown as `⟨U+202E⟩` — for the gate only."""
    s = "" if text is None else str(text)
    if not _FORMAT.search(s):
        return s
    return _FORMAT.sub(lambda m: f"⟨U+{ord(m.group()):04X}⟩", s)


# json.dumps(ensure_ascii=False) escapes C0 but writes DEL, C1 and the two Unicode line
# separators raw. They can only occur inside JSON strings, where a \u escape is equivalent.
_JSON_RAW = re.compile("[\x7f-\x9f  ]")


def json_terminal_safe(dumped: str) -> str:
    """A `json.dumps(..., ensure_ascii=False)` result with its remaining raw control characters
    \\u-escaped: `json.loads` returns exactly the same value, and `cat` shows no control byte."""
    return _JSON_RAW.sub(lambda m: f"\\u{ord(m.group()):04x}", dumped)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q`
Expected: all pass (≈ 25 passed).

- [ ] **Step 5: Commit**

```bash
git add textutil.py tests/test_terminal_safe.py
git commit -m "textutil: visible_controls — terminal control characters made inert and visible"
```

---

### Task 2: The source layer — tool output, attachments, `!cmd`

**Files:**
- Modify: `nodes/tools.py` (`tool_node`, after `observation = str(observation)`; new constant
  `CONTROL_NOTE`)
- Modify: `core/mentions.py` (`expand`, the final `return`)
- Modify: `app/bang.py` (`attachment`)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: `visible_controls`, `visible_controls_n` (Task 1).
- Produces: `nodes.tools.CONTROL_NOTE: str` (a format string with `{n}`), read by Task 4's error
  text and Task 8's test.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_terminal_safe.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "neutralised or note or attachment"`
Expected: FAIL — raw `\x1b` in the ToolMessage content; `ImportError: cannot import name
'CONTROL_NOTE'`.

- [ ] **Step 3: Implement**

In `nodes/tools.py`, extend the `textutil` import:

```python
from textutil import CALL_RESULT_SEP, clip, fmt_call, head_tail, visible_controls_n
```

Add beside `_MAX_OBSERVATION`:

```python
# Appended when the source layer turned control characters into symbols, so the model does not
# read `␛` as text the page or file literally contains (and edit_file can say why a match fails).
CONTROL_NOTE = ("\n[{n} terminal control character(s) in this output are shown as symbols such "
                "as ␛ (escape); they are not literal text]")
```

In `tool_node`, replace

```python
        observation = str(observation)
        # Clamp what flows back into the model (ToolMessage + paired tool_results) so one large
        # result can't overflow the context window; the UI preview is derived from the same
        # clamped text. The _preview cap above is just for the one-line tool-I/O tree.
        clamped = _clamp_observation(observation)
```

with

```python
        # Terminal controls become visible symbols BEFORE anything else reads the text (the
        # clamp, quarantine, state, the trace, the model, the rail preview): an untrusted page
        # or file must not reach the terminal as a live escape sequence, and the record stays
        # safe to replay (textutil.visible_controls).
        observation, n_controls = visible_controls_n(observation)
        # Clamp what flows back into the model (ToolMessage + paired tool_results) so one large
        # result can't overflow the context window; the UI preview is derived from the same
        # clamped text. The _preview cap above is just for the one-line tool-I/O tree.
        clamped = _clamp_observation(observation)
        if n_controls:
            clamped += CONTROL_NOTE.format(n=n_controls)
```

In `core/mentions.py`, add the import at the top with the others:

```python
from textutil import visible_controls
```

and replace the last line of `expand`:

```python
    return "\n".join(parts), paths
```

with

```python
    # File and clipboard text is read by the model and may be echoed to the terminal: controls
    # become visible symbols, exactly as tool output does (nodes/tools.py).
    return visible_controls("\n".join(parts)), paths
```

In `app/bang.py`, import `visible_controls` from `textutil` and change the first line of
`attachment`:

```python
    body = visible_controls(output)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_quarantine.py tests/test_tool_node_helpers.py tests/test_mentions.py tests/test_bang.py tests/test_tool_failures.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/tools.py core/mentions.py app/bang.py tests/test_terminal_safe.py
git commit -m "tools: terminal controls in tool output, attachments and !cmd become visible symbols"
```

---

### Task 3: The `terminal-escape` quarantine finding

**Files:**
- Modify: `trust/quarantine.py` (`_PATTERNS`)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: the neutralised form from Task 2 (`␛]`, `␛[…A` etc.).
- Produces: finding kind `"terminal-escape"`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "flag or scan_matches"`
Expected: FAIL — `KeyError: 'quarantine'` / empty finding set.

- [ ] **Step 3: Implement**

In `trust/quarantine.py`, add one entry at the end of `_PATTERNS` (after `chat-markup`):

```python
    # Terminal escape sequences (2026-10-01): an OSC / DCS / APC / PM string, or a CSI that
    # moves the cursor, erases or switches modes. Matched raw (ESC, 8-bit CSI/OSC) and in the
    # visible form nodes/tools.py writes (␛). Colour (SGR, `…m`) is excluded: harmless, removed
    # at the source, and common in shell output.
    ("terminal-escape", re.compile(
        "[\x1b␛](?:[\\]PX^_]|\\[[0-9;?]*[A-HJKSTfhlsu])|[\x9b\x9d]")),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_quarantine.py tests/test_quarantine_admission.py -q`
Expected: all pass. (`test_scan_quiet_on_ordinary_text` must still pass — none of its samples
contains `ESC` or `␛`.)

- [ ] **Step 5: Commit**

```bash
git add trust/quarantine.py tests/test_terminal_safe.py
git commit -m "quarantine: terminal escape sequences in untrusted output are a finding"
```

---

### Task 4: `edit_file` says why a pictured control cannot match

**Files:**
- Modify: `tools/files.py` (`edit_file`, the `count == 0` branch)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: `has_controls`, `is_control_picture` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run the tests to verify the first fails**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k edit_file`
Expected: the first FAILS (`Regex pattern did not match` — the generic "was not found" text);
the second passes.

- [ ] **Step 3: Implement**

In `tools/files.py`, import `has_controls, is_control_picture` from `textutil`, and replace

```python
    count = content.count(old_string)
    if count == 0:
        raise ToolError(
```

with

```python
    count = content.count(old_string)
    if count == 0 and has_controls(content) and any(is_control_picture(c) for c in old_string):
        # read_file showed this file's control characters as symbols (nodes/tools.py); the
        # file holds the raw character, which a displayed copy can never match.
        raise ToolError(
            f"old_string contains a symbol such as ␛ that read_file shows in place of a terminal "
            f"control character; {file_path} holds the control character itself, which edit_file "
            "cannot match from the displayed copy. Edit text that does not include those "
            "symbols, or rewrite the whole file with write_file."
        )
    if count == 0:
        raise ToolError(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_tool_failures.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tools/files.py tests/test_terminal_safe.py
git commit -m "files: edit_file says why a displayed control character cannot match"
```

---

### Task 5: The sink — one safe console, the command printer, the startup spill

**Files:**
- Modify: `tui/ui/_base.py` (new `SafeConsole`; `_console = SafeConsole(highlight=False)`)
- Modify: `commands/_framework.py` (`_print`)
- Modify: `tui/ui/art.py` (`splash`, the `spill` write)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: `visible_controls` (Task 1).
- Produces: `tui.ui._base.SafeConsole` (a `rich.console.Console` subclass); every module that
  binds `_console` from `_base` gets the safe instance unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# ── Task 5: the sink ──────────────────────────────────────────────────────────────────────────

_LIVE_SEQ = re.compile(r"\x1b(?!\[[0-9;]*m)")  # any ESC that is not Rich's own colour code


def _safe_console(**kw):
    import io

    from tui.ui._base import SafeConsole

    buf = io.StringIO()
    return SafeConsole(file=buf, force_terminal=True, width=100, highlight=False, **kw), buf


def test_rich_still_has_the_method_safe_console_overrides():
    from rich.console import Console

    assert "_render_buffer" in vars(Console), (
        "Rich renamed Console._render_buffer: tui/ui/_base.SafeConsole no longer neutralises "
        "terminal escapes — move the override to the new method")


def test_production_console_is_safe():
    from tui.ui import _base

    assert isinstance(_base._console, _base.SafeConsole)


@pytest.mark.parametrize("make", ["text", "markdown", "padding"])
def test_safe_console_neutralises_every_renderable(make):
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.text import Text

    body = "x " + OSC52 + " " + OSC8 + " " + ERASE_UP + " " + C1_CSI + " \r y"
    obj = {"text": Text(body, style="bold"), "markdown": Markdown("# t\n\n" + body),
           "padding": Padding(Text(body), (0, 0, 0, 2))}[make]
    con, buf = _safe_console()
    con.print(obj)
    out = buf.getvalue()
    assert not _LIVE_SEQ.search(out) and "\x9b" not in out and "\r" not in out
    assert "␛]52" in out


def test_safe_console_keeps_rich_colour_and_live_cursor_control():
    from rich.live import Live
    from rich.text import Text

    con, buf = _safe_console()
    with Live(Text("start"), console=con, transient=True, auto_refresh=False) as live:
        live.update(Text("evil " + OSC52, style="green"), refresh=True)
    out = buf.getvalue()
    assert "\x1b]52" not in out
    assert re.search(r"\x1b\[[0-9;]*m", out)          # Rich's own styling survives
    assert re.search(r"\x1b\[(2K|\d*A|\?25)", out)    # Rich's own Live cursor control survives


def test_rail_renders_an_old_record_with_raw_escapes_clean(capsys):
    import importlib

    base = importlib.import_module("tui.ui._base")
    trace = importlib.import_module("tui.ui.trace")
    base._trace_started, base._t_last = False, None
    trace.show_node("tools", {"tool_events": [{
        "name": "web_extract", "args": {"url": "https://x.example"},
        "result": "page " + OSC52 + ERASE_UP, "dur": 0.01, "ok": True}]})
    out = capsys.readouterr().out
    assert no_raw_controls(out) and "␛]52" in out


def test_command_printer_neutralises(capsys):
    from commands._framework import _print

    _print("  fact: likes tea " + OSC52)
    out = capsys.readouterr().out
    assert no_raw_controls(out.rstrip("\n")) and "␛]52" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "sink or safe_console or production_console or old_record or command_printer or rich_still"`
Expected: FAIL — `ImportError: cannot import name 'SafeConsole'`.

- [ ] **Step 3: Implement**

In `tui/ui/_base.py`, replace

```python
from rich.text import Text

from textutil import truncate as _truncate

_console = Console(highlight=False)
```

with

```python
from rich.segment import Segment
from rich.text import Text

from textutil import truncate as _truncate
from textutil import visible_controls


class SafeConsole(Console):
    """The one console every TUI surface prints through. Rich passes ESC and the C1 controls in
    text straight to the terminal (its own sanitiser drops only BEL, BS, VT, FF and CR — and
    `Text` keeps CR), so a tool result, a model answer or an old recorded run could write the
    clipboard, relink text or redraw the rail. Text segments are made inert here
    (textutil.visible_controls), in the one place Rich turns segments into output; control
    segments — Rich's own cursor movement for `Live` — and the colour codes Rich renders from
    styles pass untouched. `_render_buffer` is a private Rich method: tests/test_terminal_safe.py
    fails if a Rich upgrade renames it."""

    def _render_buffer(self, buffer):
        return super()._render_buffer(
            seg if seg.control else Segment(visible_controls(seg.text), seg.style, seg.control)
            for seg in buffer
        )


_console = SafeConsole(highlight=False)
```

In `commands/_framework.py`:

```python
def _print(line: str = "") -> None:
    # Slash-command output carries stored text (memory facts, trace records, MCP descriptions,
    # egress hosts): terminal controls in it are made visible, never sent to the terminal.
    print(visible_controls(line))
```

with `from textutil import visible_controls` added to its imports.

In `tui/ui/art.py` (`splash`), import `visible_controls` from `textutil` and change

```python
        real_out.write(spill if spill.endswith("\n") else spill + "\n")
```

to

```python
        spill = visible_controls(spill)  # startup output can carry a server's or a hook's text
        real_out.write(spill if spill.endswith("\n") else spill + "\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_tui_polish.py tests/test_turn_display_guard.py tests/test_agent_loop.py tests/test_help.py tests/test_command_grammar.py -q`
Expected: all pass. (Tests that monkeypatch a plain `Console` into a module keep working: they
bypass the override, which is fine — production uses `SafeConsole`.)

- [ ] **Step 5: Commit**

```bash
git add tui/ui/_base.py commands/_framework.py tui/ui/art.py tests/test_terminal_safe.py
git commit -m "tui: one safe console — terminal controls in printed text are made visible"
```

---

### Task 6: The gate shows bidi and zero-width characters

**Files:**
- Modify: `tui/ui/approval.py` (`_frame_row`, `_grant_note`, the `ask` closure inside
  `ask_approval`)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: `visible_format_chars` (Task 1); the safe console (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# ── Task 6: the gate ──────────────────────────────────────────────────────────────────────────

def _gate_capture(monkeypatch, answers):
    import builtins
    import importlib
    import types

    approval = importlib.import_module("tui.ui.approval")
    con, buf = _safe_console()
    monkeypatch.setattr("tools.registry", types.SimpleNamespace(TOOL_RISK={}), raising=False)
    monkeypatch.setattr(approval, "_console", con)
    monkeypatch.setattr(approval, "_live_stop", lambda: None)
    monkeypatch.setattr(approval, "_live_start", lambda: None)
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))
    return approval, buf


def test_gate_shows_a_bidi_override_in_a_shell_command(monkeypatch):
    approval, buf = _gate_capture(monkeypatch, ["n"])
    cmd = "rm -rf ~/x ‮#txt.olleh"
    approval.ask_approval({"tool_calls": [
        {"id": "1", "name": "run_shell", "risk": "destructive", "args": {"command": cmd}}]})
    out = buf.getvalue()
    assert "‮" not in out and "⟨U+202E⟩" in out


def test_gate_shows_zero_width_and_controls_in_full_arguments(monkeypatch):
    approval, buf = _gate_capture(monkeypatch, ["n"])
    approval.ask_approval({"tool_calls": [
        {"id": "1", "name": "send_message", "risk": "destructive",
         "args": {"to": "+1555​0100", "text": "hi " + OSC52}}]})
    out = buf.getvalue()
    assert "⟨U+200B⟩" in out and "\x1b]52" not in out


def test_answers_keep_rtl_and_zwj_text(capsys):
    from tui.ui import response

    text = "שלום — family 👨‍👩‍👧"
    response.response(text)
    out = capsys.readouterr().out
    assert "שלום" in out and "👨‍👩‍👧" in out and "U+200D" not in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "gate_shows or rtl"`
Expected: the two gate tests FAIL (raw `‮` / `​` in the output); the RTL test passes.
(If `send_message`'s preview needs a Contacts lookup in this code path, the test still renders
the arguments through `_render_full_args`; if it raises, `_render_call_safely` falls back to
`_plain_call_lines`, which also goes through `_frame_row`.)

- [ ] **Step 3: Implement**

In `tui/ui/approval.py`, import `visible_format_chars` from `textutil` (extend the existing
`from textutil import fmt_args, head_tail`), then:

```python
def _frame_row(*spans: "tuple[str, str]") -> None:
    """One row inside the approval frame: the bold `┃` gutter, then each `(text, style)` span.
    Bidi overrides and zero-width characters are shown by code point — the rows ARE what the
    human approves, and an RTL override can make a command display in another order
    (textutil.visible_format_chars). Terminal controls are handled by the console itself."""
    row = Text()
    row.append("  ┃ ", style="bold")
    for text, style in spans:
        row.append(visible_format_chars(text), style=style)
    _console.print(row)
```

```python
def _grant_note(msg: str) -> None:
    """Disclosure line for an always-grant — yellow, not dim: widening the gate is exactly the
    line the user must not skim past."""
    _console.print(Text(f"  {visible_format_chars(msg)}", style="yellow"))
```

and in `ask_approval`:

```python
    def ask(p):
        return _console.input(visible_format_chars(p), markup=False, emoji=False)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_gate_ux.py tests/test_gate_events.py -q`
Expected: all pass. (`test_gate_ux.py`'s byte-faithful assertions use ASCII commands, which
`visible_format_chars` returns unchanged.)

- [ ] **Step 5: Commit**

```bash
git add tui/ui/approval.py tests/test_terminal_safe.py
git commit -m "gate: bidi overrides and zero-width characters are shown by code point"
```

---

### Task 7: Headless, `/trace export`, diag, memory, `/copy`

**Files:**
- Modify: `app/headless.py` (`_q_progress`, `headless_approver`, the `--json` and plain
  `print` of the answer, `print(f"error: {exc}")`)
- Modify: `commands/trace.py` (the export `write_text`)
- Modify: `diag.py` (`_get` — the formatter)
- Modify: `stores/memory_registry.py` (`_clean_text`)
- Modify: `commands/conversation.py` (`_copy`)
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: `visible_controls`, `json_terminal_safe` (Task 1).
- Produces: `app.headless._stdout(text: str) -> None`; `diag._SafeFormatter`.

- [ ] **Step 1: Write the failing tests**

```python
# ── Task 7: headless, export, diag, memory, /copy ─────────────────────────────────────────────

def test_headless_answer_and_progress_are_neutralised(capsys):
    from app import headless

    headless._stdout("answer " + OSC52)
    assert no_raw_controls(capsys.readouterr().out.rstrip("\n"))
    lines = []
    on_progress = headless._q_progress(lines.append)
    on_progress("tools", {"plan": [{"label": "read " + ERASE_UP, "status": "done"}]})
    assert lines and all(no_raw_controls(ln) for ln in lines)


def test_headless_approver_notes_are_neutralised(capsys, monkeypatch):
    from app import headless
    from trust import policy

    monkeypatch.setattr(policy, "gate_off", lambda: False)
    headless.headless_approver({"type": "approval_request", "notes": ["why " + OSC52],
                                "tool_calls": [{"id": "1", "name": "write_file"}]})
    assert no_raw_controls(capsys.readouterr().err.rstrip("\n"))


def test_diag_formatter_neutralises():
    import logging

    import diag

    rec = logging.LogRecord("saturn.diag", logging.INFO, __file__, 1, "bad " + OSC52, None, None)
    assert no_raw_controls(diag._SafeFormatter("%(message)s").format(rec))


def test_memory_facts_are_stored_neutralised(isolated_paths):
    from stores import memory_registry as mr

    mr.add_memory("likes tea " + OSC52, layer="user", by="user")
    assert all(no_raw_controls(e["text"]) for e in mr.entries())


def test_copy_puts_the_neutralised_answer_on_the_clipboard(monkeypatch):
    from types import SimpleNamespace

    from langchain.messages import AIMessage

    from commands import conversation

    copied = []
    monkeypatch.setattr(conversation, "_pbcopy", lambda text: copied.append(text) or True)
    ctx = SimpleNamespace(state={"messages": [AIMessage(content="done " + ESC + "[201~rm -rf ~")]})
    conversation._copy(ctx, [])
    assert copied and no_raw_controls(copied[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "headless or diag or memory_facts or copy"`
Expected: FAIL — `AttributeError: module 'app.headless' has no attribute '_stdout'`, raw
controls in the other outputs.

- [ ] **Step 3: Implement**

`app/headless.py` — import `json_terminal_safe, visible_controls` from `textutil` (beside the
existing `fmt_args` import), then:

```python
def _stdout(text: str) -> None:
    """The plain-text answer on stdout: a pipe is read by programs and `cat`ted later, so
    terminal controls in the model's text are made visible here too."""
    print(visible_controls(text))
```

In `_q_progress`, wrap the emitter once, right after the `if emit is None:` block:

```python
    raw_emit = emit
    emit = lambda line: raw_emit(visible_controls(line))  # noqa: E731 — plan labels are model text
```

In `headless_approver`, every `print(…, file=sys.stderr)` whose text includes payload values
(the `notes`, the tool names, the `ask_user` question) becomes
`print(visible_controls(<the same f-string>), file=sys.stderr)`. Replace

```python
        print(_json.dumps(payload, ensure_ascii=False, default=str))
```

with

```python
        print(json_terminal_safe(_json.dumps(payload, ensure_ascii=False, default=str)))
```

`print(answer)` with `_stdout(answer)`, and `print(f"error: {exc}", file=sys.stderr)` with
`print(visible_controls(f"error: {exc}"), file=sys.stderr)`.

`commands/trace.py` — in the export, import `json_terminal_safe` from `textutil` and wrap the
dump: `json_terminal_safe(json.dumps(payload, ensure_ascii=False, indent=2))`.

`diag.py` — add `from textutil import visible_controls` below the stdlib imports (textutil is a
leaf with no project imports, so diag stays import-safe from anywhere), define

```python
class _SafeFormatter(logging.Formatter):
    """diag.log lines carry exception text and model output; `tail -f diag.log` and the
    SATURN_DEBUG echo must not hand a terminal a live escape sequence."""

    def format(self, record):
        return visible_controls(super().format(record))
```

and use `_SafeFormatter(...)` in place of `logging.Formatter(...)` for both the file handler and
the stderr echo handler in `_get`. Update the module docstring's "No project imports" sentence
to "imports only `textutil`, itself a leaf".

`stores/memory_registry.py` — `_clean_text`:

```python
    text = " ".join(visible_controls(str(fact or "")).split())
```

with `from textutil import visible_controls` added to its imports.

`commands/conversation.py` — `_copy`, import `visible_controls` from `textutil` and change
`if _pbcopy(answer):` to `if _pbcopy(visible_controls(answer)):` (a pasted `ESC [ 201 ~` would
end bracketed paste and run the rest as typed keys).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py tests/test_cli.py tests/test_memory_registry.py tests/test_memory_layers.py tests/test_replay_and_source.py tests/test_ask_user.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/headless.py commands/trace.py diag.py stores/memory_registry.py commands/conversation.py tests/test_terminal_safe.py
git commit -m "headless: answers, progress, exports, diag lines and memory facts carry no live escapes"
```

---

### Task 8: End-to-end regression

**Files:**
- Test: `tests/test_terminal_safe.py`

**Interfaces:**
- Consumes: everything above; `stores.trace.Tracer(db_path)`, `.start_run(thread_id, query)`,
  `.log_event(run_id, node, delta)`; `nodes.agent.agent_node`, `nodes.agent._generate` (the
  model seam); `tui.ui.response.response(text)`; `tui.ui.trace.show_node(node, delta)`.

- [ ] **Step 1: Write the tests**

```python
# ── Task 8: end to end ────────────────────────────────────────────────────────────────────────

ALL_SEQS = "x " + OSC52 + " " + OSC8 + " " + ERASE_UP + " " + C1_CSI + " y"


def test_a_hostile_tool_result_never_reaches_state_trace_rail_or_answer(
        monkeypatch, gate_mode, tmp_path, capsys):
    import importlib
    import sqlite3

    from stores.trace import Tracer

    delta = _run_fake_tool(monkeypatch, "web_extract", ALL_SEQS)
    assert no_raw_controls(delta["messages"][0].content)

    tracer = Tracer(str(tmp_path / "trace.db"))
    run = tracer.start_run("t1", "q")
    tracer.log_event(run, "tools", delta)
    (data,) = sqlite3.connect(tmp_path / "trace.db").execute(
        "SELECT data FROM events WHERE run_id = ?", (run,)).fetchone()
    assert "\\u001b" not in data and no_raw_controls(data)

    base = importlib.import_module("tui.ui._base")
    trace = importlib.import_module("tui.ui.trace")
    base._trace_started, base._t_last = False, None
    trace.show_node("tools", delta)
    assert no_raw_controls(capsys.readouterr().out)


def test_a_model_answer_that_echoes_an_escape_renders_inert(monkeypatch, capsys):
    from langchain.messages import AIMessage, HumanMessage

    from nodes import agent
    from tui.ui import response

    monkeypatch.setattr(agent, "_generate",
                        lambda i, *, tools, think=False: AIMessage(content="done " + ALL_SEQS))
    state = {"messages": [HumanMessage(content="q")], "current_query": "q", "context": "",
             "plan": [], "iteration": 0, "tools_called": [], "tool_results": [],
             "documents_retrieved": [], "tool_events": [], "gate_events": []}
    final = agent.agent_node(state)["messages"][-1].content
    response.response(final)
    out = capsys.readouterr().out
    assert no_raw_controls(out) and "␛]52" in out
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_terminal_safe.py -q -k "hostile or echoes"`
Expected: both pass (they pin Tasks 2–5 together; if either fails, the task it names regressed).

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (≈ 1,330 tests, ~5 s).

- [ ] **Step 4: Commit**

```bash
git add tests/test_terminal_safe.py
git commit -m "tests: a hostile tool result and an echoing answer stay inert end to end"
```

---

### Task 9: Docs

**Files:**
- Modify: `CLAUDE.md` (Trust stack bullets; the safe-leaves sentence in "Models and config")
- Modify: `docs/ARCHITECTURE.md` (the `nodes/tools.py` bullet in "life of a turn")
- Modify: `CHANGELOG.md` (the first `### Security` heading inside `## [Unreleased]`)

- [ ] **Step 1: CLAUDE.md**

Add a bullet at the end of the "### Trust stack (`trust/`)" list:

```markdown
- Terminal safety — text from outside Saturn never reaches the terminal as a live escape
  sequence. `textutil.visible_controls` is the one rule (SGR removed, CR → LF, every other
  control → its picture `␛`); `nodes/tools.py` applies it to every observation before the clamp
  and quarantine (also `@file`/`@clipboard` and `!cmd` attachments), and the sink applies it
  again: `tui/ui/_base.SafeConsole` (all TUI output), `commands/_framework._print`, headless
  stdout/stderr, `diag.log`. The gate shows bidi/zero-width characters as `⟨U+202E⟩`
  (`visible_format_chars`). Never print model or tool text with a bare `print()`.
```

and change "`config.py`, `diag.py`, `textutil.py` import nothing project-side and are safe
leaves" to "`config.py` and `textutil.py` import nothing project-side and `diag.py` imports only
`textutil`; all three are safe leaves".

- [ ] **Step 2: ARCHITECTURE.md**

In the `nodes/tools.py` bullet, change "clamps the observation" to "makes terminal controls
visible (`textutil.visible_controls`), clamps the observation".

- [ ] **Step 3: CHANGELOG.md**

Under the first `### Security` heading in `## [Unreleased]`, add first:

```markdown
- **A web page, email or file can no longer take over your terminal.** Text can carry hidden
  terminal commands — escape sequences that write your clipboard, turn text into a link to a
  different address, or move the cursor and erase lines so what you see is not what happened.
  Saturn now shows them as visible symbols (`␛`) everywhere it prints: the trace, answers,
  `/trace` and replays (including runs recorded before this change), slash commands, headless
  output and the debug log. Colour codes from shell commands are removed. Content that carries
  such sequences is flagged like an injection attempt, and the next action asks first. At the
  approval prompt, characters that reverse text direction or are invisible are shown by name
  (`⟨U+202E⟩`), so a command cannot display differently from how it runs.
```

- [ ] **Step 4: Verify and commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

```bash
git add CLAUDE.md docs/ARCHITECTURE.md CHANGELOG.md
git commit -m "docs: terminal safety in the trust stack"
```

---

## Self-Review (done while writing)

**1. Spec coverage.** Source layer for tool output, attachments and `!cmd` (Task 2); the
neutraliser's rules, idempotence, speed and ordinary-text guarantee (Task 1); the quarantine kind
(Task 3); `edit_file` after a pictured read (Task 4); the sink for every TUI surface through one
console, slash commands and the startup spill (Task 5); bidi/zero-width at the gate, including
prompts and grant notes, and not in answers (Task 6); headless stdout, stderr and `--json`,
`/trace export`, the diag log, memory writes and `/copy` (Task 7); the end-to-end regression the
design calls for — ToolMessage, trace record, rail, and a model answer through the `_generate`
seam (Task 8); docs (Task 9). Sink paths checked and found already safe: the rail's call line,
`-q`'s call line and the gate's compact argument row (all `fmt_args` → `repr`); notification
text (not a terminal). Every other sink path in the Design section has a task.

**2. Placeholder scan.** Every code step shows its code. Task 7's `headless_approver` edit names
the exact transformation (wrap each payload-bearing f-string in `visible_controls`) rather than
reprinting four unchanged messages.

**3. Type consistency.** `visible_controls(text) -> str`, `visible_controls_n(text) -> (str,
int)`, `has_controls(text) -> bool`, `is_control_picture(ch) -> bool`,
`visible_format_chars(text) -> str`, `json_terminal_safe(dumped) -> str` (Task 1) are the names
used in Tasks 2–8. `CONTROL_NOTE.format(n=…)` (Task 2) is the form Task 2's test asserts.
`SafeConsole` (Task 5) is what Tasks 6 and 8 construct or assert. `_stdout` and `_SafeFormatter`
(Task 7) are defined where their tests import them.

**4. Review Focus.** Each of the five lines names its test and the task that owns it; all five
tests are written out above.

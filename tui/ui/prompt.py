"""
The `»` input line + the startup banner. prompt_toolkit drives a live-highlighted input where a
typed `/command` is colored by how it matches the command set, `@path` mentions stand out and
Tab-complete, and input is multiline: Enter submits; Shift+Enter (on terminals that distinguish
it), Ctrl+Enter, Ctrl+J, Alt+Enter and backslash+Enter all insert a newline. A large paste is
compacted to a `[paste #N +L lines]` chip so a wall of code doesn't flood the prompt — the full
text is kept aside, re-expanded into the message at submit, and Ctrl+E with the cursor on the chip
re-expands it in place for editing. `banner` prints the two-line session header. The live posture
flags (⚠ GATE OFF / ⛓ AIRGAP) ride the prompt's right edge as the rprompt — the status bar that
carries them mid-turn is torn down around every input(), exactly when the user picks their next
action.
"""

import re as _re

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal as _ptk_run_in_terminal
from prompt_toolkit.application.current import get_app as _ptk_get_app
from prompt_toolkit.completion import (
    Completer as _PTKCompleter,
    Completion as _PTKCompletion,
    PathCompleter as _PTKPathCompleter,
)
from prompt_toolkit.document import Document as _PTKDocument
from prompt_toolkit.filters import Condition as _PTKCondition
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES as _PTK_ANSI_SEQ
from prompt_toolkit.key_binding import KeyBindings as _PTKKeyBindings
from prompt_toolkit.keys import Keys as _PTKKeys
from prompt_toolkit.lexers import Lexer as _PTKLexer
from prompt_toolkit.styles import Style as _PTKStyle

from ._base import (
    Text, _console,
    _ACCENT, _DIM, _POSTURE_STYLE, _RISK,
    _active_ctx_window, _git_branch, _human_tokens, _posture_flags, _short_cwd,
)
from .statusbar import _live_stop

_ptk_session = None  # one PromptSession for the process -> free line history across turns


# ── startup banner ─────────────────────────────────────────────────────────────
def banner(model: str, n_tools: int, n_docs: int) -> None:
    """Session header: a quiet two-line identity block — the name + model/tier + context window,
    then a dim facts line (tools · docs · git branch · working dir · the `/help` pointer). The
    posture line below it carries the trust facts."""
    win = _active_ctx_window()
    ctx = _human_tokens(win) if win else "?"
    branch = _git_branch()
    cwd = _short_cwd()

    head = Text("  ")
    head.append("saturday.ai", style=f"bold {_ACCENT}")
    head.append("  ", style=_DIM)
    head.append(model, style="default")
    head.append(" · ctx ", style=_DIM)
    head.append(ctx, style="default")
    _console.print(head)

    info = Text("  ")
    info.append(f"{n_tools} tools · {n_docs} docs", style=_DIM)
    if branch:
        info.append(f" · git {branch}", style=_DIM)
    info.append(f" · {cwd}", style=_DIM)
    info.append(" · ", style=_DIM)
    info.append("/help", style=_ACCENT)
    _console.print(info)


def posture_line() -> None:
    """One line under the banner stating the live trust posture (trust.receipt.posture_spans) —
    deviation-only: a default-safe posture prints NOTHING; the line appears only when something
    is loosened or leaves the machine, with the `/policy` pointer riding along. Also prints
    nothing if the posture can't be read: a guessed posture is worse than none."""
    try:
        from trust import receipt

        spans = receipt.posture_spans()
    except Exception:
        spans = []
    if not spans:
        return
    line = Text("  ")
    for i, (text, kind) in enumerate(spans):
        if i:
            line.append(" · ", style=_DIM)
        line.append(text, style=_POSTURE_STYLE.get(kind, _DIM))
    line.append("   ", style=_DIM)
    line.append("/policy", style=_ACCENT)
    _console.print(line)


# ── input prompt ───────────────────────────────────────────────────────────────
# Live highlight for the `»` line: a `/token` is colored by how it matches the command set, so
# a typo never blends in with a real command. Valid -> cyan, a prefix of some command (mid-type)
# -> yellow, anything else -> red. Args after the token stay dim. The palette mirrors the rest of
# the UI (cyan accent, semantic status colors).
_PTK_STYLE = _PTKStyle.from_dict({
    "prompt": "ansicyan bold",
    "prompt.cont": "ansibrightblack",
    "cmd.valid": "ansicyan bold",
    "cmd.partial": "ansiyellow",
    "cmd.unknown": "ansired bold",
    "cmd.args": "ansibrightblack",
    "mention": "ansibrightblue",
    "paste": "reverse ansibrightblack",  # the [paste #N …] chip — reads as one atomic token
    "posture.gate": "ansired bold",
    "posture.airgap": "ansicyan bold",
    "posture.sep": "ansibrightblack",
})

# posture kind -> rprompt style class (the ptk twin of _POSTURE_STYLE). Only the loud flags ride
# the prompt — a loosened-but-not-open tier stays a status-bar fact.
_POSTURE_PTK = {"risk": "class:posture.gate", "accent": "class:posture.airgap"}


def _posture_rprompt():
    """rprompt fragments for the `»` line — a callable, so the flags are re-derived live on
    each render. Empty when nothing is active, so the default prompt stays clean."""
    frags = []
    for label, kind in _posture_flags() or []:
        if kind not in _POSTURE_PTK:
            continue
        if frags:
            frags.append(("class:posture.sep", " · "))
        frags.append((_POSTURE_PTK[kind], label))
    return frags


# Teach the vt100 parser the Shift/Ctrl+Enter escape sequences modern POSIX terminals emit
# (kitty, foot, Ghostty, WezTerm send CSI-u by default; xterm-family sends the modifyOtherKeys
# form when that mode is on). Mapped to (Escape, Enter) so they land on the same newline
# binding as Alt+Enter. NOTE: prompt_toolkit ships `\x1b[27;2;13~` mapped to plain Enter —
# without the override a Shift+Enter there would *submit*, the opposite of what the user
# meant. Safe to extend at import: the parser's prefix cache is lazy, and nothing has parsed
# yet.
for _seq in ("\x1b[13;2u", "\x1b[13;5u", "\x1b[27;2;13~", "\x1b[27;5;13~"):
    _PTK_ANSI_SEQ[_seq] = (_PTKKeys.Escape, _PTKKeys.ControlM)
del _seq

# An `@mention` inside a normal (non-slash) line: `@` at a word boundary + a run of
# non-space/non-`@`, or a double-quoted run (`@"path with spaces"` — what dragging a file in
# after typing `@` produces; the close quote is optional so it highlights mid-type).
# Highlighted live so the user sees which token will attach a file (mentions.expand resolves
# them for real at submit; see mentions.py).
_MENTION_LEX_RE = _re.compile(r'(?<!\S)@(?:"[^"\n]*"?|[^\s@]+)')
# Same grammar but capturing just the path fragment up to the cursor, for Tab completion —
# the quoted form first (its fragment may contain spaces), then the bare form.
_AT_QUOTED_FRAGMENT_RE = _re.compile(r'(?:^|\s)@"([^"\n]*)$')
_AT_FRAGMENT_RE = _re.compile(r"(?:^|\s)@([^\s@]*)$")

# ── paste compaction (the [paste #N …] chips) ─────────────────────────────────
# A multi-line/huge paste would flood the prompt with raw text; instead it's stored here and
# the buffer gets one compact chip. The chip is ordinary buffer text — deletable like a word —
# and is swapped back for the full text at submit (`_expand_paste_tags`) or in place via
# Ctrl+E (`_ptk_expand_paste`) when something in it needs editing. Ids never reset, so a
# chip recalled from line HISTORY still expands — but the store is BOUNDED (the last
# _PASTE_STORE_MAX pastes): a day-long session of big log pastes must not retain every one
# of them forever. An evicted chip degrades to its literal text (_expand_paste_tags keeps
# unknown ids), exactly like a hand-typed chip.
_PASTE_STORE: dict[int, str] = {}
_PASTE_STORE_MAX = 30
_paste_seq = 0  # last id handed out
# Whether the line being edited (then: the line last submitted) is NOT known to be typed by
# hand: it received a paste of ANY size, or it was recalled from history (Up) — where it may
# have been a paste the first time, edited since or not. A short paste is inserted verbatim
# and looks typed afterwards, so this is the only place that knows; the REPL stamps it on the
# turn's message (core.state.PASTED_KEY) and auto-learn does not take such a line for the
# user's own statement (core/auto_memory).
_pasted = False


def line_was_pasted() -> bool:
    """Whether the line `prompt()` last returned carried a paste or came back from history."""
    return _pasted


def _recalled(buf) -> bool:
    """Whether the buffer is showing a history entry rather than the new line. prompt_toolkit
    keeps the new line last in its working lines; the attribute is private, so a version
    without it reads as recalled (not typed by hand) rather than as typed."""
    lines = getattr(buf, "_working_lines", None)
    return lines is None or buf.working_index != len(lines) - 1

_PASTE_TAG_RE = _re.compile(r"\[paste #(\d+)[^\]\n]*\]")
_PASTE_TAG_LINES = 3    # compact a paste of >= this many lines
_PASTE_TAG_CHARS = 600  # ... or this many chars (single-line walls); dragged paths stay raw

def _paste_tag_at(text: str, pos: int):
    """The chip regex-match spanning (or abutting) cursor offset `pos`, else None."""
    for m in _PASTE_TAG_RE.finditer(text):
        if m.start() <= pos <= m.end():
            return m
    return None

def _expand_paste_tags(text: str) -> str:
    """Replace every chip with its stored full text — run on the submitted line, so the agent
    always sees what was actually pasted. An unknown id (hand-typed chip) stays literal."""
    return _PASTE_TAG_RE.sub(
        lambda m: _PASTE_STORE.get(int(m.group(1)), m.group(0)), text
    )

def _mention_fragments(line: str):
    """Split one plain line into prompt_toolkit (style, text) fragments, coloring `@mention`
    runs and `[paste #N …]` chips. Used for normal turns so the tokens that mean something
    (a file attach, a stored paste) stand out from plain text as they're typed."""
    spans = [(m.start(), m.end(), "class:mention") for m in _MENTION_LEX_RE.finditer(line)]
    spans += [(m.start(), m.end(), "class:paste") for m in _PASTE_TAG_RE.finditer(line)]
    spans.sort()
    frags = []
    pos = 0
    for start, end, style in spans:
        if start < pos:  # overlapping span (mention swallowing a chip) — first one wins
            continue
        if start > pos:
            frags.append(("", line[pos:start]))
        frags.append((style, line[start:end]))
        pos = end
    if pos < len(line):
        frags.append(("", line[pos:]))
    return frags or [("", line)]

def _slash_token(text: str):
    """Split a prompt line into `(lead, token, args)` around the leading `/command` word —
    `lead` is any whitespace before the slash, `token` the command word (no slash, original
    case), `args` the remainder (its leading space included). Returns `None` for a non-slash
    line. The single definition of the `/token` grammar, shared by the lexer and the completer."""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    lead = text[: len(text) - len(stripped)]  # preserve leading whitespace verbatim
    body = stripped[1:]
    cut = len(body)
    for i, ch in enumerate(body):
        if ch.isspace():
            cut = i
            break
    return lead, body[:cut], body[cut:]

class _CommandLexer(_PTKLexer):
    """Colors the first `/token` of the line against a known-command set, live as it's typed.
    Only the command token is styled; normal (non-slash) turns render plain."""

    def __init__(self, names):
        self._names = names  # canonical names + aliases, lowercased, no leading slash

    def _style_for(self, key: str) -> str:
        if key in self._names:
            return "class:cmd.valid"
        if not key or any(n.startswith(key) for n in self._names):
            return "class:cmd.partial"  # lone "/" or still typing a real command
        return "class:cmd.unknown"      # a typo — make it loud

    def lex_document(self, document):
        # Per-line so multiline input (Alt+Enter / paste) highlights each line. Only the
        # FIRST line is treated as a possible `/command` (a slash command is always single-
        # line); every line gets `@mention` highlighting.
        lines = document.lines

        def get_line(lineno):
            line = lines[lineno] if 0 <= lineno < len(lines) else ""
            parsed = _slash_token(line) if lineno == 0 else None
            if parsed is None:
                return _mention_fragments(line)
            lead, token, args = parsed
            frags = []
            if lead:
                frags.append(("", lead))
            frags.append((self._style_for(token.lower()), "/" + token))
            if args:
                frags.append(("class:cmd.args", args))
            return frags

        return get_line

class _CommandCompleter(_PTKCompleter):
    """Tab-completes two things, depending on where the cursor sits:
      - the leading `/command` token of a slash line (against the known command set), and
      - an `@path` mention anywhere in a normal line (against the filesystem, so a file can be
        pulled into the turn inline — see mentions.py).
    Command completion fires only on the first token of a slash line (a space ends it); `@path`
    completion fires on the partial path after an `@` at a word boundary. `display_meta` carries
    each command's one-line summary into the menu."""

    def __init__(self, meta):
        self._meta = meta  # list of (token, summary), tokens lowercased, no leading slash
        # Delegate the actual path matching to prompt_toolkit's PathCompleter; we only carve
        # out the `@`-prefixed fragment and re-base its replacement offset.
        self._paths = _PTKPathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        # `@path` mention completion takes precedence: it can occur anywhere on the line,
        # including inside what would otherwise be a slash command's args.
        before = document.text_before_cursor
        at = _AT_QUOTED_FRAGMENT_RE.search(before) or _AT_FRAGMENT_RE.search(before)
        if at is not None:
            frag = at.group(1)  # the partial path typed after `@` (no `@`)
            sub = _PTKDocument(frag, len(frag))
            for comp in self._paths.get_completions(sub, complete_event):
                yield comp  # offsets are relative to `frag`'s end == the cursor, so they map 1:1
            return

        parsed = _slash_token(before)
        if parsed is None:
            return
        _lead, token, args = parsed
        if args:  # past the command token, into the args
            return
        word = token.lower()
        for tok, summary in self._meta:
            if tok.startswith(word):
                # Replace just the typed token (the leading "/" stays put).
                yield _PTKCompletion(
                    tok,
                    start_position=-len(token),
                    display="/" + tok,
                    display_meta=summary,
                )

# Multiline input: Enter submits; Shift+Enter / Ctrl+Enter / Ctrl+J / Alt+Enter insert a
# newline. Shift+Enter reaches us two ways depending on the terminal: the CSI-u /
# modifyOtherKeys sequences taught to the vt100 parser above (kitty/foot/Ghostty/WezTerm/
# xterm), or — on terminals that simply can't distinguish it (Apple Terminal, default
# iTerm2/VS Code) — the backslash+Enter fallback below. A pasted multi-line block never
# submits: it arrives as ONE BracketedPaste event and is compacted to a chip. When
# the completion menu is open, Enter accepts the highlighted completion rather than submitting.
_PTK_KB = _PTKKeyBindings()

@_PTK_KB.add("enter")
def _ptk_enter(event):
    global _pasted
    buf = event.current_buffer
    if buf.complete_state and buf.complete_state.current_completion:
        buf.apply_completion(buf.complete_state.current_completion)
    else:
        if _recalled(buf):
            _pasted = True  # read before the submit resets the buffer to a new line
        buf.validate_and_handle()  # submit the line

@_PTK_KB.add("escape", "enter")  # Alt/Option+Enter, Esc-then-Enter, and Shift/Ctrl+Enter
@_PTK_KB.add("c-j")              # Ctrl+J everywhere (LF byte); Ctrl+Enter in some terminals
def _ptk_newline(event):
    event.current_buffer.insert_text("\n")

# Backslash+Enter -> newline (the Claude Code convention): the universal fallback for
# terminals where Shift+Enter is indistinguishable from Enter. The backslash is consumed
# (it was a line continuation, not text).
@_PTK_KB.add("\\", "enter")
def _ptk_newline_backslash(event):
    event.current_buffer.insert_text("\n")

@_PTK_KB.add(_PTKKeys.BracketedPaste)
def _ptk_paste(event):
    """One paste = one event. Small pastes insert verbatim (newlines included — never a
    submit); anything bigger is stored and rendered as a `[paste #N +L lines]` chip so the
    prompt stays one clean line. The full text rides into the message at submit."""
    global _paste_seq, _pasted
    _pasted = True  # whatever its size: the line is no longer only what the user typed
    data = event.data.replace("\r\n", "\n").replace("\r", "\n")
    n_lines = data.count("\n") + 1
    if n_lines < _PASTE_TAG_LINES and len(data) < _PASTE_TAG_CHARS:
        event.current_buffer.insert_text(data)
        return
    _paste_seq += 1
    _PASTE_STORE[_paste_seq] = data
    while len(_PASTE_STORE) > _PASTE_STORE_MAX:  # evict oldest (dicts keep insertion order)
        _PASTE_STORE.pop(next(iter(_PASTE_STORE)))
    size = f"+{n_lines} lines" if n_lines > 1 else f"{_human_tokens(len(data))} chars"
    tag = f"[paste #{_paste_seq} {size}]"
    event.current_buffer.insert_text(tag)

    def _notify():
        _console.print(Text(f"  · paste captured as {tag} — sent in full; Ctrl+E on the chip "
                            "edits it", style=_DIM))

    _ptk_run_in_terminal(_notify)

@_PTKCondition
def _cursor_on_paste_tag() -> bool:
    doc = _ptk_get_app().current_buffer.document
    return _paste_tag_at(doc.text, doc.cursor_position) is not None

@_PTK_KB.add("c-e", filter=_cursor_on_paste_tag)
def _ptk_expand_paste(event):
    """Ctrl+E with the cursor on a chip swaps it back to the full pasted text, in place, so
    it can be edited like anything else. Filtered: anywhere else Ctrl+E keeps its normal
    end-of-line meaning."""
    buf = event.current_buffer
    doc = buf.document
    m = _paste_tag_at(doc.text, doc.cursor_position)
    full = _PASTE_STORE.get(int(m.group(1))) if m else None
    if full is None:
        return
    buf.text = doc.text[: m.start()] + full + doc.text[m.end():]
    buf.cursor_position = m.start() + len(full)

@_PTK_KB.add("s-tab")  # Shift+Tab: cycle the gate policy's auto-approve threshold
def _ptk_cycle_permission(event):
    from config import RISK_ORDER
    from trust import policy
    current = policy.tier()
    idx = RISK_ORDER.index(current) if current in RISK_ORDER else 0
    next_tier = RISK_ORDER[(idx + 1) % len(RISK_ORDER)]
    policy.set_tier(next_tier)

    def _notify():
        msg = Text()
        msg.append("  permission: ", style=_DIM)
        msg.append(next_tier, style=_RISK.get(next_tier, "default"))
        msg.append("  (Shift+Tab to cycle)", style=_DIM)
        _console.print(msg)

    _ptk_run_in_terminal(_notify)

def _ptk_continuation(width, line_number, is_soft_wrap):
    """Gutter for continuation lines of a multiline entry — a dim `·` aligned under the `»`."""
    return [("class:prompt.cont", "· ".rjust(width))] if not is_soft_wrap else ""


def prompt(command_meta) -> str:
    """Read the `»` input line. `command_meta` is the `(token, summary)` list: a typed `/command`
    is highlighted live (valid=cyan, typo=red), Tab completes the leading `/command` token or an
    `@path` mention, and the line is multiline: Enter submits; Shift+Enter / Ctrl+Enter / Ctrl+J /
    Alt+Enter (POSIX fallback: backslash+Enter) insert a newline. A large paste renders as a
    `[paste #N …]` chip (Ctrl+E on it re-expands for editing) and is swapped back to the full text
    in the returned line. Active posture flags render live at the line's right edge (rprompt).
    Returns the raw line (slash-command + @mention handling happen upstream)."""
    global _ptk_session, _pasted
    _pasted = False
    _live_stop()  # never read a line under an active Live (also clears a bar left by an error)
    if _ptk_session is None:
        _ptk_session = PromptSession()
    names = {token for token, _ in command_meta}  # valid-command set for the live highlight
    return _expand_paste_tags(_ptk_session.prompt(
        [("class:prompt", "» ")],
        lexer=_CommandLexer(names),
        style=_PTK_STYLE,
        completer=_CommandCompleter(command_meta),
        complete_while_typing=False,  # Tab-triggered, so the menu never fights live typing
        multiline=True,
        key_bindings=_PTK_KB,
        prompt_continuation=_ptk_continuation,
        rprompt=_posture_rprompt,  # live posture flags at the right edge (read each render)
    ))


def answer_question(value: dict) -> str:
    """The ask_user tool's prompt: render the agent's question and read the user's one-line
    answer (the interrupt's resume value). An empty reply / Ctrl-C / EOF returns "" — the tool
    reports "no answer" honestly rather than blocking. Renders through the one block-header
    vocabulary (listing.section); the live bar is already down (run_turn stops the type-ahead
    reader before any interrupt is resolved, and ask() tears down any Live regardless)."""
    from .listing import section

    question = str((value or {}).get("question") or "").strip() or "(no question given)"
    section("the agent asks")
    _console.print(f"  {question}", markup=False)
    return ask("your answer (Enter = no answer) » ")


def pause_prompt(value: dict) -> dict:
    """The Esc pause: show where the turn is, then one line decides — Enter continues, typed
    text steers the running turn, q aborts it. The resume value nodes/agent.py reads."""
    from .listing import section
    from .plan import render_plan

    section("paused", str((value or {}).get("reason") or "esc"))
    plan = (value or {}).get("plan") or []
    if plan:
        render_plan(plan)
    reply = ask("[Enter] continue · type a correction to steer · q abort » ", on_interrupt="q")
    if reply.lower() in ("q", "quit", "abort", "stop"):
        return {"action": "abort"}
    if reply:
        return {"action": "steer", "text": reply}
    return {"action": "continue"}


def discard_pending_input() -> bool:
    """Whether more input was ALREADY waiting behind the line `ask` just read — the rest of a
    multi-line paste — after throwing it away. A plain line prompt reads one line; the other
    lines of a paste would otherwise answer the prompts that follow, and the last would reach
    the `»` prompt as a request. A fragment with no newline yet cannot be seen on a terminal in
    line mode, so the queue is flushed either way. False off a terminal, or where it cannot be
    polled."""
    import os
    import sys

    try:
        import select
        import termios

        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return False
        waiting = bool(select.select([fd], [], [], 0.05)[0])
        termios.tcflush(fd, termios.TCIFLUSH)
        return waiting
    except Exception:
        return False


def ask(prompt_text: str, *, on_interrupt: str = "") -> str:
    """Read a single line for an interactive command prompt (e.g. the /models picker). Tears down
    any live status bar first — input() can't run under an active Live — and returns the raw,
    stripped reply.

    Ctrl-C / Ctrl-D return `on_interrupt` — "" by default, which every y/N prompt reads as no.
    A prompt whose EMPTY reply means yes (Enter = take the default) must pass its own refusal
    here, or an interrupt would select what a bare Enter selects."""
    _live_stop()
    try:
        # markup=False: prompts carry literal brackets (e.g. "[Enter] continue") that Rich
        # would otherwise eat as style tags.
        return _console.input(f"  {prompt_text}", markup=False).strip()
    except (EOFError, KeyboardInterrupt):
        return on_interrupt

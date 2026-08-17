"""
The freeze editor — interrupt-and-correct's freeze-then-edit interaction.

Reached when the user presses Esc while the final answer is streaming: the stream has already
stopped cleanly (the buffer is static — there is deliberately NO live-text selection) and the
answer_gate interrupt hands the frozen, provenance-tagged text here. Two beats:

  1. Say what was captured — ONE dim line (`✂ frozen — 412 chars, 3 low-confidence runs`).
     Deliberately not the text: the editor below already holds it, and re-printing it here made
     the same answer pass through four different geometries between Esc and the cursor (streaming
     tail → this tail → the editor's copy → the final markdown), so it moved three times before
     the user could type. The exception is the wizard floor below, which has no editor to show
     the text — there the tail still prints, at the streaming tail's own indent and measure.
  2. Edit: with prompt_toolkit, the whole buffer opens PRE-FILLED in the same multiline editor
     the `»` prompt uses (Enter submits, Shift+Enter/Ctrl+J newline) — move the cursor, delete
     the bad span, type the correction, submit. Without it, a two-question wizard covers the
     truncate-and-append floor: cut from a fragment's last occurrence, then type the correction.

The decision returns to the answer_gate as `{"action": "resume"|"done", "text": <edited>}` —
resume continues generation from the edited text; done accepts it as the final answer. Ctrl-C /
EOF anywhere resumes unchanged (never silently aborts a turn from inside an editor). The span
diffing and the audit record are the gate's job (core/provenance.apply_edit); this module only
collects text.
"""

from ._base import (
    Padding, Text, _console, _RICH, _ACCENT, _DIM,
    _HUMAN_STYLE, _LOW_CONF_STYLE, _term_width,
)
from .listing import section
from .prompt import ask
from .statusbar import _live_start

# `_HUMAN_STYLE` / `_LOW_CONF_STYLE` come from _base: this surface, the live streaming tail and
# the final answer body must mark the same characters identically, and duplicated literals drift.

_TAIL_LINES = 8


def _tail_offset(text: str, max_lines: int = _TAIL_LINES) -> int:
    """Character offset where the displayed tail begins (the last `max_lines` lines)."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return 0
    return len("\n".join(lines[:-max_lines])) + 1


def _low_runs(text: str, confidence) -> list:
    """The low-confidence character runs to mark, or [] (absent overlay, any failure —
    the marking is additive, never the editor's problem)."""
    if not confidence:
        return []
    try:
        from core import confidence as conf

        return conf.low_runs(confidence, text)
    except Exception:
        return []


def _frozen_summary(text: str, spans: list, confidence=None) -> str:
    """The one-line description of what the freeze captured: size, where the model was unsure,
    and any corrections carried from an earlier freeze."""
    n = len(text)
    parts = [f"{n} char{'' if n == 1 else 's'}"]
    runs = _low_runs(text, confidence)
    if runs:
        parts.append(f"{len(runs)} low-confidence run{'' if len(runs) == 1 else 's'}")
    human = sum(1 for sp in spans or [] if isinstance(sp, dict) and sp.get("author") == "human")
    if human:
        parts.append(f"{human} earlier correction{'' if human == 1 else 's'}")
    return "✂ frozen — " + ", ".join(parts)


def _print_frozen_tail(text: str, spans: list, confidence=None) -> None:
    """The frozen buffer's tail, with human-authored spans styled and low-confidence runs marked
    (human cyan layers over the low-confidence style — an already-corrected region never
    re-alarms). A dim legend names the marking when any is visible.

    Rendered at the SAME 2-space indent and measure as the streaming tail and the finished answer
    (`response._BODY_WIDTH`) — being frozen is signalled by STYLE, not by a different gutter
    width. The old 4-column `│ ` gutter re-flowed the text at a third geometry on the way to the
    editor, so the answer visibly moved before the user could touch it."""
    start = _tail_offset(text)
    n_earlier = text.count("\n", 0, start)
    if _RICH:
        from .response import _BODY_WIDTH

        width = min(_term_width(), _BODY_WIDTH)
        if n_earlier:
            _console.print(Text(f"  (… {n_earlier} earlier line{'s' if n_earlier != 1 else ''})",
                                style=_DIM))
        body = Text(text[start:], style=_DIM)  # dim IS the frozen signal
        marked = False
        for s, e in _low_runs(text, confidence):
            s, e = max(s, start), min(e, len(text))
            if e > s:
                body.stylize(_LOW_CONF_STYLE, s - start, e - start)
                marked = True
        for sp in spans or []:  # after: the later stylize wins, human cyan on top
            if not isinstance(sp, dict) or sp.get("author") != "human":
                continue
            s, e = max(int(sp.get("start", 0)), start), int(sp.get("end", 0))
            if e > s:
                body.stylize(_HUMAN_STYLE, s - start, e - start)
        _console.print(Padding(body, (0, 0, 0, 2)), width=width)
        if marked:
            _console.print(Text("  (marked = the model's own low-confidence runs — "
                                "the likeliest places to check)", style=_DIM))
    else:
        if n_earlier:
            print(f"  (… {n_earlier} earlier line{'s' if n_earlier != 1 else ''})")
        for ln in text[start:].split("\n"):
            print(f"  {ln}")


def _print_frozen(text: str, spans: list, confidence=None, *, show_tail: bool = False) -> None:
    """What the freeze prints before handing over to the editor.

    By default just the one-line summary. The buffer used to be re-printed here in full — which
    made the freeze render the same answer FOUR times in four geometries between Esc and the
    cursor (streaming tail → this gutter-rendered tail → the editor's pre-filled copy → the final
    markdown), so the text moved three times before the user could type a character. The editor
    below pre-fills the whole buffer and is the single presentation.

    `show_tail=True` restores the body for the one case that needs it: the no-prompt_toolkit
    wizard floor, where nothing else ever shows the user the text they are cutting from."""
    line = _frozen_summary(text, spans, confidence)
    if _RICH:
        _console.print(Text(f"  {line}", style=_DIM))
    else:
        print(f"  {line}")
    if show_tail:
        _print_frozen_tail(text, spans, confidence)


def _prompt_module():
    """The real `tui.ui.prompt` MODULE. importlib, not `from . import prompt`: the package
    __init__ re-exports the prompt() FUNCTION under the same name, which shadows the module on
    attribute lookup — the import system's module registry is the only unambiguous way to it."""
    import importlib

    return importlib.import_module(".prompt", __package__)


def _inline_available() -> bool:
    """Whether the pre-filled editor will open. When it will, it IS the presentation of the
    frozen text and printing a tail above it only makes the answer move again; when it won't, the
    wizard floor needs the tail because nothing else ever shows the user what they are cutting
    from. Any failure reads as "not available" — showing the text redundantly is the safe miss."""
    try:
        return bool(_prompt_module()._PTK)
    except Exception:
        return False


def _edit_inline(text: str) -> "str | None":
    """The prompt_toolkit path: the whole buffer pre-filled in the same multiline editor as the
    `»` prompt (shared key bindings — Enter submits, Shift+Enter/Ctrl+J insert a newline). None
    when prompt_toolkit isn't available or the edit was cancelled — the caller falls back."""
    _p = _prompt_module()

    if not _p._PTK:
        return None
    try:
        from prompt_toolkit import PromptSession

        session = PromptSession(input=_p._make_ptk_input())
        edited = session.prompt(
            # Indented to the app's 2-space rhythm: at column 0 the editor's first line sat two
            # columns left of every other line on screen, so the pre-filled text appeared to
            # shift the moment the editor opened.
            [("class:prompt", "  ✎ ")],
            default=text,
            multiline=True,
            key_bindings=_p._PTK_KB,
            style=_p._PTK_STYLE,
            prompt_continuation=_p._ptk_continuation,
        )
        return _p._expand_paste_tags(edited)
    except (KeyboardInterrupt, EOFError):
        return text  # cancelled: resume unchanged, never lose the buffer
    except Exception:
        return None  # editor unavailable — the wizard below still works


def _edit_wizard(text: str) -> str:
    """The no-prompt_toolkit floor: truncate-and-append. Question one locates the cut (from the
    LAST occurrence of a typed fragment to the end — the hallucination being corrected is almost
    always the tail); question two types the correction appended at the cut."""
    frag = ask("cut from (a fragment of the text; deleted from its LAST occurrence to the end; "
               "Enter = keep everything) » ")
    if frag:
        i = text.rfind(frag)
        if i == -1:
            _console.print(Text("  (fragment not found — nothing cut)", style=_DIM)) if _RICH \
                else print("  (fragment not found — nothing cut)")
        else:
            text = text[:i]
    typed = ask("your correction (appended where the cut was made; Enter = none) » ")
    if typed:
        text = text + typed
    return text


def edit_answer(value: dict) -> dict:
    """The freeze editor: show the frozen tail, collect the edit, ask resume-or-done. Returns
    the answer_gate resume value `{"action": "resume"|"done", "text": <full edited text>}`."""
    text = str(value.get("text") or "")
    spans = value.get("spans") or []

    # The editor, when available, IS the presentation of the frozen text — so print only the
    # one-line summary above it. The wizard floor has no editor, so it gets the tail.
    inline = _inline_available()
    section("answer frozen", "edit the text, then resume — the model continues from exactly "
                             "what you leave")
    _print_frozen(text, spans, value.get("confidence"), show_tail=not inline)
    if _RICH:
        _console.print()
    else:
        print()

    edited = _edit_inline(text)
    if edited is None:
        edited = _edit_wizard(text)

    what = "your edit" if edited != text else "here (unchanged)"
    resp = ask(f"resume generation from {what}? [Y]es / [d]one — accept as the final answer  "
               f"(Enter = resume) » ").lower()
    action = "done" if resp.startswith("d") else "resume"
    # Re-pin the status bar, exactly as the other blocking editors do on the way out
    # (approval.ask_approval, plan.review_plan). The graph now resumes and the model re-primes
    # its context before the first continued token arrives — seconds of a completely static
    # screen otherwise, right after the most interactive moment in the product. The bar was
    # already down (ResponseStream._begin stopped it when the answer started streaming), and
    # response.ResponseStream._reopen / .finish stop it again before touching the screen, so only
    # one Live is ever active.
    _live_start()
    return {"action": action, "text": edited}

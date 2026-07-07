"""
The freeze editor — interrupt-and-correct's freeze-then-edit interaction.

Reached when the user presses Esc while the final answer is streaming: the stream has already
stopped cleanly (the buffer is static — there is deliberately NO live-text selection) and the
answer_gate interrupt hands the frozen, provenance-tagged text here. Two beats:

  1. Show the frozen tail (human-authored spans already marked, if this is a second freeze).
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

from ._base import Text, _console, _RICH, _ACCENT, _DIM, _term_width
from .listing import section
from .prompt import ask

# Human-authored characters render in the answer's correction style everywhere (here, the final
# answer body, the rail echo): cyan is the palette's "the human is acting" color.
_HUMAN_STYLE = "bold cyan underline"

_TAIL_LINES = 8


def _tail_offset(text: str, max_lines: int = _TAIL_LINES) -> int:
    """Character offset where the displayed tail begins (the last `max_lines` lines)."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return 0
    return len("\n".join(lines[:-max_lines])) + 1


def _print_frozen(text: str, spans: list) -> None:
    """The frozen buffer's tail under a dim rail, with human-authored spans styled — the static
    text the user is about to edit. Plain path prints the tail unstyled."""
    start = _tail_offset(text)
    n_earlier = text.count("\n", 0, start)
    if _RICH:
        if n_earlier:
            _console.print(Text(f"  │ (… {n_earlier} earlier line{'s' if n_earlier != 1 else ''})",
                                style=_DIM))
        human = sorted(
            (max(int(s.get("start", 0)), start), int(s.get("end", 0)))
            for s in spans or []
            if s.get("author") == "human" and int(s.get("end", 0)) > start
        )
        pos = start
        body = Text()
        for s, e in human:
            body.append(text[pos:s])
            body.append(text[s:e], style=_HUMAN_STYLE)
            pos = max(pos, e)
        body.append(text[pos:])
        for ln in body.split("\n"):
            row = Text("  │ ", style=_DIM)
            row.append_text(ln)
            _console.print(row)
    else:
        for ln in text[start:].split("\n"):
            print(f"  | {ln}")


def _edit_inline(text: str) -> "str | None":
    """The prompt_toolkit path: the whole buffer pre-filled in the same multiline editor as the
    `»` prompt (shared key bindings — Enter submits, Shift+Enter/Ctrl+J insert a newline). None
    when prompt_toolkit isn't available or the edit was cancelled — the caller falls back."""
    # importlib, not `from . import prompt`: the package __init__ re-exports the prompt()
    # FUNCTION under the same name, which shadows the module on attribute lookup — the
    # import system's module registry is the only unambiguous way to the module itself.
    import importlib

    _p = importlib.import_module(".prompt", __package__)

    if not _p._PTK:
        return None
    try:
        from prompt_toolkit import PromptSession

        session = PromptSession(input=_p._make_ptk_input())
        edited = session.prompt(
            [("class:prompt", "✎ ")],
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

    section("answer frozen", "edit the text, then resume — the model continues from exactly "
                             "what you leave")
    _print_frozen(text, spans)
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
    return {"action": action, "text": edited}

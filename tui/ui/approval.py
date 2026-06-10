"""
The approval gate — the one place that gets to shout. A heavy rule + risk-colored tier breaks it out
of the dim trace rail; a pending write_file renders a colored unified diff of what it will change
(gotcha #2: write_file overwrites by default), and a pending run_shell shows its full command. The
human approving the exact diff/command — not a path jail — is the safety boundary.
"""

import textwrap
import time

from . import _base
from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _RISK, _RISK_HINT,
    _term_width, _truncate,
)
from .statusbar import _live_start, _live_stop


# Cap the diff preview so a huge rewrite can't flood the gate; the agent still sees the full
# content, this is just the human-facing safety preview.
_MAX_DIFF_LINES = 60


def _workspace_old_text(file_path: str) -> "tuple[str, bool]":
    """Current contents of a workspace file (for the write_file diff preview) + whether it exists.
    Resolved exactly like the write_file tool (sandboxed to the workspace), so the preview matches
    what the write will actually touch. Any failure degrades to ('', False) — the preview is
    best-effort and must never block the gate."""
    try:
        from config import get_config

        workspace = get_config().path("workspace")
        target = (workspace / file_path).resolve()
        if not target.is_relative_to(workspace) or not target.exists():
            return "", False
        return target.read_text(encoding="utf-8", errors="replace"), True
    except Exception:
        return "", False


def _unified_rows(old: str, new: str) -> "tuple[list, int]":
    """Unified-diff rows between two texts: ([(kind, text), ...], hidden_count) with kind ∈
    {add, del, hunk, ctx}, capped at _MAX_DIFF_LINES."""
    import difflib

    rows: list = []
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2))
    for line in diff[2:]:  # skip the two file-name headers positionally (content may start with +++/---)
        if line.startswith("@@"):
            rows.append(("hunk", line))
        elif line.startswith("+"):
            rows.append(("add", line[1:]))
        elif line.startswith("-"):
            rows.append(("del", line[1:]))
        else:
            rows.append(("ctx", line[1:] if line.startswith(" ") else line))
    hidden = max(0, len(rows) - _MAX_DIFF_LINES)
    return rows[:_MAX_DIFF_LINES], hidden


def _diff_lines(file_path: str, content: str, overwrite: bool) -> "tuple[list, bool, int]":
    """Diff rows for a pending write_file: (rows, is_new_file, hidden_count). An append
    (overwrite=False) diffs old-vs-(old+content) so the appended text reads as additions."""
    old, existed = _workspace_old_text(file_path)
    new = content if overwrite else (old + content)
    rows, hidden = _unified_rows(old, new)
    return rows, not existed, hidden


_DIFF_STYLE = {"add": "green", "del": "red", "hunk": _ACCENT, "ctx": _DIM}
_DIFF_SIGN = {"add": "+", "del": "-", "hunk": "", "ctx": " "}


def _render_diff_rows(mode: str, file_path: str, rows: list, hidden: int) -> None:
    """Print pre-built diff rows inside the approval frame (rich or plain). Shared by the
    write_file and edit_file previews — the diff IS the safety surface for both."""
    if _RICH:
        head = Text()
        head.append("  ┃ ", style="bold")
        head.append(f"    ↳ diff ({mode}) ", style=_DIM)
        head.append(file_path, style="default")
        _console.print(head)
        if not rows:
            empty = Text()
            empty.append("  ┃ ", style="bold")
            empty.append("        (no textual change)", style=_DIM)
            _console.print(empty)
        width = max(20, _term_width() - 12)  # loop-invariant — compute once
        for kind, text in rows:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append(f"      {_DIFF_SIGN[kind]} ", style=_DIFF_STYLE[kind])
            row.append(_truncate(text, width), style=_DIFF_STYLE[kind])
            _console.print(row)
        if hidden:
            more = Text()
            more.append("  ┃ ", style="bold")
            more.append(f"        … {hidden} more diff line(s)", style=_DIM)
            _console.print(more)
    else:
        print(f"  ┃     -> diff ({mode}) {file_path}")
        for kind, text in rows:
            print(f"  ┃       {_DIFF_SIGN[kind]} {text}")
        if hidden:
            print(f"  ┃        … {hidden} more diff line(s)")


def _render_write_diff(args: dict) -> None:
    """Render the colored unified diff for a pending write_file inside the approval frame, so the
    user sees exactly what changes before approving an overwrite (write_file overwrites by default —
    see gotcha #2). An append (overwrite=False) diffs old-vs-(old+content) so the appended text
    reads as additions. Falls back to a plain +/- listing without rich."""
    file_path = str(args.get("file_path", ""))
    content = str(args.get("content", ""))
    overwrite = bool(args.get("overwrite", True))

    rows, is_new, hidden = _diff_lines(file_path, content, overwrite)
    mode = "new file" if is_new else ("overwrite" if overwrite else "append")
    _render_diff_rows(mode, file_path, rows, hidden)


def _render_edit_diff(args: dict) -> None:
    """Render a pending edit_file as the unified diff it will produce, computed exactly the way
    the tool computes it (count + unique/replace_all rules). When the edit would fail (missing
    file, no match, ambiguous match) the preview says so — the user is about to approve a no-op."""
    file_path = str(args.get("file_path", ""))
    old_string = str(args.get("old_string", ""))
    new_string = str(args.get("new_string", ""))
    replace_all = bool(args.get("replace_all", False))

    old, existed = _workspace_old_text(file_path)
    note = None
    rows: list = []
    hidden = 0
    if not existed:
        note = "file does not exist — this edit will fail"
    else:
        count = old.count(old_string) if old_string else 0
        if count == 0:
            note = "old_string not found in the file — this edit will fail"
        elif count > 1 and not replace_all:
            note = f"old_string matches {count} places without replace_all — this edit will fail"
        else:
            new = old.replace(old_string, new_string) if replace_all else old.replace(
                old_string, new_string, 1
            )
            rows, hidden = _unified_rows(old, new)
    _render_diff_rows("edit", file_path, rows, hidden)
    if note:
        if _RICH:
            warn = Text()
            warn.append("  ┃ ", style="bold")
            warn.append(f"        ⚠ {note}", style="yellow")
            _console.print(warn)
        else:
            print(f"  ┃        ! {note}")


def _render_shell_command(args: dict) -> None:
    """Render a pending run_shell call's full command inside the approval frame. run_shell is
    `destructive` and the command is the entire safety surface, so — like write_file's diff — it is
    shown in full (wrapped, not truncated to the 80-char arg repr that would hide the tail of a long
    one-liner)."""
    command = str(args.get("command", ""))
    lines = command.splitlines() or [""]
    tip = "tip: /allow <prefix> auto-approves trusted commands like `git status`"
    if _RICH:
        head = Text()
        head.append("  ┃ ", style="bold")
        head.append("    ↳ command", style=_DIM)
        _console.print(head)
        width = max(20, _term_width() - 12)
        for line in lines:
            # Hard-wrap each logical line so nothing runs off the frame or gets silently clipped.
            for chunk in textwrap.wrap(line, width) or [""]:
                row = Text()
                row.append("  ┃ ", style="bold")
                row.append("      $ ", style=_DIM)
                row.append(chunk, style="default")
                _console.print(row)
        trow = Text()
        trow.append("  ┃ ", style="bold")
        trow.append(f"      {tip}", style=_DIM)
        _console.print(trow)
    else:
        print("  ┃     -> command")
        for line in lines:
            print(f"  ┃       $ {line}")
        print(f"  ┃       {tip}")


def _always_allow(tool_calls: list) -> None:
    """The `a(lways)` answer: drop each gated tool to the auto-approved tier for THIS session —
    exactly what `/risk <tool> read_only` does, without making the user know the command. Session-
    only by design (registry.TOOL_RISK is in-memory); the declared tiers return on restart."""
    import registry

    names = sorted({tc.get("name", "") for tc in tool_calls if tc.get("name")})
    for name in names:
        registry.TOOL_RISK[name] = "read_only"
    listing = ", ".join(names)
    msg = (
        f"  always-allowing this session: {listing}  "
        "(undo: /risk <tool> <tier> · persist: /risk <tool> read_only --save)"
    )
    if _RICH:
        _console.print(Text(msg, style=_DIM))
    else:
        print(msg)


def _select_calls(tool_calls: list, ask) -> "bool | dict":
    """Per-call decisions (`s(elect)`): ask y/N for each gated call. Collapses to True/False when
    the answers were unanimous; otherwise returns the partial-approval dict the gate understands."""
    approved = []
    for tc in tool_calls:
        r = ask(f"      allow {tc.get('name')}? y/N » ").strip().lower()
        if r in ("y", "yes"):
            approved.append(tc.get("id"))
    if len(approved) == len(tool_calls):
        return True
    if not approved:
        return False
    return {"approved_ids": approved}


def _resolve_decision(resp: str, tool_calls: list, ask) -> "bool | dict":
    """Map the gate prompt's answer to the approval node's resume value. Default (anything
    unrecognized) is reject — the gate must fail closed."""
    if resp in ("y", "yes"):
        return True
    if resp in ("a", "always"):
        _always_allow(tool_calls)
        return True
    if resp in ("s", "select", "sel"):
        return _select_calls(tool_calls, ask)
    return False


def ask_approval(value: dict) -> "bool | dict":
    """Compact, high-signal gate. Heavy rule + risk-colored tier so it breaks out of the dim
    trace rail. A write_file call additionally renders a colored unified diff of what it will
    change (gotcha #2: write_file overwrites by default). Answers: `y` approves the batch, `n`
    (default) rejects it, `s` decides per call, `a` approves AND auto-approves these tools for
    the rest of the session. Returns True/False or {"approved_ids": [...]} for a partial batch."""
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    _live_stop()  # the gate blocks on input(); the bar can't be live while it does

    def arg_repr(v) -> str:
        r = repr(v)
        return r if len(r) <= 80 else r[:79] + "…"

    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("approval required", style=f"bold {_ACCENT}")
        top.append(" " + "━" * 36, style="bold")
        _console.print(top)
        for tc in tool_calls:
            risk = str(tc.get("risk", "destructive"))
            risk_style = _RISK.get(risk, "bold red")
            head = Text()
            head.append("  ┃ ", style="bold")
            head.append(f"{risk:<14} ", style=risk_style)  # tier chip, risk-colored
            head.append(f"{tc.get('name')}", style="default")
            _console.print(head)
            is_write = tc.get("name") == "write_file"
            is_edit = tc.get("name") == "edit_file"
            is_shell = tc.get("name") == "run_shell"
            for k, v in (tc.get("args") or {}).items():  # one line per argument — full clarity
                # For write_file the `content` arg is shown as a diff below, for edit_file the
                # old/new strings are shown as a diff, and for run_shell the `command` is shown in
                # full below — not as a truncated repr. In each case that arg IS the safety
                # surface, so the 80-char repr would hide the part that matters.
                if is_write and k == "content":
                    continue
                if is_edit and k in ("old_string", "new_string"):
                    continue
                if is_shell and k == "command":
                    continue
                arow = Text()
                arow.append("  ┃ ", style="bold")
                arow.append(f"    {k} = ", style=_DIM)
                arow.append(arg_repr(v), style="default")
                _console.print(arow)
            if is_write:
                _render_write_diff(tc.get("args") or {})
            if is_edit:
                _render_edit_diff(tc.get("args") or {})
            if is_shell:
                _render_shell_command(tc.get("args") or {})
            hint = _RISK_HINT.get(risk)
            if hint:
                hrow = Text()
                hrow.append("  ┃ ", style="bold")
                hrow.append(f"    ↳ {hint}", style=risk_style)
                _console.print(hrow)
        resp = _console.input(
            "  [bold]┗━[/] approve? [bold]y[/]es / [bold]N[/]o / [bold]s[/]elect / [bold]a[/]lways » "
        ).strip().lower()
        decision = _resolve_decision(resp, tool_calls, _console.input)
    else:
        print("  ┏━ approval required " + "━" * 30)
        for tc in tool_calls:
            print(f"  ┃ [{tc.get('risk')}] {tc.get('name')}")
            is_write = tc.get("name") == "write_file"
            is_edit = tc.get("name") == "edit_file"
            is_shell = tc.get("name") == "run_shell"
            for k, v in (tc.get("args") or {}).items():
                if is_write and k == "content":
                    continue
                if is_edit and k in ("old_string", "new_string"):
                    continue
                if is_shell and k == "command":
                    continue
                print(f"  ┃     {k} = {arg_repr(v)}")
            if is_write:
                _render_write_diff(tc.get("args") or {})
            if is_edit:
                _render_edit_diff(tc.get("args") or {})
            if is_shell:
                _render_shell_command(tc.get("args") or {})
            hint = _RISK_HINT.get(str(tc.get("risk")))
            if hint:
                print(f"  ┃     -> {hint}")
        resp = input("  ┗━ approve? [y]es / [N]o / [s]elect / [a]lways » ").strip().lower()
        decision = _resolve_decision(resp, tool_calls, input)

    _base._t_last = time.perf_counter()  # don't bill the human's decision time to the next node
    _live_start()  # the turn continues (tools -> agent -> …); re-pin the bar
    return decision

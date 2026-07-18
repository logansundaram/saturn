"""
The approval gate — the one place that gets to shout. A heavy rule + risk-colored tier breaks it out
of the dim trace rail; a pending write_file renders a colored unified diff of what it will change
(gotcha #2: write_file overwrites by default), and a pending run_shell shows its full command. The
human approving the exact diff/command — not a path jail — is the safety boundary.
"""

import textwrap
import time

from textutil import head_tail

from . import _base
from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _RISK, _RISK_HINT,
    _fmt_args, _term_width, _truncate,
)
from .statusbar import _live_start, _live_stop


# Cap the diff preview so a huge rewrite can't flood the gate; the agent still sees the full
# content, this is just the human-facing safety preview.
_MAX_DIFF_LINES = 60


# ── first-gate teaching preamble ──────────────────────────────────────────────
# Two dim lines before the first gate a user ever sees, so the prompt reads as a feature, not an
# error. Once per install via receipt.take_hint — THE one-shot discovery-hint mechanism (a
# `database/.hint_<name>` sentinel, so deleting the database resets discovery along with
# first-run; fails safe to at-most-once-per-session when the sentinel can't be read or written).
_GATE_PREAMBLE = (
    "This is the approval gate — Saturn never acts without you.",
    "Nothing below runs unless you approve it; e explains why it is asking.",
)


def _preamble_due() -> bool:
    """Whether the first-gate preamble should print, marking it shown as a side effect (one call
    decides AND records, so a failure between the two can't double-print). Delegates to
    receipt.take_hint — the one sentinel mechanism, not a second copy of it (receipt is a leaf;
    no cycle from the TUI)."""
    from trust import receipt

    return receipt.take_hint("gate_seen")


def _show_preamble_if_due() -> None:
    if not _preamble_due():
        return
    for line in _GATE_PREAMBLE:
        if _RICH:
            _console.print(Text(f"  {line}", style=_DIM))
        else:
            print(f"  {line}")


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


def _wrap_exact(line: str, width: int) -> "list[str]":
    """Byte-faithful hard wrap of one logical line: plain width-slicing, NO whitespace mutation.
    textwrap.wrap would rewrite the very characters the human is approving — tabs become spaces,
    runs of spaces collapse at wrap boundaries, continuation indentation drops, a whitespace-only
    line vanishes. The arguments (and the shell command, and the HTTP request) ARE the safety
    surface, so the only transformation allowed is the line break itself; joining the chunks
    reproduces the input exactly. An empty line renders as itself, never as nothing."""
    if not line:
        return [""]
    return [line[i:i + width] for i in range(0, len(line), width)]


def _frame_wrapped(lines: "list[str]", prefix: str, width: "int | None" = None) -> None:
    """Print logical lines inside the approval frame, hard-wrapped byte-faithfully via
    `_wrap_exact` — no whitespace rewriting: tabs, space runs, and indentation reach the human
    exactly as the tool would receive them (textwrap.wrap would rewrite the safety surface
    itself). THE one home of the framed wrap loop: the shell-command, http-request, and
    full-width-arg views all render their bodies through it, so the contract can't drift between
    them. The plain fallback prints each logical line unwrapped (the terminal wraps;
    byte-faithfulness still holds). `width` lets a caller hoist the _term_width() read out of a
    per-key loop (loop-invariant — compute once)."""
    if width is None:
        width = max(20, _term_width() - 12)
    if _RICH:
        for line in lines:
            for chunk in _wrap_exact(line, width):
                row = Text()
                row.append("  ┃ ", style="bold")
                row.append(f"      {prefix}", style=_DIM)
                row.append(chunk, style="default")
                _console.print(row)
    else:
        for line in lines:
            print(f"  ┃       {prefix}{line}")


def _render_shell_command(args: dict) -> None:
    """Render a pending run_shell call's full command inside the approval frame. run_shell is
    `destructive` and the command is the entire safety surface, so — like write_file's diff — it is
    shown in full (wrapped, not truncated to the 80-char arg repr that would hide the tail of a long
    one-liner)."""
    command = str(args.get("command", ""))
    lines = command.splitlines() or [""]
    tip = "tip: /policy allow <prefix> auto-approves trusted commands like `git status`"
    if _RICH:
        head = Text()
        head.append("  ┃ ", style="bold")
        head.append("    ↳ command", style=_DIM)
        _console.print(head)
    else:
        print("  ┃     -> command")
    _frame_wrapped(lines, "$ ")
    # The tip stays out of _frame_wrapped on purpose: it's advice, not approved bytes, so it
    # keeps the renderer's soft handling rather than the byte-faithful hard wrap.
    if _RICH:
        trow = Text()
        trow.append("  ┃ ", style="bold")
        trow.append(f"      {tip}", style=_DIM)
        _console.print(trow)
    else:
        print(f"  ┃       {tip}")


# Tools with a bespoke full-surface renderer above. ONE table is the single source: which
# argument keys the compact repr must SKIP (that arg IS the safety surface, shown in full by the
# bespoke view) and which renderer draws it. Adding a bespoke-rendered tool is exactly one entry
# here — the membership tuple below derives from it, so the skip keys, the renderer dispatch, and
# _full_width_args can never drift apart. (http_request's full-request renderer left with the
# tool, 2026-07-16 — MCP tools render through the generic full-width view.)
_BESPOKE = {
    "write_file": (("content",), _render_write_diff),
    "edit_file": (("old_string", "new_string"), _render_edit_diff),
    "run_shell": (("command",), _render_shell_command),
}

# Derived view for the branch test: their decisive argument is already shown in full above, so
# the generic full-width view would duplicate it.
_BESPOKE_RENDERED = tuple(_BESPOKE)

# Per-value cap for the full-width argument view: big enough to read a whole API payload, small
# enough that one fat value can't flood the gate.
_MAX_ARG_VALUE = 2000


def _full_width_args(name, risk: str) -> bool:
    """Whether a gated call renders each argument full-width (wrapped + head/tail-clamped) instead
    of the compact 80-char repr. Every side_effecting/destructive call WITHOUT a bespoke renderer
    qualifies — notably all mcp_* tools, which fail closed to destructive precisely because
    they're untrusted, so hiding the tail of their arguments behind a truncated repr is exactly
    backwards. read_only calls (reaching the gate only via a quarantine escalation) keep
    the compact form."""
    return risk in ("side_effecting", "destructive") and name not in _BESPOKE_RENDERED


def _clamp_value(text: str, cap: int = _MAX_ARG_VALUE) -> str:
    """Bound one argument value for the full-width view: textutil.head_tail at gate scale (head +
    tail with an explicit elision marker) — the start carries the intent, the tail is where a long
    payload hides the part that matters, so neither is silently cut. Delegates to THE one home of
    the head+tail idiom, never a third hand-rolled copy."""
    return head_tail(text, cap)


def _render_full_args(args: dict) -> None:
    """Render every argument of a gated call full-width inside the approval frame — hard-wrapped
    like the run_shell command view, never the 80-char repr. For a tool with no bespoke safety
    surface the arguments ARE the safety surface."""
    width = max(20, _term_width() - 12)  # loop-invariant — compute once, pass through
    for k, v in (args or {}).items():
        value = v if isinstance(v, str) else repr(v)
        lines = _clamp_value(value).splitlines() or [""]
        if _RICH:
            head = Text()
            head.append("  ┃ ", style="bold")
            head.append(f"    {k} =", style=_DIM)
            _console.print(head)
        else:
            print(f"  ┃     {k} =")
        _frame_wrapped(lines, "", width=width)


def _frame_note(text: str, style: str = "yellow") -> None:
    """One wrapped annotation line(s) inside the approval frame (`  ┃     <text>`), shared by the
    secret-scan warning, the quarantine banner, and the explain view."""
    width = max(20, _term_width() - 12)
    for i, chunk in enumerate(textwrap.wrap(text, width) or [""]):
        if _RICH:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append(("    " if i == 0 else "      ") + chunk, style=style)
            _console.print(row)
        else:
            print(f"  ┃     {chunk}" if i == 0 else f"  ┃       {chunk}")


def _render_secret_warnings(args: dict) -> None:
    """Warn when a gated call's arguments carry a secret-like value (an API key in an MCP call's
    args, a token inline in a run_shell command): approving the call sends the secret wherever the
    call goes. Reuses the redaction scanner; emails are excluded here (common, legitimate argument
    content — this warning is about credentials). Best-effort: a scan failure never blocks the
    gate."""
    try:
        from trust import redaction

        findings = [f for f in redaction.scan_args(args) if f.kind != "email"]
    except Exception:
        return
    if not findings:
        return
    labels: list[str] = []
    for f in findings:
        label = f"{f.kind} ({f.preview})"
        if label not in labels:
            labels.append(label)
    shown = ", ".join(labels[:3]) + (f" +{len(labels) - 3} more" if len(labels) > 3 else "")
    _frame_note(f"⚠ argument carries a secret-like value: {shown} — approving sends it "
                "wherever this call goes")


def _arg_repr(v) -> str:
    """Compact one-line value form for the non-bespoke argument rows."""
    return _truncate(repr(v), 80)


def _render_call(tc: dict) -> None:
    """Render one gated call inside the approval frame — the per-call body shared by the rich
    and plain prompts (the rendering IS the safety surface, so the two paths must never drift):
    the risk-tier head line, the argument view (full-width when the call has no bespoke renderer,
    compact 80-char repr otherwise — minus the keys the bespoke view shows in full), the bespoke
    safety surface itself (diff / command, from the one _BESPOKE table), the secret warning, and
    the per-tier hint."""
    risk = str(tc.get("risk", "destructive"))
    risk_style = _RISK.get(risk, "bold red")
    name = tc.get("name")
    args = tc.get("args") or {}
    skip_keys, bespoke = _BESPOKE.get(name, ((), None))

    if _RICH:
        head = Text()
        head.append("  ┃ ", style="bold")
        head.append(f"{risk:<14} ", style=risk_style)  # tier chip, risk-colored
        head.append(f"{name}", style="default")
        _console.print(head)
    else:
        print(f"  ┃ [{risk}] {name}")

    if _full_width_args(name, risk):
        _render_full_args(args)
    else:
        for k, v in args.items():  # one line per argument — full clarity
            # The keys a bespoke renderer shows in full below (write_file's content as a diff,
            # edit_file's old/new strings, run_shell's command) are skipped here: that arg IS
            # the safety surface, so the 80-char repr would hide the part that matters.
            if k in skip_keys:
                continue
            if _RICH:
                arow = Text()
                arow.append("  ┃ ", style="bold")
                arow.append(f"    {k} = ", style=_DIM)
                arow.append(_arg_repr(v), style="default")
                _console.print(arow)
            else:
                print(f"  ┃     {k} = {_arg_repr(v)}")

    if bespoke:
        bespoke(args)
    _render_secret_warnings(args)
    hint = _RISK_HINT.get(risk)
    if hint:
        if _RICH:
            hrow = Text()
            hrow.append("  ┃ ", style="bold")
            hrow.append(f"    ↳ {hint}", style=risk_style)
            _console.print(hrow)
        else:
            print(f"  ┃     -> {hint}")


def _render_quarantine_banner(value: dict) -> None:
    """When the batch follows a tool result flagged for embedded instructions, say so up front —
    the calls below may have been steered by injected content, so the human should check the
    arguments are what THEY intended, not what a web page asked for."""
    q = value.get("quarantine") if isinstance(value, dict) else None
    flags = (q or {}).get("flags") or []
    if not flags:
        return
    detail = "; ".join(f"{f.get('tool')}: {', '.join(f.get('kinds') or [])}" for f in flags[:3])
    _frame_note("⚠ injection quarantine: earlier tool output this turn contained "
                f"instruction-like content ({detail}) — approve only if these calls are what "
                "YOU intended")


def _render_explain(value: dict) -> None:
    """The `e(xplain)` answer: WHY the agent wants this batch — the plan step it is fulfilling and
    its recorded pre-action reasoning (the same provenance /trace why reconstructs afterward,
    shown at the moment of decision). Opens with the one full key legend: the first-gate preamble
    teaches `e`, so this is where `s` and `a` — `a` permanently widens the gate, which nobody
    should discover by experiment — become discoverable."""
    _frame_note(f"keys: {_KEY_LEGEND_FULL}", style=_DIM)
    step = value.get("step") if isinstance(value, dict) else None
    reasoning = " ".join(str((value or {}).get("reasoning") or "").split())
    shown = False
    if isinstance(step, dict) and step.get("label"):
        tool = f"  [{step['intended_tool']}]" if step.get("intended_tool") else ""
        _frame_note(f"↳ plan step {step.get('step_id')}: {step.get('label')}{tool}", style=_DIM)
        shown = True
    if reasoning:
        _frame_note(f"↳ reasoning: {_truncate(reasoning, 600)}", style=_DIM)
        shown = True
    if not shown:
        _frame_note("↳ no plan step or recorded reasoning for this batch — the model chose the "
                    "call directly (full provenance after the turn: /trace why)", style=_DIM)


def _grant_note(msg: str) -> None:
    """Disclosure line for an always-grant — yellow, not dim: widening the gate is exactly the
    line the user must not skim past."""
    if _RICH:
        _console.print(Text(f"  {msg}", style="yellow"))
    else:
        print(f"  {msg}")


def _propose_shell_prefix(command: str) -> "str | None":
    """The /policy allow-style prefix to offer when `a(lways)` covers a run_shell call: the FULL
    command (whitespace-normalized) — the narrowest grant that covers exactly what the human just
    reviewed. Proposing the bare leading token would let one confirmation un-gate every future
    `git …` (including `git push --force`) — exactly the broad grant /policy allow's own help
    warns against; a shorter prefix stays available, but only by the user TYPING it deliberately. None
    when no prefix could ever exempt this command — empty, or carrying shell metacharacters
    (policy.shell_allowed refuses those wholesale, so offering a prefix would teach a false
    rule). Asks policy's own public screen (shell_prefix_rejects), never a second copy of its
    rule."""
    from trust import policy

    if policy.shell_prefix_rejects(command):
        return None
    return " ".join(command.split()) or None


def _always_allow(tool_calls: list, ask) -> dict:
    """The `a(lways)` answer: COLLECT the grants the user confirms and return them in the
    decision dict for the approval node to apply PAST the interrupt — nothing mutates here.
    Mutating at decision time (the old shape) let LangGraph's node re-run recompute the batch
    as ungated and return at the fast path before the gate_event recording site, so the human's
    decision vanished from the record (gotcha #7: empty must always mean "never asked").

    Non-shell tools collect a session tier drop to read_only (what /policy risk <tool> read_only
    does). run_shell is the exception: read_only would un-gate EVERY future shell command from one
    keypress, so it gets a SCOPED grant instead — the FULL command offered as a /policy allow-style
    prefix (the narrowest grant covering exactly what was reviewed; a shorter/broader prefix only
    by the user typing it), validated through the one policy matcher (grant_shell_prefix
    dry_run=True — the node persists). Declining or failing validation leaves run_shell gated;
    the rest of the batch still gets its grant."""
    from trust import policy

    names = sorted({tc.get("name", "") for tc in tool_calls if tc.get("name")})
    granted = [n for n in names if n != "run_shell"]
    decision: dict = {"approved": True, "tools": granted, "shell_grants": []}
    if granted:
        listing = ", ".join(granted)
        _grant_note(f"always-allowing this session: {listing}  "
                    "(undo: /policy risk <tool> <tier> · persist: /policy risk <tool> "
                    "read_only --save)")

    if "run_shell" not in names:
        return decision
    commands = list(dict.fromkeys(
        str((tc.get("args") or {}).get("command", ""))
        for tc in tool_calls if tc.get("name") == "run_shell"
    ))
    for command in commands:
        proposal = _propose_shell_prefix(command)
        if proposal is None:
            _grant_note("run_shell: no prefix could cover this command (shell metacharacters "
                        "always face the gate) — it keeps prompting")
            continue
        resp = ask(f'      always-allow this exact command as a prefix? "{proposal}"  '
                   "y / N / or type a shorter prefix  (Enter = no) » ").strip()
        if resp.lower() in ("y", "yes"):
            chosen = proposal
        elif resp and resp.lower() not in ("n", "no"):
            chosen = resp
        else:
            _grant_note("run_shell: no prefix granted — it keeps prompting")
            continue
        ok, msg = policy.grant_shell_prefix(chosen, command, dry_run=True)
        _grant_note(msg)
        if ok:
            decision["shell_grants"].append({"prefix": chosen, "command": command})
    return decision


def _select_calls(tool_calls: list, ask) -> "bool | dict":
    """Per-call decisions (`s(elect)`): ask y/N for each gated call. Each prompt carries a clamped
    arg summary — two same-tool calls in one batch (exactly the mixed-trust case `s` exists for:
    two run_shell commands, the user wants one) would otherwise read identically, forcing the
    human to remember listing order from a frame that may have scrolled past several diffs.
    Approving the wrong twin is a real hazard at the safety gate. The summary renders literally:
    ask_approval hands this a markup-disabled input on the rich path, so bracketed argument text
    is never eaten as Rich tags. Collapses to True/False when the answers were unanimous;
    otherwise returns the partial-approval dict the gate understands."""
    approved = []
    for tc in tool_calls:
        summary = _fmt_args(tc.get("args") or {}, 48)
        r = ask(f"      allow {tc.get('name')}({summary})? y / N  (Enter = no) » ").strip().lower()
        if r in ("y", "yes"):
            approved.append(tc.get("id"))
    if len(approved) == len(tool_calls):
        return True
    if not approved:
        return False
    return {"approved_ids": approved}


# THE gate key table — one home for the answer vocabulary: display key, accepted spellings,
# terse legend label (the unrecognized-answer note), teaching legend label (the explain view).
# The resolver, the recognized-answer set, the prompt's key choices, and both legends all derive
# from it, so adding/renaming a key is exactly one row — a key the resolver accepts but a legend
# doesn't name (or vice versa) cannot exist (the same single-source pattern as _BESPOKE).
# `e`'s spellings never reach _resolve_decision — the prompt loop consumes them (explain +
# re-prompt) before resolving — but they are still recognized answers, never "treated as no".
_GATE_KEYS = (
    ("y", ("y", "yes"), "approve", "approve the batch"),
    ("N", ("", "n", "no"), "reject", "reject (default)"),
    ("s", ("s", "select", "sel"), "per call", "decide per call"),
    ("a", ("a", "always"), "always-allow",
     "always-allow these tools this session (run_shell: scoped prefix grant)"),
    ("e", ("e", "explain", "why", "?"), "explain", "this explanation"),
)
_ANSWER = {key: spellings for key, spellings, _terse, _long in _GATE_KEYS}
_KNOWN_ANSWERS = tuple(s for _key, spellings, _terse, _long in _GATE_KEYS for s in spellings)
_KEY_CHOICES = " / ".join(key for key, *_ in _GATE_KEYS)  # "y / N / s / a / e"
_KEY_LEGEND = " · ".join(f"{key} {terse}" for key, _spellings, terse, _long in _GATE_KEYS)
_KEY_LEGEND_FULL = " · ".join(f"{key} {long}" for key, _spellings, _terse, long in _GATE_KEYS)


def _unrecognized_note(resp: str) -> "str | None":
    """One-line feedback when the gate answer isn't a known key, or None for every recognized one.
    The decision itself is UNCHANGED — anything unrecognized still rejects (the documented
    fail-closed contract) — but a user typing `help` must not have their confusion silently
    recorded as a deliberate rejection: without this, the only acknowledgment is the later rail
    echo `✗ you rejected …`."""
    if resp in _KNOWN_ANSWERS:
        return None
    return f'unrecognized "{resp}" — treated as no  ({_KEY_LEGEND})'


def _resolve_decision(resp: str, tool_calls: list, ask) -> "bool | dict":
    """Map the gate prompt's answer to the approval node's resume value. Default (anything
    unrecognized) is reject — the gate must fail closed. The always-allow answer returns the
    collected-grants decision dict; the node applies the grants past the interrupt."""
    if resp in _ANSWER["y"]:
        return True
    if resp in _ANSWER["a"]:
        return _always_allow(tool_calls, ask)
    if resp in _ANSWER["s"]:
        return _select_calls(tool_calls, ask)
    return False


def ask_approval(value: dict) -> "bool | dict":
    """Compact, high-signal gate. Heavy rule + risk-colored tier so it breaks out of the dim
    trace rail. A write_file call additionally renders a colored unified diff of what it will
    change (gotcha #2: write_file overwrites by default); any other gated side_effecting/
    destructive call renders its arguments full-width (`_render_full_args`). Answers: `y`
    approves the batch, `N` (the default — bare Enter or anything unrecognized) rejects it, `s`
    decides per call, `a` approves AND auto-approves these tools for the rest of the session
    (run_shell instead gets a scoped /policy allow-style prefix grant), `e` explains WHY the agent wants
    the batch (plan step + recorded reasoning) and re-prompts. Arguments carrying secret-like
    values warn inline (redaction scanner); a batch following quarantine-flagged tool output
    opens with a banner saying so. Returns True/False or {"approved_ids": [...]} for a partial
    batch."""
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    _live_stop()  # the gate blocks on input(); the bar can't be live while it does

    # The gate lands mid-trace — one blank line so the blocking decision reads as its own moment.
    _console.print() if _RICH else print()
    _show_preamble_if_due()  # first gate ever: two lines saying what this prompt IS

    # Count the calls that faced the gate this turn — the trust receipt's `n gated` segment
    # (receipt.py) is the permanent echo of this prompt having happened.
    _base._status["gates"] = _base._status.get("gates", 0) + len(tool_calls)

    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("approval required", style=f"bold {_ACCENT}")
        _console.print(top)
    else:
        print("  ┏━ approval required")
    _render_quarantine_banner(value)
    for tc in tool_calls:
        _render_call(tc)

    # The sub-prompts (_always_allow's prefix proposal, _select_calls' arg summaries) embed RAW
    # command/argument text, and rich's Console.input parses markup AND emoji codes by default:
    # a legitimate `grep [error]` would be silently eaten as a tag, a `[/...]`-shaped token would
    # raise MarkupError mid-gate, and a `:name:` token would become an emoji glyph — in the very
    # text the human is confirming as a persistent grant. Disable both so the displayed text is
    # byte-identical to what is granted (prompt.ask does the same; the plain builtin input is
    # already literal). The main gate prompt below keeps its intentional [bold] markup.
    ask = (lambda p: _console.input(p, markup=False, emoji=False)) if _RICH else input

    # The bold capital N marks the fail-closed default: bare Enter (or anything unrecognized)
    # rejects the batch. Both prompt strings derive their key choices from the one _GATE_KEYS
    # table, like the legends.
    _choices_rich = " / ".join("[bold]N[/]" if k == "N" else k for k, *_ in _GATE_KEYS)
    while True:
        if _RICH:
            resp = _console.input(
                f"  [bold]┗━[/] approve? {_choices_rich}  (Enter = no) » "
            ).strip().lower()
        else:
            resp = input(
                f"  ┗━ approve? {_KEY_CHOICES}  (Enter = no) » "
            ).strip().lower()
        if resp in _ANSWER["e"]:
            _render_explain(value)
            continue
        note = _unrecognized_note(resp)
        if note:
            # Feedback only — the reject below stands (fail closed), but typed confusion like
            # `help` is acknowledged at the prompt instead of surfacing later as `✗ you rejected`.
            _frame_note(note, style=_DIM)
        decision = _resolve_decision(resp, tool_calls, ask)
        break

    _base._t_last = time.perf_counter()  # don't bill the human's decision time to the next node
    _live_start()  # the turn continues (tools -> agent -> …); re-pin the bar
    return decision

"""
Tool-argument recovery for the execute node (transplanted from the agentic_benchmark harness,
2026-07-03).

Small local models emit almost-right tool calls: the right tool with the wrong key ("file"
instead of "file_path"), a text-format call instead of a native one, or an empty call. Instead
of failing the step, this layer maps known aliases onto each tool's real schema, parses gemma's
text-format calls, and hands the execute node a precise schema hint for its retry.

The alias tables cover the built-in registry; a tool without a table (MCP tools) passes its
args through unchanged when they are a dict — the remote schema is the server's business.

Leaf module: imports nothing project-side, so every layer may use it.
"""

from __future__ import annotations

import re
from typing import Optional

# canonical arg -> the names models actually emit for it, per REQUIRED argument. A tool listed
# here with an empty dict has no required args (current_time). Tools absent from this table are
# unknown (MCP): their args pass through unchanged.
_ARG_ALIASES: dict[str, dict[str, list[str]]] = {
    "read_file": {
        "file_path": ["file_path", "path", "file", "filename", "filepath", "fname"],
    },
    "list_directory": {},  # directory is optional (defaults to the workspace root)
    "find_files": {
        "pattern": ["pattern", "glob", "name", "filename", "query"],
    },
    "search_files": {
        "pattern": ["pattern", "query", "q", "search", "text", "keyword", "keywords", "regex"],
    },
    "edit_file": {
        "file_path": ["file_path", "path", "file", "filename", "filepath"],
        "old_string": ["old_string", "old", "old_text", "find", "search", "target", "before"],
        "new_string": ["new_string", "new", "new_text", "replacement", "replace", "after"],
    },
    "write_file": {
        "file_path": ["file_path", "path", "file", "filename", "filepath"],
        "content": ["content", "text", "contents", "data", "body", "value", "string"],
    },
    "search_knowledge_base": {
        "query": ["query", "q", "search", "text", "question", "keywords"],
    },
    "calculate": {
        "expression": ["expression", "expr", "equation", "formula", "calc", "input"],
    },
    "current_time": {},
    "web_search": {
        "query": ["query", "q", "search", "text", "question", "keywords"],
    },
    "web_extract": {
        "url": ["url", "link", "href", "page", "address"],
    },
    "run_shell": {
        "command": ["command", "cmd", "shell", "script", "code", "bash", "powershell"],
    },
    "remember": {
        "fact": ["fact", "text", "note", "content", "memory"],
    },
    "recall": {},  # query is optional (empty returns everything)
    "ask_user": {
        "question": ["question", "prompt", "query", "q", "text", "message", "ask"],
    },
    "schedule_notification": {
        "when": ["when", "time", "at", "datetime", "date", "delay", "in", "schedule"],
        "title": ["title", "message", "text", "reminder", "content", "note", "subject"],
    },
}

# Required args for which the EMPTY STRING is a legitimate value — deleting text via
# edit_file(new_string="") or creating an empty file via write_file(content="") — so "" must
# count as present for these, not as a missing value to retry.
_EMPTY_OK: dict[str, set[str]] = {
    "edit_file": {"new_string"},
    "write_file": {"content"},
}

# Optional args passed through when present (correctly named) — never required, never invented.
_OPTIONAL: dict[str, list[str]] = {
    "list_directory": ["directory"],
    "find_files": ["directory"],
    "search_files": ["directory", "file_glob"],
    "write_file": ["overwrite"],
    "edit_file": ["replace_all"],
    "remember": ["category", "layer", "replaces", "sensitivity"],
    "recall": ["query"],
    "schedule_notification": ["body"],
}

# The exact call shape quoted back at the model when its attempt was rejected.
_SCHEMA_SHAPES: dict[str, str] = {
    "read_file": "read_file(file_path=<workspace-relative file path>)",
    "list_directory": "list_directory(directory=<workspace-relative directory, '.' for the root>)",
    "find_files": "find_files(pattern=<filename or glob like *.csv>)",
    "search_files": "search_files(pattern=<text to find inside files>)",
    "edit_file": "edit_file(file_path=<file path>, old_string=<existing text copied "
    "verbatim, appearing exactly once>, new_string=<replacement text>)",
    "write_file": "write_file(file_path=<file path>, content=<exact text to write>)",
    "search_knowledge_base": "search_knowledge_base(query=<search text>)",
    "calculate": "calculate(expression=<numeric expression, e.g. 4.25*12+9.99*7>)",
    "current_time": "current_time()",
    "web_search": "web_search(query=<web search terms>)",
    "web_extract": "web_extract(url=<the page URL>)",
    "run_shell": "run_shell(command=<shell command line>)",
    "remember": "remember(fact=<one concise statement>, layer=<user|entities|commitments|negative|"
    "agent|memo, optional>, replaces=<#id of the fact this corrects, optional>)",
    "recall": "recall(query=<filter text, or empty for everything>)",
    "ask_user": "ask_user(question=<the ONE question to ask the user>)",
    "schedule_notification": "schedule_notification(when=<future time: ISO 8601, 'in 20 minutes', "
    "'tomorrow at 09:00'>, title=<short headline>, body=<optional detail>)",
}


# ── concrete-step argument fill (2026-09-03) ──────────────────────────────────────────────────
#
# A planned step whose ONLY argument is already spelled out in its label ("Read notes.md") does
# not need a model call to produce {"file_path": "notes.md"}: the call is a copy. Measured
# before this: a "read all files" turn spent one ~3 s tool-call generation per read_file step
# (15 calls, 143 s on the 9b) to emit paths the plan had already named. Deliberately narrow —
# the fill is a copy, never a guess:
#   - current_time takes no arguments;
#   - read_file / list_directory only when the label names EXACTLY ONE path-like token and the
#     caller's `kind_of` says it is a file / a directory in the workspace (the spelling is taken
#     from the label as written, so the tool sees what the plan said);
#   - find_files only when the label carries EXACTLY ONE glob ("*.csv", "report*",
#     "**/drafts/*.md") and no other path, searched from the workspace root.
# Two paths, a placeholder ("the file the listing names"), a folder named in words ("the
# reports folder"), or a name that isn't there all fall through to the model as before.
# Everything downstream (the review revocation lock, effect authorization, the stall detector,
# the approval gate) reads the arguments and runs unchanged — the fill only replaces WHO wrote
# them.
# A slash-joined path (optionally ending in "/"), a bare name ending in "/" ("data/"), or a
# name with an extension. Tokens keep the label's spelling; a trailing "/" is dropped.
_LABEL_PATH_RE = re.compile(
    r"[A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)+/?|[A-Za-z0-9_.\-]+/|[A-Za-z0-9_\-]+\.[A-Za-z0-9]{1,8}"
)
_LABEL_GLOB_RE = re.compile(r"[A-Za-z0-9_.\-/*?\[\]]*[*?][A-Za-z0-9_.\-/*?\[\]]*")

_CONCRETE_NO_ARG_TOOLS = ("current_time",)


def _label_tokens(label, pattern: "re.Pattern") -> list:
    out: list = []
    for m in pattern.finditer(str(label or "")):
        tok = m.group(0).strip(".,;:").rstrip("/")
        if tok and tok not in out:
            out.append(tok)
    return out


def concrete_args(tool_name: str, label, kind_of) -> Optional[dict]:
    """Arguments copied from the step label when they are unambiguous, else None (generate them).
    `kind_of(rel_path)` answers "file", "dir", or None for a workspace-relative path."""
    if tool_name in _CONCRETE_NO_ARG_TOOLS:
        return {}
    if tool_name == "find_files":
        globs = _label_tokens(label, _LABEL_GLOB_RE)
        # plain paths are scanned with the globs blanked out, so "**/drafts/*.md" is one glob,
        # not a glob plus the path "/drafts"
        plain = _label_tokens(_LABEL_GLOB_RE.sub(" ", str(label or "")), _LABEL_PATH_RE)
        if len(globs) == 1 and not plain:
            return {"pattern": globs[0]}
        return None
    if tool_name not in ("read_file", "list_directory"):
        return None
    tokens = _label_tokens(label, _LABEL_PATH_RE)
    if len(tokens) != 1:
        return None
    (path,) = tokens
    try:
        kind = kind_of(path)
    except Exception:
        kind = None
    if tool_name == "read_file" and kind == "file":
        return {"file_path": path}
    if tool_name == "list_directory" and kind == "dir":
        return {"directory": path}
    return None


def parse_text_call(content: str) -> Optional[dict]:
    """Recover key/value args from a TEXT-format tool call (gemma's `key: <|"|>value<|"|>`
    dialect, or bare JSON-ish "key": "value" pairs). None when nothing parses."""
    pairs = re.findall(r'(\w+)\s*:\s*<\|"\|>(.*?)<\|"\|>', content, re.DOTALL)
    if not pairs:
        pairs = re.findall(r'"(\w+)"\s*:\s*"(.*?)"', content, re.DOTALL)
    return {k: v for k, v in pairs} if pairs else None


# --- the laundering refusal (transplanted from the engine isolate, 2026-08-15) ---------------
#
# calculate(expression="551") computes nothing: it mints TOOL provenance for a number the model
# already worked out — a value a reasoning step invented, laundered into a "computed" result that
# nothing downstream can tell from a real one. The test is decidable and needs no judgment: an
# expression with no infix operator and no function call is a bare value. Function names track
# tools/calculator._ALLOWED_FUNCS; `**` matches on `*`.

_ARITH_OP_RE = re.compile(r"[+\-*/%^]|\b(?:abs|round|min|max|pow|sum)\s*\(")


def launders_a_value(tool_name: str, args) -> bool:
    """Whether this call would manufacture provenance for a value instead of computing one."""
    if tool_name != "calculate" or not isinstance(args, dict):
        return False
    expr = str(args.get("expression") or "").strip()
    if not expr:
        return False  # an empty expression is coercion's failure, not laundering
    # A LEADING sign is part of the literal, not an operation: "-551" computes nothing. Strip
    # leading signs/brackets/space so only an INFIX operator or a function counts as arithmetic.
    return not _ARITH_OP_RE.search(expr.lstrip("+-( \t"))


def coerce_args(name: str, args) -> Optional[dict]:
    """Map emitted args onto `name`'s real schema via the alias tables. Returns the corrected
    dict, or None when a REQUIRED arg is missing under every alias (the caller retries with a
    schema hint). A tool without a table (MCP) passes a dict through unchanged."""
    if not isinstance(args, dict):
        return None
    aliases = _ARG_ALIASES.get(name)
    if aliases is None:
        return args  # unknown/remote tool: its schema is not ours to police
    lower = {k.lower(): v for k, v in args.items() if isinstance(k, str)}
    empty_ok = _EMPTY_OK.get(name, set())
    out: dict = {}
    for canon, names in aliases.items():
        missing = (None,) if canon in empty_ok else (None, "")
        val = next((lower[a] for a in names if lower.get(a) not in missing), None)
        if val is None:
            return None
        out[canon] = val
    for opt in _OPTIONAL.get(name, []):
        if opt in lower and lower[opt] is not None:
            out[opt] = lower[opt]
    return out


def schema_hint(name: str, problem: str) -> str:
    """The retry corrective appended to the context after a rejected attempt."""
    shape = _SCHEMA_SHAPES.get(name, f"{name}(<arguments matching the tool's schema>)")
    return (
        f"Your previous attempt was rejected: {problem}. "
        f"Call the tool exactly as {shape}. If this step needs the tool, call "
        f"it now with correct arguments; otherwise answer in plain text."
    )

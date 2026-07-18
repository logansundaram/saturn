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
    "remember": ["category"],
    "recall": ["query"],
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
    "remember": "remember(fact=<one concise statement>)",
    "recall": "recall(query=<filter text, or empty for everything>)",
    "ask_user": "ask_user(question=<the ONE question to ask the user>)",
}


def parse_text_call(content: str) -> Optional[dict]:
    """Recover key/value args from a TEXT-format tool call (gemma's `key: <|"|>value<|"|>`
    dialect, or bare JSON-ish "key": "value" pairs). None when nothing parses."""
    pairs = re.findall(r'(\w+)\s*:\s*<\|"\|>(.*?)<\|"\|>', content, re.DOTALL)
    if not pairs:
        pairs = re.findall(r'"(\w+)"\s*:\s*"(.*?)"', content, re.DOTALL)
    return {k: v for k, v in pairs} if pairs else None


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

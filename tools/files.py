"""
File tools — read, write, edit, list, and search files inside the folders Saturn can reach.

Every path is resolved per call through `core/workspace.resolve` — the launch folder plus any
`/add-dir` folders — so a tool call can never reach anything else. `write_file` and
`edit_file` are the mutating tools here (gated via registry.TOOL_RISK), and both snapshot the
target's turn-start state first (stores/snapshots.py) so `/undo` can reverse them.

`search_files` (content regex) and `find_files` (name glob) are the navigation primitives:
without them the agent's only way to locate something is list_directory + reading whole files,
which burns context and iterations. Both are read_only and hard-capped so a huge workspace
can't flood an observation (the tool node clamps again — gotcha #5 — but staying small at the
source keeps the useful part of the result intact).
"""

import fnmatch
import os
import re
from pathlib import Path

from textutil import truncate
from tools.toolspec import register_tool

from core import workspace as _ws
from stores.snapshots import snapshot_file


# ── cross-module observation contracts (one producer, one parser — the DECLINE_TEXT rule) ────
# The write tools report refusals as ordinary strings, so their SUCCESS wording is a contract:
# nodes/synthesize.verify_writes re-reads only files whose observation starts with one of these
# markers. Change a constant and its return statement together; never re-type the strings there.
MSG_OVERWROTE = "File overwritten successfully"
MSG_CREATED = "File created successfully"
MSG_APPENDED = "Content appended to file successfully"
EDIT_PREFIX = "Edited "  # edit_file's success line: f"{EDIT_PREFIX}{path}: replaced …"
WRITE_SUCCESS_MARKERS = (MSG_OVERWROTE, MSG_CREATED, MSG_APPENDED, EDIT_PREFIX)

# The not-found refusals nodes/rectify's dead-end classifier keys on (lowercase startswith —
# rectify lowercases observations before matching). Same producer/parser contract as above.
NOT_FOUND_PREFIXES = (
    "file not found:",           # edit_file: missing target
    "no matches for",            # search_files: empty content search
    "no files matching",         # find_files: empty name glob
    "path is not a directory",   # _resolve_dir refusal (a guessed directory that isn't one)
)


def _resolve(path: str):
    """Resolve a path the model gave against the folders Saturn can reach
    (core/workspace.resolve — the ONE containment check). Returns (root, target, error): `error`
    is the refusal string when the path is outside every reachable folder (then `target` must
    not be used), else None. Every tool below starts here, and so does the approval preview."""
    target, refusal = _ws.resolve(path)
    return _ws.root(), target, refusal


def _resolve_dir(path: str):
    """`_resolve` for tools that need an existing directory to walk."""
    root, target, error = _resolve(path)
    if error is None and not target.is_dir():
        error = "Path is not a directory."
    return root, target, error


def _not_found_text(file_path: str) -> str:
    """The read_file refusal for a path that is not a workspace file. When its basename is an
    ingested knowledge-base document, say so and name the tool — the manifest read is one
    mtime-validated stat (document_registry's memo), so this costs nothing on the hot path."""
    from stores.document_registry import manifest_entries, read_documents_manifest

    text = f"File not found in the workspace: {file_path}."
    try:
        base = Path(file_path).name.lower()
        names = {e["name"].lower() for e in manifest_entries(read_documents_manifest())}
    except Exception:  # a broken manifest must not turn a not-found into a crash
        names = set()
    if base and base in names:
        text += (
            " It is a knowledge base document, not a workspace file — read it with "
            "search_knowledge_base instead."
        )
    return text


@register_tool("read_only")
def read_file(file_path: str):
    """Reads the contents of a file and returns it as a string. file_path is relative to the working folder; an absolute or ~ path inside a folder Saturn can reach also works."""
    _, target_path, error = _resolve(file_path)
    if error:
        return error
    if not target_path.is_file():
        # RAISED, not returned: a missing file is a failed step (status error, disclosed as an
        # incident) exactly as before — only the text changed. The raw OSError told a redraft
        # nothing; the two namespaces (workspace vs. knowledge base, one tool each) are the
        # confusion a small model actually has, so the refusal names the namespace and, when
        # the name matches an ingested document, the tool that reads it (2026-09-02).
        raise FileNotFoundError(_not_found_text(file_path))
    # Always UTF-8: the workspace holds user docs/notes that routinely carry non-cp1252
    # characters, and the default Windows encoding (cp1252) would raise UnicodeDecodeError on
    # them. errors="replace" degrades an undecodable byte to a marker rather than failing the
    # whole read (e.g. when pointed at a binary file by mistake).
    with open(target_path, "r", encoding="utf-8", errors="replace") as file:
        return file.read()


@register_tool("side_effecting")
def write_file(file_path: str, content: str, overwrite: bool = True):
    """Writes content to a file. file_path is relative to the working folder; an absolute or ~ path inside a folder Saturn can reach also works. content is the text to write. overwrite=True (default) replaces the file's contents; pass overwrite=False to append to the existing file instead. To change PART of an existing file, prefer edit_file — it can't accidentally drop the rest of the contents."""
    _, target_path, error = _resolve(file_path)
    if error:
        return error
    # Create the workspace and any intermediate directories so a nested path (e.g.
    # "notes/todo.md") works — without this, writing into a not-yet-existing subdirectory raised
    # FileNotFoundError. Safe: target_path is already verified to be inside the sandbox above.
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Capture the turn-start state (or the file's absence) so /undo can reverse this write.
    snapshot_file(target_path)
    existed = target_path.exists()
    if overwrite:
        with open(target_path, "w", encoding="utf-8") as file:
            file.write(content)
        return MSG_OVERWROTE if existed else MSG_CREATED
    with open(target_path, "a", encoding="utf-8") as file:
        file.write(content)
    return MSG_APPENDED


@register_tool("read_only")
def list_directory(directory: str = "."):
    """Lists the files and folders inside a directory. directory is relative to the working folder (an absolute or ~ path inside a reachable folder also works). Use '.' to list the working folder."""
    _, target_path, error = _resolve_dir(directory)
    if error:
        return error
    return [item.name for item in target_path.iterdir() if not _hidden(item.name)]


def _hidden(name: str) -> bool:
    """Whether a directory entry is hidden: a dot-name. One rule for the navigation tools
    (list_directory / find_files / search_files) and the manifest sync: the registry's own
    `.manifest.md`, `.git`, `.DS_Store` and editor droppings are not workspace CONTENT — on an
    empty workspace the manifest was listed, read, and relayed as the user's data (2026-09-02).
    read_file by explicit path is unchanged: a user may name a dotfile on purpose."""
    return name.startswith(".")


@register_tool("side_effecting")
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Makes a targeted edit to an existing file by replacing an exact text snippet. file_path is relative to the working folder; an absolute or ~ path inside a reachable folder also works. old_string must match the file contents EXACTLY (including whitespace) and must be unique in the file — include surrounding lines to disambiguate, or pass replace_all=True to replace every occurrence. Prefer this over write_file when changing part of a file: it cannot accidentally drop the rest of the contents."""
    _, target_path, error = _resolve(file_path)
    if error:
        return error
    if not target_path.is_file():
        return f"File not found: {file_path}. Use write_file to create a new file."
    # Strict UTF-8 on purpose (unlike read_file's errors='replace'): a replace-decode here
    # would silently corrupt every undecodable byte OUTSIDE the edited snippet when the file
    # is written back. Refusing to edit a non-UTF-8 file is the safe failure.
    try:
        content = target_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot edit {file_path}: it is not valid UTF-8 text (binary or legacy encoding)."

    if old_string == new_string:
        return "old_string and new_string are identical — nothing to change."
    count = content.count(old_string)
    if count == 0:
        return (
            "old_string was not found in the file. It must match the current contents "
            "exactly, including whitespace and indentation — read the file again and retry."
        )
    if count > 1 and not replace_all:
        return (
            f"old_string appears {count} times in the file. Include more surrounding context "
            "to make it unique, or pass replace_all=True to replace every occurrence."
        )

    # Capture the turn-start state so /undo can reverse this edit.
    snapshot_file(target_path)
    new_content = content.replace(old_string, new_string)
    target_path.write_text(new_content, encoding="utf-8")
    n = count if replace_all else 1
    return f"{EDIT_PREFIX}{file_path}: replaced {n} occurrence(s)."


# search_files caps — small at the source so a huge workspace can't flood one observation.
_SEARCH_MAX_MATCHES = 100      # total matching lines returned
_SEARCH_MAX_PER_FILE = 20      # matching lines per file (one log file can't eat the budget)
_SEARCH_MAX_LINE = 200         # chars of each matched line
_SEARCH_MAX_FILE_BYTES = 2_000_000  # skip files larger than this
_FIND_MAX_RESULTS = 200


def _is_binary(path) -> bool:
    """Cheap binary sniff: a NUL byte in the first KB. Wrong for exotic encodings, right for the
    things that matter (images, archives, executables, sqlite files)."""
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(1024)
    except OSError:
        return True


@register_tool("read_only")
def search_files(pattern: str, directory: str = ".", file_glob: str = "*"):
    """Searches the CONTENTS of files for a regular-expression pattern (case-insensitive) and returns matching lines as 'path:line_number: text'. Use this to find where something is mentioned without reading every file. directory is relative to the working folder ('.' = the whole working folder); file_glob filters which files are searched by name (e.g. '*.md'). For finding files by NAME, use find_files instead."""
    workspace, target_path, error = _resolve_dir(directory)
    if error:
        return error
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regular expression: {exc}"

    matches: list[str] = []
    truncated = False
    walk = _ws.Walk(target_path)
    for path in walk:
        if len(matches) >= _SEARCH_MAX_MATCHES:
            truncated = True
            break
        if not fnmatch.fnmatch(path.name, file_glob):
            continue
        try:
            if path.stat().st_size > _SEARCH_MAX_FILE_BYTES or _is_binary(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _ws.relative(path)
        in_file = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if not rx.search(line):
                continue
            matches.append(f"{rel}:{lineno}: {truncate(line.strip(), _SEARCH_MAX_LINE)}")
            in_file += 1
            if in_file >= _SEARCH_MAX_PER_FILE:
                matches.append(f"{rel}: … more matches in this file (capped at {_SEARCH_MAX_PER_FILE})")
                break
            if len(matches) >= _SEARCH_MAX_MATCHES:
                truncated = True
                break
        if truncated:
            break

    if not matches:
        out = f"No matches for /{pattern}/ in {directory!r} (files matching {file_glob!r})."
        return out + ("\n" + _ws.walk_note() if walk.capped else "")
    out = "\n".join(matches)
    if truncated:
        out += f"\n… stopped at {_SEARCH_MAX_MATCHES} matches — narrow the pattern, directory, or file_glob."
    elif walk.capped:
        out += "\n" + _ws.walk_note()
    return out


@register_tool("read_only")
def find_files(pattern: str, directory: str = "."):
    """Finds files and folders by NAME using a glob pattern and returns their paths relative to the working folder. A bare pattern like '*.md' or 'report*' searches recursively under directory; a pattern with '/' (e.g. 'notes/*.txt' or '**/drafts/*.md') is matched as a path. Use this to locate a file when the exact path is unknown; for searching file CONTENTS, use search_files."""
    workspace, target_path, error = _resolve_dir(directory)
    if error:
        return error
    # A bare name pattern means "anywhere under here" — that's what the asker wants from '*.md'.
    # A pattern containing '/' matches the path relative to `directory`; a leading '**/' also
    # matches at the top level (glob's zero-directory reading).
    walk = _ws.Walk(target_path, dirs=True)
    results = []
    for p in walk:
        rel_to_dir = p.relative_to(target_path).as_posix()
        if "/" in pattern:
            hit = fnmatch.fnmatch(rel_to_dir, pattern) or (
                pattern.startswith("**/") and fnmatch.fnmatch(rel_to_dir, pattern[3:]))
        else:
            hit = fnmatch.fnmatch(p.name, pattern)
        if hit:
            results.append(_ws.relative(p) + ("/" if p.is_dir() else ""))
    results.sort()
    if not results:
        out = f"No files matching {pattern!r} under {directory!r}."
        return out + ("\n" + _ws.walk_note() if walk.capped else "")
    if len(results) > _FIND_MAX_RESULTS:
        extra = len(results) - _FIND_MAX_RESULTS
        results = results[:_FIND_MAX_RESULTS] + [f"… {extra} more — narrow the pattern."]
    if walk.capped:
        results.append(_ws.walk_note())
    return "\n".join(results)

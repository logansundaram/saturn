"""
File tools — read, write, edit, list, and search files inside the folders Saturn can reach.

Every path is resolved per call through `core/workspace.resolve` — the launch folder plus any
`/add-dir` folders — so a tool call can never reach anything else. `write_file`,
`edit_file`, `move_file` and `delete_file` are the mutating tools here (gated via
registry.TOOL_RISK), and each records the turn-start state first (stores/snapshots.py) so `/undo`
can reverse it. A delete is a move to the user's Trash: undoable, and recoverable from Finder
after the undo history has moved on.

`search_files` (content regex) and `find_files` (name glob) are the navigation primitives:
without them the agent's only way to locate something is list_directory + reading whole files,
which burns context and iterations. Both are read_only and hard-capped so a huge workspace
can't flood an observation (the tool node clamps again — gotcha #5 — but staying small at the
source keeps the useful part of the result intact).
"""

import fnmatch
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from textutil import truncate
from tools.toolspec import ToolError, register_tool

from core import doctext, hooks
from core import workspace as _ws
from stores.snapshots import record_move, snapshot_file


# The write tools' success lines (tests and the loop benchmark match on them).
MSG_OVERWROTE = "File overwritten successfully"
MSG_CREATED = "File created successfully"
MSG_APPENDED = "Content appended to file successfully"
EDIT_PREFIX = "Edited "  # edit_file's success line: f"{EDIT_PREFIX}{path}: replaced …"


def _resolve(path: str, *, follow: bool = True):
    """Resolve a path the model gave against the folders Saturn can reach
    (core/workspace.resolve — the ONE containment check). Returns (root, target, error): `error`
    is the refusal string when the path is outside every reachable folder (then `target` must
    not be used), else None. Every tool below starts here, and so does the approval preview."""
    target, refusal = _ws.resolve(path, follow=follow)
    return _ws.root(), target, refusal


def _resolve_dir(path: str):
    """`_resolve` for tools that need an existing directory to walk."""
    root, target, error = _resolve(path)
    if error is None and not target.is_dir():
        error = "Path is not a directory."
    return root, target, error


def _control_files() -> "dict[Path, str]":
    """The files that control Saturn itself — never the agent's to write, even through the gate:
    a write to one could loosen the gate (config.yaml's auto_approve, a saved always-allow),
    plant a command that runs ungated (hooks.yaml), plant a standing instruction that loads into
    every later turn's prompt (the two SATURN.md files), or plant a "fact" past the memory
    review gate (the memory file and its pending queue — a bullet stamped `by=user` would read
    as something the user said). Launched from ~ they sit inside the workspace, so the
    containment check alone does not keep them out. The user edits these by hand; the memory
    file changes only through `remember` and /memory; the workspace SATURN.md through /init."""
    from config import config_path, get_config
    from core.memory_review import pending_path
    from nodes.ground import INSTRUCTIONS_FILE, global_instructions_path

    return {
        hooks.hooks_path(): "holds the user's hook commands",
        config_path(): "holds Saturn's settings, the approval gate's included",
        get_config().path("permissions"): "holds the approval gate's saved permissions",
        get_config().path("memory"): "holds Saturn's memory, written only through the review gate",
        pending_path(): "holds the memory review's pending queue",
        global_instructions_path(): "holds the user's standing instructions, loaded every turn",
        _ws.root() / INSTRUCTIONS_FILE: "holds the workspace's standing instructions, loaded every turn",
    }


def _touches(target: Path, control: Path) -> bool:
    """Whether writing or moving `target` reaches `control`: it IS the file, or it is a folder
    the file lives in (moving that folder away and another into its place replaces the file
    without ever naming it). File identity where the paths exist — on macOS's case-insensitive
    disk CONFIG.YAML is config.yaml — and case-folded spelling where they do not exist yet."""
    if _ws._inside(control, target):
        return True
    return Path(str(control).casefold()).is_relative_to(str(target).casefold())


def _refuse_control_file(target_path) -> None:
    """Raise when `target_path` is a control file (`_control_files`) or a folder holding one:
    never the agent's to write or move."""
    try:
        target = Path(target_path).resolve()
        protected = {Path(p).resolve(): why for p, why in _control_files().items()}
    except OSError:
        protected, target = {}, None
    hit = next((p for p in protected if target is not None and _touches(target, p)), None)
    if hit is not None:
        if len(target.parts) < len(hit.parts):
            raise PermissionError(f"{hit} {protected[hit]}, and {target} is a folder it lives "
                                  "in; Saturn never moves or replaces that folder. Ask the "
                                  "user to do it by hand.")
        raise PermissionError(f"{hit} {protected[hit]}; Saturn never writes it. "
                              "Ask the user to edit it by hand.")


def _check_write_allowed(target_path, tool: str) -> None:
    """Raise when a write must not happen: a control file is never the agent's to write or move
    (`_refuse_control_file`), and the user's before-write hooks may say no. RAISED, like
    read_file's not-found, so the round is a failed step the answer's incidents note
    discloses. Asked only once the write is otherwise certain, so a hook never fires for a call
    refused for another reason."""
    _refuse_control_file(target_path)
    refusal = hooks.before_write(target_path, tool)
    if refusal:
        raise PermissionError(refusal)


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


@register_tool("read_only", untrusted=True)
def read_file(file_path: str):
    """Reads the contents of a file and returns it as a string. Text files are returned as written; PDF, Word (.docx) and Excel (.xlsx) files are returned as their text. file_path is relative to the working folder; an absolute or ~ path inside a folder Saturn can reach also works."""
    _, target_path, error = _resolve(file_path)
    if error:
        raise ToolError(error)
    if not target_path.is_file():
        # RAISED, not returned: a missing file is a failed call (status error, disclosed as an
        # incident) exactly as before — only the text changed. The raw OSError told the model
        # nothing; the two namespaces (workspace vs. knowledge base, one tool each) are the
        # confusion a small model actually has, so the refusal names the namespace and, when
        # the name matches an ingested document, the tool that reads it (2026-09-02).
        raise FileNotFoundError(_not_found_text(file_path))
    # A PDF / Word / Excel file is read as its text (core/doctext) — "summarize the PDF on my
    # desktop" is a direct read, not a knowledge-base ingest. Any other binary file is refused
    # by name instead of returned as replacement-character soup.
    document = doctext.extract(target_path)
    if document is not None:
        return document
    try:
        with open(target_path, "rb") as fh:
            binary = b"\0" in fh.read(1024)  # _is_binary's sniff, but an OSError raises below
    except OSError:
        binary = False
    if binary:
        raise ToolError(f"{file_path} is a binary file ({target_path.suffix or 'no extension'}); "
                "read_file reads text, PDF, .docx and .xlsx files.")
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
        raise ToolError(error)
    _check_write_allowed(target_path, "write_file")
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
        result = MSG_OVERWROTE if existed else MSG_CREATED
    else:
        with open(target_path, "a", encoding="utf-8") as file:
            file.write(content)
        result = MSG_APPENDED
    hooks.run("after-write", file=str(target_path), tool="write_file")
    return result


@register_tool("read_only")
def list_directory(directory: str = "."):
    """Lists the files and folders inside a directory. directory is relative to the working folder (an absolute or ~ path inside a reachable folder also works). Use '.' to list the working folder."""
    _, target_path, error = _resolve_dir(directory)
    if error:
        raise ToolError(error)
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
        raise ToolError(error)
    if not target_path.is_file():
        raise ToolError(f"File not found: {file_path}. Use write_file to create a new file.")
    # Strict UTF-8 on purpose (unlike read_file's errors='replace'): a replace-decode here
    # would silently corrupt every undecodable byte OUTSIDE the edited snippet when the file
    # is written back. Refusing to edit a non-UTF-8 file is the safe failure.
    try:
        content = target_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"Cannot edit {file_path}: it is not valid UTF-8 text (binary or legacy encoding).")

    if old_string == new_string:
        raise ToolError("old_string and new_string are identical — nothing to change.")
    count = content.count(old_string)
    if count == 0:
        raise ToolError(
            "old_string was not found in the file. It must match the current contents "
            "exactly, including whitespace and indentation — read the file again and retry."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {count} times in the file. Include more surrounding context "
            "to make it unique, or pass replace_all=True to replace every occurrence."
        )

    _check_write_allowed(target_path, "edit_file")
    # Capture the turn-start state so /undo can reverse this edit.
    snapshot_file(target_path)
    new_content = content.replace(old_string, new_string)
    target_path.write_text(new_content, encoding="utf-8")
    hooks.run("after-write", file=str(target_path), tool="edit_file")
    n = count if replace_all else 1
    return f"{EDIT_PREFIX}{file_path}: replaced {n} occurrence(s)."


@register_tool("side_effecting")
def move_file(source: str, destination: str, overwrite: bool = False):
    """Moves or renames a file or folder. source and destination are relative to the working folder (an absolute or ~ path inside a reachable folder also works). If destination is an existing folder the item is moved INTO it under its own name; missing parent folders are created. An existing file at the destination is replaced only with overwrite=True. Use this to rename, sort or archive files — never read a file and write it back under a new name."""
    # The source is the directory ENTRY: a symlink is moved as a link, never the file it
    # points to (the gate showed the link's name).
    _, src, error = _resolve(source, follow=False)
    if error:
        raise ToolError(error)
    _, dst, error = _resolve(destination)
    if error:
        raise ToolError(error)
    if not (src.exists() or src.is_symlink()):
        raise ToolError(f"File not found in the workspace: {source}.")
    # readme.md -> README.md: on a case-insensitive disk the new spelling "exists" and is the
    # same file, but it is a rename like any other.
    case_only = _ws.case_only(src, dst)
    if dst.is_dir() and not _ws.same(src, dst):
        dst = dst / src.name            # "move it into that folder"
    if _ws.same(src, dst) and not case_only:
        raise ToolError("source and destination are the same place — nothing to move.")
    # By file identity as well as by spelling: `Notes/x` is inside `notes` on a case-insensitive
    # disk, where the rename fails and shutil.move would copy the folder into itself and then
    # delete it. (A symlink to a folder is only a link: it may move into the folder it names.)
    if src.is_dir() and (dst.is_relative_to(src)
                         or (not src.is_symlink() and _ws._inside(dst.parent, src))):
        raise ToolError("cannot move a folder inside itself.")
    if dst.is_dir() and not case_only:
        raise ToolError(f"{_ws.relative(dst)} is a folder; move_file never replaces a folder.")
    if dst.exists() and not (overwrite or case_only):
        raise ToolError(f"{_ws.relative(dst)} already exists. Pick another name, or pass "
                        "overwrite=true to replace it.")
    _refuse_control_file(dst)                # before the source's hook: none fires for a refused move
    _check_write_allowed(src, "move_file")   # a control file is never moved away either
    _check_write_allowed(dst, "move_file")
    was = _ws.relative(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    replacing = dst.exists() and not case_only
    if replacing:
        snapshot_file(dst)               # the file about to be replaced: /undo restores its bytes
    try:
        # A rename replaces an existing FILE in one step, so nothing is deleted before the
        # move is known to work; across disks shutil.move copies, then removes the source.
        if replacing and src.is_dir():
            raise OSError("a folder cannot replace a file")
        shutil.move(str(src), str(dst))
    except OSError as exc:
        raise ToolError(f"could not move {was}: {exc}; nothing was changed.") from exc
    record_move(src, dst)
    hooks.run("after-write", file=str(dst), tool="move_file")
    return f"Moved {was} to {_ws.relative(dst)}"


def _trash_dir() -> Path:
    """Where `delete_file` puts things: the user's own Trash — ~/.Trash on macOS, the
    freedesktop Trash's `files` folder elsewhere."""
    if sys.platform == "darwin":
        return Path.home() / ".Trash"
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "Trash" / "files"


def _trash_slot(trash: Path, name: str) -> Path:
    """A free name in the Trash: `todo.md`, then `todo 2.md`, `todo 3.md`, … — never over
    something already deleted."""
    slot = trash / name
    stem, suffix = (Path(name).stem, Path(name).suffix) if not name.startswith(".") else (name, "")
    n = 2
    while os.path.lexists(slot):
        slot = trash / f"{stem} {n}{suffix}"
        n += 1
    return slot


def _trash_info(slot: Path, origin: Path) -> None:
    """The freedesktop Trash's record of where an item came from (`info/<name>.trashinfo`);
    without it a file manager treats the item as an orphan. macOS needs none."""
    if sys.platform == "darwin":
        return
    from datetime import datetime
    from urllib.parse import quote

    info = slot.parent.parent / "info" / f"{slot.name}.trashinfo"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(f"[Trash Info]\nPath={quote(str(origin))}\n"
                    f"DeletionDate={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n",
                    encoding="utf-8")


@register_tool("side_effecting")
def delete_file(file_path: str):
    """Deletes a file or folder by moving it to the Trash, where the user can still restore it; /undo puts it back. file_path is relative to the working folder (an absolute or ~ path inside a reachable folder also works). Always delete with this tool, never with rm in run_shell, which cannot be undone. One item per call: to delete several, call it once for each."""
    # The directory ENTRY, like move_file: a symlink is deleted as a link, never what it names.
    _, target, error = _resolve(file_path, follow=False)
    if error:
        raise ToolError(error)
    if not os.path.lexists(target):
        raise ToolError(f"File not found in the workspace: {file_path}.")
    if any(_ws.same(target, r) for r in _ws.roots()):
        raise ToolError(f"{_ws.relative(target)} is a working folder Saturn was given; it never "
                        "deletes one. Delete the items inside it instead.")
    _check_write_allowed(target, "delete_file")   # a control file, or a folder holding one
    was = _ws.relative(target)
    trash = _trash_dir()
    try:
        trash.mkdir(parents=True, exist_ok=True)
        slot = _trash_slot(trash, target.name)
        shutil.move(str(target), str(slot))
    except OSError as exc:
        raise ToolError(f"could not delete {was}: {exc}; nothing was changed.") from exc
    record_move(target, slot)
    _trash_info(slot, target)
    hooks.run("after-write", file=str(target), tool="delete_file")
    return f"Deleted {was} (moved to the Trash; /undo puts it back)."


# search_files caps — small at the source so a huge workspace can't flood one observation.
_SEARCH_MAX_MATCHES = 100      # total matching lines returned
_SEARCH_MAX_PER_FILE = 20      # matching lines per file (one log file can't eat the budget)
_SEARCH_MAX_LINE = 200         # chars of each matched line
_SEARCH_MAX_FILE_BYTES = 2_000_000  # skip files larger than this
# Wall-clock budget for one content search. Launched from ~, a search that matches nothing reads
# every text file under home — measured 80 s (17,101 files, 796 MB, 2026-09-29). Past the budget
# the search stops and SAYS it stopped, so "no matches" is never claimed for a partial scan.
_SEARCH_MAX_SECONDS = 10.0
_FIND_MAX_RESULTS = 200

# ── Spotlight (macOS) behind search_files ────────────────────────────────────────────────────
# macOS has already indexed file contents, PDFs included: a phrase query over ~ answers in about
# a second where the direct walk needs 80 (both measured 2026-10-01). The index is a CANDIDATE
# source, never the judge — it matches words across line breaks and lags behind edits — so every
# candidate is read and matched by the same regex, and passes the same containment and pruning
# rules as the walk. A small folder is still walked whole; Spotlight covers what the walk could
# not reach in `_SEARCH_WALK_SECONDS`, plus the document types the walk skips as binary.
_SEARCH_WALK_SECONDS = 2.0     # the direct walk's budget once the index has answered
_SPOTLIGHT_SECONDS = 5.0       # mdfind's own timeout
_SPOTLIGHT_MAX_DOCS = 15       # PDFs / .docx / .xlsx reported per search
# Opening a document is a full parse (fifteen PDFs made one search over ~ take 11 s,
# 2026-10-01). Past this budget a document is reported as an index match, with its date, and
# not opened — the answer to "find the tax return I saved last spring" is the file.
_SPOTLIGHT_DOC_SECONDS = 3.0
_DOC_SUFFIXES = (".pdf", ".docx", ".xlsx")
# What Spotlight can be asked: a plain word or phrase. Anything with regex syntax (`+`
# included: `a+b` matches "aab", never the text "a+b"), or a quote or backslash that would need
# escaping in the query, stays with the walk.
_PLAIN_PHRASE = re.compile(r"\w[\w ,'@:/-]{2,}")
SPOTLIGHT_NOTE = ("… searched the nearest files directly and the rest through Spotlight's index; "
                  "a file Spotlight has not indexed may be missing — narrow the directory to "
                  "search it in full.")


def _platform() -> str:
    return sys.platform


def _mdfind(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _spotlight_query(literal: str, directory: Path) -> "list[Path] | None":
    """The files under `directory` whose indexed content holds `literal`, or None when there is
    no index answer (not macOS, mdfind missing, failed or timed out, or a folder the index
    does not cover)."""
    if _platform() != "darwin":
        return None
    query = f'kMDItemTextContent == "*{literal}*"cd'
    try:
        proc = _mdfind(["mdfind", "-onlyin", str(directory), query], _SPOTLIGHT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    paths = [Path(line) for line in (proc.stdout or "").splitlines() if line.strip()]
    if not paths and not _spotlight_knows(directory):
        return None
    return paths


def _spotlight_knows(directory: Path) -> bool:
    """Whether the index covers `directory`. mdfind exits 0 with nothing for a folder Spotlight
    does not index (a Privacy exclusion, a hidden folder, /tmp), so an empty answer means
    "nothing matches" only when this is true. Asked of the folder's own record: the date-added
    attribute exists only in the index (probed 2026-10-01: set under ~/Documents, null under
    /private/tmp and a .venv), and one mdls is instant where counting a home folder's files
    through mdfind took minutes."""
    try:
        proc = _mdfind(["mdls", "-raw", "-name", "kMDItemDateAdded", str(directory)],
                       _SPOTLIGHT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and (proc.stdout or "").strip() not in ("", "(null)")


# The seam search_files calls (tests replace it; conftest turns it off by default).
_spotlight = _spotlight_query


def _index_spelling(top: Path, paths: "list[Path]") -> Path:
    """`top` as the index spells it. On macOS's case-insensitive disk `documents` is
    `Documents`, and mdfind answers with the disk's spelling: a candidate is recognized as
    inside `top`, and as a file the walk already read, only when both are spelled alike."""
    for path in paths[:1]:
        for anc in path.parents:
            if anc == top:
                break
            if _ws.same(anc, top):
                return anc
    return top


def _spotlight_candidates(paths: "list[Path]", top: Path, file_glob: str) -> "list[Path]":
    """The index's answers that the walk itself would have visited: inside `top` and a reachable
    folder, no hidden or dependency folder on the way, the name matching `file_glob`."""
    home_library = Path.home().resolve() / "Library"
    out = []
    for path in paths:
        try:
            rel = path.relative_to(top)
        except ValueError:
            continue
        if any(part.startswith(".") or part in _ws._HEAVY_DIRS for part in rel.parts):
            continue
        if path.is_relative_to(home_library) and not top.is_relative_to(home_library):
            continue
        if not fnmatch.fnmatch(path.name, file_glob) or not path.is_file():
            continue
        if _ws.resolve(str(path))[1] is not None:
            continue
        out.append(path)
    # Text files first (cheap to check), then documents, newest first.
    text = sorted(p for p in out if p.suffix.lower() not in _DOC_SUFFIXES)
    docs = [p for p in out if p.suffix.lower() in _DOC_SUFFIXES]
    return text + sorted(docs, key=lambda p: (-_mtime(p), p))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_binary(path) -> bool:
    """Cheap binary sniff: a NUL byte in the first KB. Wrong for exotic encodings, right for the
    things that matter (images, archives, executables, sqlite files)."""
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(1024)
    except OSError:
        return True


@register_tool("read_only", untrusted=True)
def search_files(pattern: str, directory: str = ".", file_glob: str = "*"):
    """Searches the CONTENTS of files for a regular-expression pattern (case-insensitive) and returns matching lines as 'path:line_number: text'. Use this to find where something is mentioned without reading every file. directory is relative to the working folder ('.' = the whole working folder); file_glob filters which files are searched by name (e.g. '*.md'). On macOS a plain word or phrase is also found inside PDF / Word / Excel files. For finding files by NAME, use find_files instead."""
    workspace, target_path, error = _resolve_dir(directory)
    if error:
        raise ToolError(error)
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}")

    matches: list[str] = []
    truncated = timed_out = False
    deadline = time.monotonic() + _SEARCH_MAX_SECONDS

    def scan(path, text) -> None:
        """Append `path`'s matching lines; sets `truncated` at the total cap."""
        nonlocal truncated
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

    def plain_text(path) -> "str | None":
        try:
            if path.stat().st_size > _SEARCH_MAX_FILE_BYTES or _is_binary(path):
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    # Ask the index first (about a second): with its answer in hand the walk only needs to
    # cover the nearest files, and the rest comes from the candidates below.
    literal = pattern.strip() if _PLAIN_PHRASE.fullmatch(pattern.strip()) else None
    indexed = _spotlight(literal, target_path) if literal else None
    # The walk's budget starts once the index has answered: a slow mdfind must not leave the
    # nearest files unread, and one that gave no answer (it timed out on a common word) leaves
    # the walk as the whole search, on its whole budget.
    if indexed is None:
        walk_deadline = deadline = time.monotonic() + _SEARCH_MAX_SECONDS
    else:
        target_path = _index_spelling(target_path, indexed)
        walk_deadline = min(deadline, time.monotonic() + _SEARCH_WALK_SECONDS)

    walked: set = set()
    walk_cut = False
    walk = _ws.Walk(target_path)
    for path in walk:
        if len(matches) >= _SEARCH_MAX_MATCHES:
            truncated = True
            break
        if time.monotonic() > walk_deadline:
            walk_cut = True
            break
        walked.add(path)
        if not fnmatch.fnmatch(path.name, file_glob):
            continue
        text = plain_text(path)
        if text is not None:
            scan(path, text)
        if truncated:
            break

    if indexed is not None and not truncated:
        docs = 0
        docs_deadline = None
        for path in _spotlight_candidates(indexed, target_path, file_glob):
            if time.monotonic() > deadline:
                timed_out = True
                break
            if len(matches) >= _SEARCH_MAX_MATCHES:
                truncated = True
                break
            if path.suffix.lower() in _DOC_SUFFIXES:
                if docs >= _SPOTLIGHT_MAX_DOCS:
                    continue
                docs += 1
                if docs_deadline is None:
                    docs_deadline = time.monotonic() + _SPOTLIGHT_DOC_SECONDS
                if time.monotonic() > docs_deadline:
                    day = time.strftime("%Y-%m-%d", time.localtime(_mtime(path)))
                    matches.append(f"{_ws.relative(path)}: in Spotlight's index, not opened "
                                   f"(modified {day}) — read_file opens it")
                    continue
                try:
                    text = doctext.extract(path)
                except Exception:        # a corrupt document is skipped, like an unreadable file
                    text = None
            elif path in walked:
                continue                 # the walk already read it
            else:
                text = plain_text(path)
            if text is not None:
                scan(path, text)
            if truncated:
                break
    elif walk_cut:
        timed_out = True                 # no index to cover the rest: the old partial-scan case

    timeout_note = (f"… stopped after {_SEARCH_MAX_SECONDS:g} s of searching; files not yet "
                    "searched may match — narrow the directory or file_glob.")
    via_index = indexed is not None and walk_cut
    if not matches:
        if timed_out:
            return (f"No matches for /{pattern}/ in the files searched so far under {directory!r} "
                    f"(files matching {file_glob!r}).\n" + timeout_note)
        out = f"No matches for /{pattern}/ in {directory!r} (files matching {file_glob!r})."
        if via_index:
            return out + "\n" + SPOTLIGHT_NOTE
        return out + ("\n" + _ws.walk_note() if walk.capped else "")
    out = "\n".join(matches)
    if truncated:
        out += f"\n… stopped at {_SEARCH_MAX_MATCHES} matches — narrow the pattern, directory, or file_glob."
    elif timed_out:
        out += "\n" + timeout_note
    elif via_index:
        out += "\n" + SPOTLIGHT_NOTE
    elif walk.capped:
        out += "\n" + _ws.walk_note()
    return out


@register_tool("read_only")
def find_files(pattern: str, directory: str = "."):
    """Finds files and folders by NAME using a glob pattern and returns their paths relative to the working folder. A bare pattern like '*.md' or 'report*' searches recursively under directory; a pattern with '/' (e.g. 'notes/*.txt' or '**/drafts/*.md') is matched as a path. Use this to locate a file when the exact path is unknown; for searching file CONTENTS, use search_files."""
    workspace, target_path, error = _resolve_dir(directory)
    if error:
        raise ToolError(error)
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

"""
Maintains per-directory markdown manifests that record metadata and LLM-generated
summaries for every document the agent can see.

Two manifests:
  database/workspace/.manifest.md  — files the agent can read/write via tools
  database/documents/.manifest.md  — files ingested into the RAG vector store

The grounding (`ground`) node reads both at the start of each turn so the planning LLM
knows exactly what documents exist and what they contain before deciding whether to
call read_file or trigger retrieval.
"""

"""
vibe coded, need to review
"""

import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path, PurePath

from langchain.messages import HumanMessage

import diag
from config import get_config

_MANIFEST_HEADER = "# Document manifest\n\n"


# Manifest paths resolve from config at *call time*, not import time, so a live `/config
# paths.*` change is honored without a restart (config.py is the single source of truth).
def _workspace_manifest() -> Path:
    return get_config().path("workspace") / ".manifest.md"


def _documents_manifest() -> Path:
    return get_config().path("documents") / ".manifest.md"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _workspace_key(file_path: str) -> str:
    """Canonical workspace-manifest key: POSIX separators, './' segments dropped. Entries are
    matched by exact string, and three writers share them — write_file, edit_file, and /undo's
    manifest sync — so register and remove must land on ONE form or the manifest accumulates
    duplicate/phantom entries the ground node then feeds the LLM every turn. Defense in depth
    only: PurePath does NOT collapse '..' components, so the load-bearing canonicalization is the
    callers passing a sandbox-RESOLVED relative path (tools/files.py derives it from the resolved
    target); this just papers over separator and './' drift from any future caller."""
    return PurePath(file_path).as_posix()


def register_workspace_file(file_path: str, content: str) -> None:
    """Called by write_file/edit_file after a successful write. Keyed by the full relative path
    (not the basename) so files with the same name in different subdirs don't collide on one
    entry; normalized via _workspace_key so the summary cache and the manifest share one
    canonical key regardless of how the caller spelled the path."""
    key = _workspace_key(file_path)
    _upsert(_workspace_manifest(), key, content, PurePath(key).suffix)


def remove_workspace_file(file_path: str) -> None:
    """Strip a workspace file's manifest entry. Called by /undo (stores/snapshots.py) when it
    deletes a file the undone turn created, so the grounding manifest never lists a file that is
    gone. The cached summary is left alone — it is keyed by content hash, so it is simply unused.
    Normalized with the same _workspace_key as register, so a remove always finds the entry the
    register created."""
    _remove_entry(_workspace_manifest(), _workspace_key(file_path))


def register_rag_document(source: str, content: str) -> None:
    """Called by rag.sync for each new/changed document. `source` is the corpus-relative path;
    key the manifest + summary cache by it (not the basename) so e.g. teamA/report.md and
    teamB/report.md get distinct entries instead of overwriting each other.

    Deliberately NOT normalized (unlike the workspace pair): rag.sync keys register, remove, AND
    its index.json `files` dict by the same str(path.relative_to(root)) form — normalizing only
    this side would desync remove_rag_document's manifest + summary-cache deletion from the
    vector index. If symmetry is ever wanted, both sides AND the index must move together."""
    _upsert(_documents_manifest(), source, content, Path(source).suffix)


def remove_rag_document(source: str) -> None:
    """Strip a document's entry from the RAG manifest and its cached summary. Called by rag.sync
    when a file is removed from the corpus, so the manifest never lists documents that are gone."""
    name = source
    _remove_entry(_documents_manifest(), name)
    cache = _read_summary_cache()
    if name in cache:
        del cache[name]
        _write_summary_cache(cache)


def read_workspace_manifest() -> str:
    return _read_manifest_text(_workspace_manifest())


def read_documents_manifest() -> str:
    return _read_manifest_text(_documents_manifest())


_META_FIELD_RE = re.compile(r"\*\*(\w+)\*\*:\s*([^|]+?)\s*(?:\||$)")


def manifest_entries(text: str) -> list[dict]:
    """Parse a manifest's `### <name>` blocks into structured rows for the /docs table:
    [{name, type, added, size, summary}]. Tolerant of hand-edited manifests — a block missing
    its metadata line still yields its name plus whatever summary text follows."""
    entries: list[dict] = []
    for block in re.split(r"^### ", text, flags=re.M)[1:]:
        lines = block.splitlines()
        if not lines or not lines[0].strip():
            continue
        entry = {"name": lines[0].strip(), "type": "", "added": "", "size": "", "summary": ""}
        body = lines[1:]
        if body and body[0].lstrip().startswith("- **"):
            for key, val in _META_FIELD_RE.findall(body[0]):
                k = key.lower()
                if k in ("type", "added", "size"):
                    entry[k] = val.strip()
            body = body[1:]
        entry["summary"] = " ".join(ln.strip() for ln in body if ln.strip())
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Summaries are an LLM call per document, so they're cached by content hash at
# `paths.cache/summaries.json` — re-summarizing only happens when a file's content changes.
# Keyed by basename (matching the manifest's `### <name>` entries).
_SUMMARY_CACHE_FILE = "summaries.json"

# mtime-validated in-memory memos: syncing N files used to re-read + re-parse the whole
# summaries.json and re-read the whole manifest PER FILE (O(N²) bytes as they grow), and ground
# re-read both manifests every turn. One stat per read validates the memo; the mtime check (not
# a blind cache) keeps a hand-edited file honest. Keyed by path so isolated test configs and a
# live `/config paths.*` change each get their own slot. Writes stay per-change (an LLM summary
# is expensive — batching writes to a sync-end flush would lose them all on a crash).
_summary_mem: "dict[str, tuple[int, dict]]" = {}
_manifest_mem: "dict[str, tuple[int, str]]" = {}


def _summary_cache_path() -> Path:
    return get_config().path("cache") / _SUMMARY_CACHE_FILE


def _read_summary_cache() -> dict:
    p = _summary_cache_path()
    key = str(p)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:  # absent (or unreadable) file: nothing cached
        _summary_mem.pop(key, None)
        return {}
    hit = _summary_mem.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        cache = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    _summary_mem[key] = (mtime, cache)
    return cache


def _write_summary_cache(cache: dict) -> None:
    p = _summary_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    try:
        _summary_mem[str(p)] = (p.stat().st_mtime_ns, cache)
    except OSError:
        _summary_mem.pop(str(p), None)


def _read_manifest_text(path: Path) -> str:
    key = str(path)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:  # no manifest yet
        _manifest_mem.pop(key, None)
        return ""
    hit = _manifest_mem.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    text = path.read_text(encoding="utf-8")
    _manifest_mem[key] = (mtime, text)
    return text


def _write_manifest_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        _manifest_mem[str(path)] = (path.stat().st_mtime_ns, text)
    except OSError:
        _manifest_mem.pop(str(path), None)


def _summarize(content: str, filename: str) -> str:
    """Ask the LLM for a 1-2 sentence summary, cached by content hash. Unchanged documents reuse
    the cached summary (no LLM call); failures fall back gracefully and are not cached, so they
    retry next run."""
    digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    cache = _read_summary_cache()
    hit = cache.get(filename)
    if hit and hit.get("hash") == digest:
        return hit["summary"]

    try:
        # Import here to avoid circular import at module load time (core.messages pulls the
        # live tool registry, which imports tools/files.py, which imports this module).
        # Summaries are a cheap background task -> the `utility` role.
        from core.llms import get_model
        from core.messages import DOC_SUMMARY_PROMPT

        start = time.perf_counter()
        msg = HumanMessage(
            content=DOC_SUMMARY_PROMPT.format(filename=filename, content=content[:4000])
        )
        response = get_model("utility").invoke([msg])
        diag.log(
            f"document_registry summary ({filename}) : {time.perf_counter() - start:.4f}s"
        )
        # Collapse to ONE line: the manifest locates entry boundaries by "\n### " searches, so a
        # multi-line summary containing a markdown heading would forge a boundary and corrupt
        # the manifest ground loads every turn (untrusted document text steers this summary).
        summary = " ".join(str(response.content).split())
    except Exception as exc:
        diag.log(f"document_registry: summary failed for {filename}: {exc}")
        return "No summary available."

    cache[filename] = {"hash": digest, "summary": summary}
    _write_summary_cache(cache)
    return summary


def _upsert(manifest_path: Path, filename: str, content: str, suffix: str) -> None:
    # The one-line collapse also runs here so a MULTI-LINE summary cached by an older version
    # can't forge a "\n### " entry boundary on its way into the manifest.
    summary = " ".join(str(_summarize(content, filename)).split())
    size_kb = len(content.encode()) / 1024

    entry_body = (
        f"- **Type**: {suffix or 'unknown'} | "
        f"**Added**: {date.today()} | "
        f"**Size**: {size_kb:.1f} KB\n"
        f"{summary}"
    )
    full_entry = f"### {filename}\n{entry_body}\n"

    content_text = _read_manifest_text(manifest_path)
    if not content_text:
        _write_manifest_text(manifest_path, _MANIFEST_HEADER + full_entry)
        return

    marker = f"### {filename}\n"

    if marker in content_text:
        start_idx = content_text.index(marker)
        rest = content_text[start_idx + len(marker) :]
        next_entry = rest.find("\n### ")
        if next_entry == -1:
            content_text = content_text[:start_idx] + full_entry
        else:
            content_text = (
                content_text[:start_idx] + full_entry + rest[next_entry + 1 :]
            )
    else:
        content_text = content_text.rstrip("\n") + "\n\n" + full_entry + "\n"

    _write_manifest_text(manifest_path, content_text)


def _remove_entry(manifest_path: Path, filename: str) -> None:
    """Delete the `### <filename>` block from a manifest, if present. Inverse of `_upsert`."""
    text = _read_manifest_text(manifest_path)
    marker = f"### {filename}\n"
    if marker not in text:
        return
    start_idx = text.index(marker)
    rest = text[start_idx + len(marker) :]
    next_entry = rest.find("\n### ")
    if next_entry == -1:
        text = text[:start_idx].rstrip("\n") + "\n"
    else:
        text = text[:start_idx] + rest[next_entry + 1 :]
    _write_manifest_text(manifest_path, text)

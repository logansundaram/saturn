"""
Maintains the knowledge-base manifest: `database/documents/.manifest.md`, one entry with
metadata and a one-line description per document ingested into the RAG vector store.

The grounding (`ground`) node reads it at the start of each turn so the agent knows what the
knowledge base holds before deciding whether to search it.

The description is MECHANICAL (first heading / first non-empty line): an ingest costs no model
call, and the manifest's job — "these files exist, roughly this is what each is" — the first
line already answers.
"""

import re
from datetime import date
from pathlib import Path

from config import get_config
from textutil import clip

_MANIFEST_HEADER = "# Document manifest\n\n"


# The manifest path resolves from config at *call time*, not import time, so a live `/config
# paths.*` change is honored without a restart (config.py is the single source of truth).
def _documents_manifest() -> Path:
    return get_config().path("documents") / ".manifest.md"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_rag_document(source: str, content: str) -> None:
    """Called by rag.sync for each new/changed document. `source` is the corpus-relative path;
    key the manifest by it (not the basename) so e.g. teamA/report.md and teamB/report.md get
    distinct entries instead of overwriting each other.

    Deliberately NOT normalized: rag.sync keys register, remove, AND its index.json `files`
    dict by the same str(path.relative_to(root)) form — normalizing only this side would desync
    remove_rag_document's manifest deletion from the vector index. If symmetry is ever wanted,
    both sides AND the index must move together."""
    _upsert(_documents_manifest(), source, content, Path(source).suffix)


def remove_rag_document(source: str) -> None:
    """Strip a document's entry from the RAG manifest. Called by rag.sync when a file is removed
    from the corpus, so the manifest never lists documents that are gone."""
    _remove_entry(_documents_manifest(), source)


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


# mtime-validated in-memory memo: ground re-reads the manifest every turn, and a sync upserts it
# once per file. One stat per read validates the memo; the mtime check (not a blind cache) keeps a hand-edited file honest. Keyed by path so isolated
# test configs and a live `/config paths.*` change each get their own slot.
_manifest_mem: "dict[str, tuple[int, str]]" = {}


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


# The description clip: long first lines truncate rather than bloat the every-turn context.
_DESC_CAP = 160


def _summarize(content: str, filename: str) -> str:
    """A mechanical one-line description: the first non-empty line (a markdown heading's `#`s
    stripped), whitespace-collapsed and clipped. Untrusted document text can't steer more than
    that one clipped line into the every-turn context."""
    for line in content.splitlines():
        line = " ".join(line.strip().lstrip("#").split())
        if line:
            return clip(line, _DESC_CAP)
    return "(empty file)"


def _upsert(manifest_path: Path, filename: str, content: str, suffix: str) -> None:
    # The one-line collapse also runs here so a multi-line summary can't forge a "\n### " entry
    # boundary on its way into the manifest.
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

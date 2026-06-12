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
from pathlib import Path

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


def register_workspace_file(file_path: str, content: str) -> None:
    """Called by write_file after a successful write. Keyed by the full relative path (not the
    basename) so files with the same name in different subdirs don't collide on one entry."""
    _upsert(_workspace_manifest(), file_path, content, Path(file_path).suffix)


def remove_workspace_file(file_path: str) -> None:
    """Strip a workspace file's manifest entry. Called by /undo (stores/snapshots.py) when it
    deletes a file the undone turn created, so the grounding manifest never lists a file that is
    gone. The cached summary is left alone — it is keyed by content hash, so it is simply unused."""
    _remove_entry(_workspace_manifest(), file_path)


def register_rag_document(source: str, content: str) -> None:
    """Called by rag.sync for each new/changed document. `source` is the corpus-relative path;
    key the manifest + summary cache by it (not the basename) so e.g. teamA/report.md and
    teamB/report.md get distinct entries instead of overwriting each other."""
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
    manifest = _workspace_manifest()
    return manifest.read_text(encoding="utf-8") if manifest.exists() else ""


def read_documents_manifest() -> str:
    manifest = _documents_manifest()
    return manifest.read_text(encoding="utf-8") if manifest.exists() else ""


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


def _summary_cache_path() -> Path:
    return get_config().path("cache") / _SUMMARY_CACHE_FILE


def _read_summary_cache() -> dict:
    p = _summary_cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_summary_cache(cache: dict) -> None:
    p = _summary_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


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
        # Import here to avoid circular import at module load time. Summaries are a cheap
        # background task -> the `utility` role.
        from core.llms import get_model

        start = time.perf_counter()
        msg = HumanMessage(
            content=(
                f"Summarize the following document in 1-2 sentences. "
                f"Be specific: name what information it contains, not just its topic. "
                f"Document name: {filename}\n\n{content[:4000]}"
            )
        )
        response = get_model("utility").invoke([msg])
        diag.log(
            f"document_registry summary ({filename}) : {time.perf_counter() - start:.4f}s"
        )
        summary = response.content.strip()
    except Exception as exc:
        diag.log(f"document_registry: summary failed for {filename}: {exc}")
        return "No summary available."

    cache[filename] = {"hash": digest, "summary": summary}
    _write_summary_cache(cache)
    return summary


def _upsert(manifest_path: Path, filename: str, content: str, suffix: str) -> None:
    summary = _summarize(content, filename)
    size_kb = len(content.encode()) / 1024

    entry_body = (
        f"- **Type**: {suffix or 'unknown'} | "
        f"**Added**: {date.today()} | "
        f"**Size**: {size_kb:.1f} KB\n"
        f"{summary}"
    )
    full_entry = f"### {filename}\n{entry_body}\n"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        manifest_path.write_text(_MANIFEST_HEADER + full_entry, encoding="utf-8")
        return

    content_text = manifest_path.read_text(encoding="utf-8")
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

    manifest_path.write_text(content_text, encoding="utf-8")


def _remove_entry(manifest_path: Path, filename: str) -> None:
    """Delete the `### <filename>` block from a manifest, if present. Inverse of `_upsert`."""
    if not manifest_path.exists():
        return
    text = manifest_path.read_text(encoding="utf-8")
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
    manifest_path.write_text(text, encoding="utf-8")

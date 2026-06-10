"""
Workspace snapshots — the undo layer behind the mutating file tools (`/undo`).

Before `write_file` / `edit_file` changes a workspace file, the file's current bytes are copied
into a per-turn snapshot batch under `config.path("snapshots")` (a file that does not exist yet is
recorded too, so undoing a creation deletes it). `/undo` restores the most recent batch and removes
it; batches are pruned to the last `_KEEP_BATCHES` turns so the directory can't grow unbounded.

Scope: only the file tools snapshot — `run_shell` can touch anything, so its effects are NOT
undoable (the approval gate showing the exact command is its safety boundary). Everything here is
best-effort in the same spirit as the rest of the stores: a snapshot failure is logged via diag and
never blocks the write it was protecting.

Batches are keyed to turns: `begin_turn(query)` (called from agent._fresh_turn) arms a new batch
lazily — the directory is only created when the first snapshot of that turn actually lands, so
read-only turns leave nothing behind.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import diag
from config import get_config

# How many snapshot batches (turns that wrote files) to retain.
_KEEP_BATCHES = 20

_MANIFEST = "manifest.json"
_FILES_DIR = "files"

# The armed-but-not-yet-created batch for the current turn: {"id": ..., "query": ...} or None.
_pending: "dict | None" = None
# The batch directory once the first snapshot lands; reset each begin_turn.
_active_dir: "Path | None" = None


def _root() -> Path:
    return get_config().path("snapshots")


def begin_turn(query: str = "") -> None:
    """Arm a fresh batch for the turn that is about to run. Lazy: no directory is created until a
    mutating file tool actually snapshots something."""
    global _pending, _active_dir
    _pending = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "query": " ".join(query.split())[:200],
    }
    _active_dir = None


def _load_manifest(batch_dir: Path) -> dict:
    return json.loads((batch_dir / _MANIFEST).read_text(encoding="utf-8"))


def _save_manifest(batch_dir: Path, manifest: dict) -> None:
    (batch_dir / _MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _ensure_batch() -> "Path | None":
    """Create (or return) the current turn's batch directory. None if no turn was armed (e.g. a
    tool called outside the loop, like benchmark setup) — snapshotting silently no-ops then."""
    global _active_dir
    if _active_dir is not None:
        return _active_dir
    if _pending is None:
        return None
    batch_dir = _root() / _pending["id"]
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / _FILES_DIR).mkdir(exist_ok=True)
    _save_manifest(
        batch_dir,
        {
            "id": _pending["id"],
            "created": datetime.now().isoformat(timespec="seconds"),
            "query": _pending["query"],
            "files": [],
        },
    )
    _active_dir = batch_dir
    _prune()
    return _active_dir


def snapshot_file(rel_path: str, target: Path) -> None:
    """Record `target` (a sandbox-resolved workspace file at workspace-relative `rel_path`) before
    it is mutated. Existing file -> its bytes are copied into the batch; missing file -> recorded
    as not-existing so an undo deletes the file the tool is about to create. First snapshot of a
    path in a batch wins (it is the turn-start state); later writes to the same file are no-ops.
    Best-effort: any failure is logged and swallowed — the write itself must not be blocked."""
    try:
        batch_dir = _ensure_batch()
        if batch_dir is None:
            return
        manifest = _load_manifest(batch_dir)
        rel = Path(rel_path).as_posix()
        if any(f["path"] == rel for f in manifest["files"]):
            return  # turn-start state already captured
        existed = target.exists()
        if existed:
            saved = batch_dir / _FILES_DIR / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)  # byte copy — workspace files may not be UTF-8
        manifest["files"].append({"path": rel, "existed": existed})
        _save_manifest(batch_dir, manifest)
    except Exception as exc:
        diag.log(f"snapshot_file failed for {rel_path}: {exc}")


def _batch_dirs() -> list[Path]:
    """All batch directories, oldest -> newest (ids are timestamp-sortable)."""
    root = _root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / _MANIFEST).exists())


def _prune(keep: int = _KEEP_BATCHES) -> None:
    """Drop the oldest batches beyond `keep`. Best-effort."""
    try:
        for stale in _batch_dirs()[:-keep] if keep > 0 else _batch_dirs():
            shutil.rmtree(stale, ignore_errors=True)
    except Exception as exc:
        diag.log(f"snapshot prune failed: {exc}")


def list_batches() -> list[dict]:
    """Summaries of the stored batches, newest first:
    {"id", "created", "query", "files": [rel, ...]}."""
    out = []
    for batch_dir in reversed(_batch_dirs()):
        try:
            m = _load_manifest(batch_dir)
            out.append(
                {
                    "id": m.get("id", batch_dir.name),
                    "created": m.get("created", ""),
                    "query": m.get("query", ""),
                    "files": [f["path"] for f in m.get("files", [])],
                }
            )
        except Exception as exc:
            diag.log(f"unreadable snapshot batch {batch_dir.name}: {exc}")
    return out


def undo_last() -> "tuple[str, list[str]]":
    """Restore the most recent snapshot batch into the workspace and delete the batch.

    Returns (batch_summary, action_lines); raises RuntimeError when there is nothing to undo.
    Each touched file is restored to its turn-start bytes; a file the batch recorded as
    not-existing (the tool created it) is deleted. Restore paths are re-resolved against the
    CURRENT workspace and sandbox-checked, mirroring the file tools."""
    batches = _batch_dirs()
    if not batches:
        raise RuntimeError("no snapshots to undo — nothing has written to the workspace yet")
    batch_dir = batches[-1]
    manifest = _load_manifest(batch_dir)
    workspace = get_config().path("workspace")
    # Imported here, not at module top: document_registry pulls in the LLM stack for its
    # summarizer, which snapshot_file (called from every gated write) shouldn't load eagerly.
    from stores.document_registry import register_workspace_file, remove_workspace_file

    actions: list[str] = []
    for entry in reversed(manifest.get("files", [])):
        rel = entry["path"]
        target = (workspace / rel).resolve()
        if not target.is_relative_to(workspace):
            actions.append(f"skipped {rel} (outside the current workspace)")
            continue
        try:
            if entry.get("existed"):
                saved = batch_dir / _FILES_DIR / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, target)
                actions.append(f"restored {rel}")
            else:
                if target.exists():
                    target.unlink()
                actions.append(f"deleted {rel} (was created by that turn)")
        except Exception as exc:
            actions.append(f"FAILED to restore {rel}: {exc}")
            continue
        # Keep the grounding manifest truthful about what's in the workspace now. Best-effort —
        # a manifest hiccup must not fail the restore that already landed. Restored content was
        # usually summarized before, so the hash-keyed cache makes this LLM-free.
        try:
            if entry.get("existed"):
                register_workspace_file(rel, target.read_text(encoding="utf-8", errors="replace"))
            else:
                remove_workspace_file(rel)
        except Exception as exc:
            diag.log(f"undo manifest sync failed for {rel}: {exc}")

    label = manifest.get("created", "") or manifest.get("id", batch_dir.name)
    query = manifest.get("query", "")
    summary = f"{label}" + (f' — "{query}"' if query else "")
    shutil.rmtree(batch_dir, ignore_errors=True)
    return summary, actions

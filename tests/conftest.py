"""
Shared pytest fixtures.

The suite tests the INVARIANT / SECURITY surfaces the docs call load-bearing (the positional
plan-accounting walkers, the shell allowlist matcher, the observation clamp, the surgical YAML
persist, the snapshot/undo layer) plus the pure helpers behind newer features (citations,
RAG loaders, sessions). Everything runs offline: no test calls
an LLM, the network, or the embedder.

`isolated_paths` points every `paths.*` entry in the live config at a throwaway tmp directory so
no test can touch the real database/ — config resolves paths against the repo root, but an
absolute path wins the join, which is exactly what tmp_path provides.
"""

import sys
from pathlib import Path

import pytest

# Make the repo root importable no matter how pytest was invoked (the app itself relies on
# running from the repo root; tests shouldn't).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every configured data path into tmp_path for the duration of one test.
    monkeypatch.setitem restores the real values afterward."""
    from config import get_config

    cfg = get_config()
    redirects = {
        "database": tmp_path / "database",
        "documents": tmp_path / "database" / "documents",
        "workspace": tmp_path / "database" / "workspace",
        "cache": tmp_path / "database" / "cache",
        "memory": tmp_path / "database" / "memory" / "memory.md",
        "db_sqlite": tmp_path / "database" / "db.sqlite",
        "sessions": tmp_path / "database" / "sessions",
        "snapshots": tmp_path / "database" / "snapshots",
        "permissions": tmp_path / "database" / "permissions.json",
        "exports": tmp_path / "logging" / "exports",
    }
    for name, p in redirects.items():
        monkeypatch.setitem(cfg._data["paths"], name, str(p))
    return tmp_path

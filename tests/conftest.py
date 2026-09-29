"""
Shared pytest fixtures.

The suite tests the INVARIANT / SECURITY surfaces the docs call load-bearing (the plan/execute
engine's data-bus invariants — test_engine.py, which replaced the deleted positional
plan-accounting walkers' tests in the 2026-07-03 transplant — the shell allowlist matcher, the
observation clamp, the surgical YAML persist, the snapshot/undo layer) plus the pure helpers
behind newer features (citations, RAG loaders, sessions). Everything runs offline: no test calls
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


@pytest.fixture(autouse=True)
def _reset_grant_lifecycle():
    """The always-allow grant lifecycle (trust/policy: task/session-scoped grants, task-boundary
    restorers, the grant log) is process-level state — clear it around every test so a grant
    made in one test can never exempt a command in another."""
    from trust import policy

    policy.reset_grants()
    yield
    policy.reset_grants()


@pytest.fixture(scope="session")
def _empty_saturn_home(tmp_path_factory):
    return tmp_path_factory.mktemp("saturn_home")


@pytest.fixture(autouse=True)
def _isolated_saturn_home(_empty_saturn_home, monkeypatch, tmp_path_factory):
    """The user's own ~/.saturn (the global SATURN.md, hooks.yaml) must never reach a test —
    a real hook would run commands mid-suite. Every test sees an EMPTY $SATURN_HOME; a test
    that needs files there points SATURN_HOME at its own tmp_path.

    HOME too: core/workspace falls back to the home folder for a launch folder that doesn't
    exist, so a fixture that forgot to create its folder wrote into the real ~ (2026-09-29).
    Every test gets a throwaway HOME; a test about home points HOME at its own tmp_path."""
    monkeypatch.setenv("SATURN_HOME", str(_empty_saturn_home))
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))


@pytest.fixture(autouse=True)
def _reset_workspace():
    """core/workspace holds the launch folder and the /add-dir folders as process state —
    clear it around every test so a root set in one test never leaks into another."""
    from core import workspace

    workspace.reset()
    yield
    workspace.reset()


@pytest.fixture(autouse=True)
def _no_prefix_priming(monkeypatch):
    """The idle prefix primes (core/prime.py) are real model requests fired from background
    threads at startup and after each turn — never under tests (no test may reach a model).
    A test that exercises priming flips the flag back and stubs core.llms.get_model."""
    from core import prime

    monkeypatch.setattr(prime, "ENABLED", False)

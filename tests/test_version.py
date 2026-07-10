"""The version is declared twice — `version` in pyproject.toml and `__version__` in
app/__init__.py (re-exported by agent.py) — synced by hand per the comment on each side.
This pins the pair so a drift fails on every push, not first at release time (the release
workflow additionally checks both against the git tag and for a CHANGELOG.md section)."""

import pathlib
import tomllib

from app import __version__

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__

"""
Minimal `.env` reader — the environment-variable lookup behind MCP's `${VAR}` expansion.

This used to be the managed API-key store behind `/config key` (ManagedKey registry, fuzzy
resolve, prefix detect, masked listings, set/unset with on_change hooks). That machinery was
CUT 2026-07-16: the registry had been EMPTY since the API-less web pivot (2026-07-06) — no
Saturn feature takes an API key (web search is keyless, inference is local, the Anthropic/
OpenAI keys left with the cloud shelve 2026-07-03) — so the picker managed nothing. Secrets
for MCP servers are plain env vars now: put them in `.env` next to the repo (or in
SATURDAY_HOME for wheel installs) or export them in the shell; `mcp.servers` `${VAR}` entries
read them through `get()` below. When a keyed provider returns, the managed registry returns
with it (the pre-cut module is in git history).

Imports nothing project-side, so it stays safe to import from anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


def _resolve_env_path() -> Path:
    # Clone mode: .env at the repo root (config.yaml — or, on a first run that hasn't seeded it
    # yet, the tracked template config.default.yaml — sits next to this file). Wheel installs
    # (pipx/uv) keep secrets in SATURDAY_HOME (~/.saturday) with the rest of the user's data,
    # mirroring config.py's lookup (kept in step by hand: no project imports here).
    root = Path(__file__).parent
    if (root / "config.yaml").exists() or (root / "config.default.yaml").exists():
        return root / ".env"
    return Path(os.environ.get("SATURDAY_HOME") or Path.home() / ".saturday") / ".env"


_ENV_PATH = _resolve_env_path()


def _file_values() -> dict[str, str]:
    """The `.env` file contents as a dict (empty if the file doesn't exist)."""
    if not _ENV_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(_ENV_PATH).items() if v is not None}


def get(name: str) -> Optional[str]:
    """The effective value: the live process environment wins over the on-disk `.env`."""
    return os.environ.get(name) or _file_values().get(name)


def is_set(name: str) -> bool:
    return bool(get(name))

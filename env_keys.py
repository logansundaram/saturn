"""
Minimal `.env` reader — the environment-variable lookup behind MCP's `${VAR}` expansion.

Secrets for MCP servers are plain env vars: put them in `.env` at the data root (the repo, or
~/.saturn for wheel installs) or export them in the shell; `mcp.servers` `${VAR}` entries read
them through `get()` below.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import dotenv_values

from diag import data_root

_ENV_PATH = data_root() / ".env"


def _file_values() -> dict[str, str]:
    """The `.env` file contents as a dict (empty if the file doesn't exist)."""
    if not _ENV_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(_ENV_PATH).items() if v is not None}


def get(name: str) -> Optional[str]:
    """The effective value: the live process environment wins over the on-disk `.env`."""
    return os.environ.get(name) or _file_values().get(name)

"""
Managed API-key store — the `.env`-backed secrets layer behind `/config key`.

`config.py` owns `config.yaml` (non-secret runtime settings); this owns the secrets in `.env`
at the repo root — the API keys the tools and providers read from the environment. They're kept
separate on purpose: secrets don't belong in `config.yaml` (which `/config` can echo and a future
`/save` could serialize), and they have different mechanics — persisted to `.env`, applied to
`os.environ` live, and masked whenever displayed.

Like the rest of the repo this imports nothing from the project (no circular-import risk), so it's
safe to import from `commands.py` or anywhere else.

Extensible by design
--------------------
To make a new key manageable, add one `ManagedKey` to `KNOWN_KEYS`:

    ManagedKey(
        name="OPENAI_API_KEY",            # the env var
        label="OpenAI",                   # human name for listings
        purpose="OpenAI-backed models",   # what it unlocks
        url="https://platform.openai.com/api-keys",
        on_change=_reset_models,          # drop any client that captured the old value
    )

The `/config key` command reads this registry to drive its listing, validation, and help — nothing
else changes when a provider is added. `on_change` is the hook that makes a key change take effect
without a restart: it should drop whatever cached the previous value (a client, a model handle).
Setting/unsetting an *unregistered* name still works (it's written to `.env` verbatim) but is
flagged as unmanaged, since no `on_change` hook can be run for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import dotenv_values, set_key, unset_key

_ENV_PATH = Path(__file__).parent / ".env"


# --- on_change hooks -------------------------------------------------------
# Lazy imports so this module stays dependency-free at import time and dodges circular imports.
def _reset_web_clients() -> None:
    from tool_registry.web import reset_web_clients

    reset_web_clients()


def _reset_models() -> None:
    from llms import reset_models

    reset_models()


@dataclass(frozen=True)
class ManagedKey:
    """A known API key the agent uses. `on_change` (optional) drops whatever cached the old value
    so a set/unset takes effect live."""

    name: str
    label: str
    purpose: str
    url: str = ""
    on_change: Optional[Callable[[], None]] = None


# The registry the /config key command renders and validates against. Add a row to expose a new key.
KNOWN_KEYS: tuple[ManagedKey, ...] = (
    ManagedKey(
        name="TAVILY_API_KEY",
        label="Tavily",
        purpose="upgrades the web tools (search/extract/research); optional — they fall back "
        "to keyless DuckDuckGo + local extraction without it",
        url="https://app.tavily.com/home",
        on_change=_reset_web_clients,
    ),
    ManagedKey(
        name="ANTHROPIC_API_KEY",
        label="Anthropic",
        purpose="cloud models for the cloud-hybrid tier (planner/synthesizer)",
        url="https://console.anthropic.com/settings/keys",
        on_change=_reset_models,
    ),
)


def find(name: str) -> Optional[ManagedKey]:
    """Look up a managed key by env-var name, case-insensitively."""
    upper = name.upper()
    for key in KNOWN_KEYS:
        if key.name.upper() == upper:
            return key
    return None


# --- read --------------------------------------------------------------------
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


def mask(value: Optional[str]) -> str:
    """A display-safe rendering of a secret: never the full value."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "****"
    return f"{value[:3]}…{value[-4:]}  ({len(value)} chars)"


# --- write -------------------------------------------------------------------
def set_value(name: str, value: str) -> None:
    """Persist a key to `.env`, apply it to the live environment, and run its `on_change` hook so
    the change takes effect this session without a restart."""
    name = name.upper()
    set_key(str(_ENV_PATH), name, value)
    os.environ[name] = value
    managed = find(name)
    if managed and managed.on_change:
        managed.on_change()


def unset_value(name: str) -> bool:
    """Remove a key from `.env` and the live environment, running its `on_change` hook. Returns
    True if it was set anywhere (file or environment)."""
    name = name.upper()
    existed = is_set(name)
    if _ENV_PATH.exists():
        unset_key(str(_ENV_PATH), name)
    os.environ.pop(name, None)
    managed = find(name)
    if managed and managed.on_change:
        managed.on_change()
    return existed

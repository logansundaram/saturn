"""
Slash-command layer for the interactive CLI loop (`agent.py`).

Converted from a single file to a package: each command lives in its own module under
commands/<name>.py, sharing the dispatch framework from commands._framework. Adding a new
command is just adding a new file — nothing else changes.

Public API (unchanged from the old commands.py):
  CommandContext, is_command, dispatch, command_completions, write_autosave
"""

from commands._framework import (
    CommandContext,
    SlashCommand,
    COMMANDS,
    command,
    is_command,
    dispatch,
    command_completions,
)
from commands._session import write_autosave

# Import each command module to trigger @command registration.
# Order doesn't matter for correctness; alphabetical for readability.
import importlib as _importlib

_COMMAND_MODULES = [
    "clear",
    "compact",
    "config",
    "context",
    "docs",
    "dryrun",
    "help",
    "init",
    "mcp",
    "memory",
    "models",
    "plan",
    "policy",  # also registers its top-level views: /risk, /allow, /autoapprove
    "privacy",
    "quit",
    "resume",
    "retry",
    "rewind",
    "source",
    "tools",
    "trace",  # also registers /glass (the front door for /trace answer)
    "undo",
    "update",
    "user_commands",
]

for _mod in _COMMAND_MODULES:
    _importlib.import_module(f"commands.{_mod}")

# Two-phase load: user-defined templates scan AFTER every built-in module above has registered,
# so the collision check ("a user file can never shadow a built-in") holds structurally instead
# of hanging on list order — appending a new built-in module anywhere above is always safe.
from commands.user_commands import load_user_commands as _load_user_commands

_load_user_commands()

del _importlib, _mod, _COMMAND_MODULES, _load_user_commands

__all__ = [
    "CommandContext",
    "SlashCommand",
    "COMMANDS",
    "command",
    "is_command",
    "dispatch",
    "command_completions",
    "write_autosave",
]

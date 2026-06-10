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
    "allow",
    "autoapprove",
    "clear",
    "compact",
    "config",
    "context",
    "docs",
    "help",
    "init",
    "load",
    "memory",
    "models",
    "plan",
    "privacy",
    "quit",
    "resume",
    "risk",
    "save",
    "system",
    "tools",
    "trace",
    "undo",
    "update",
    "workspace",
]

for _mod in _COMMAND_MODULES:
    _importlib.import_module(f"commands.{_mod}")

del _importlib, _mod, _COMMAND_MODULES

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

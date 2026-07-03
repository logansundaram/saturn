"""
Slash-command layer for the interactive CLI loop (`agent.py`).

A package of themed modules (consolidated 2026-06-11 from one-file-per-command): each module
under commands/ groups the commands of one /help theme and shares the dispatch framework from
commands._framework. Adding a new command is one @command-decorated handler in the module whose
theme fits (or a new module added to _COMMAND_MODULES below).

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
    "config",        # /config (+ key, setup) — owns the persist seam others import
    "conversation",  # /clear, /compact, /rewind, /retry, /resume
    "knowledge",     # /docs, /memory, /init, /undo
    "plan",          # /plan
    "policy",        # /policy + its views /risk, /allow, /autoapprove
    "privacy",       # /privacy
    "runtime",       # /tools, /models, /context, /mcp
    "system",        # /help, /quit, /update
    "trace",         # /trace; also /glass and /source (the provenance views)
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

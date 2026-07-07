"""
Shared utilities used by multiple command handlers.
"""
from __future__ import annotations

# All chat-model roles; used by /models and /config (incl. /config context). The canonical tuple lives in
# config.MODEL_ROLES (shared with llms.check_models and the locality readouts).
from config import MODEL_ROLES as _ROLES  # noqa: E402


def parse_toggle_status(args: list[str]) -> "bool | str | None":
    """THE on/off grammar for every status-or-set command (/policy open, /plan review,
    /privacy airgap): no argument -> None, a STATUS readout — bare is NEVER a
    flip, mutation is always an explicit verb; on/true/yes/1 -> True; off/false/no/0 -> False;
    anything else -> "invalid" (the caller prints usage). Trailing tokens are "invalid" too —
    `/policy open on garbage` must not open the gate on the strength of a half-parsed line —
    so every caller splits `--save` out (split_save_flag) BEFORE asking. One parser so the
    toggles can't drift apart."""
    if not args:
        return None
    if len(args) > 1:
        return "invalid"
    val = args[0].lower()
    if val in ("on", "true", "yes", "1"):
        return True
    if val in ("off", "false", "no", "0"):
        return False
    return "invalid"


def split_save_flag(args: list[str]) -> "tuple[list[str], bool]":
    """Split the standalone `--save` / `-s` persist flag out of `args`: case-insensitive, any
    position, exact token only. Returns (remaining args, flag present?). THE one --save parser —
    every command that persists a session edit to config.yaml reads the flag through this, so
    `/config context --save` and `/privacy airgap --save` can never disagree about what counts as
    the flag. Shared convention for `--save` with NO explicit value: persist the CURRENT value
    (it mutates nothing live, so it is safe everywhere) — never refuse, never flip."""
    rest = [a for a in args if a.lower() not in ("--save", "-s")]
    return rest, len(rest) != len(args)


# The persist-vs-session flag names for the SETTINGS commands (/config, /config context, /models).
_SESSION_FLAGS = ("--session", "--session-only", "--once")
_SAVE_FLAGS = ("--save", "-s")


def split_persist_flags(args: list[str]) -> "tuple[list[str], bool, bool]":
    """THE persist-vs-session grammar for the settings commands (/config, /config context,
    /models). These PERSIST to config.yaml BY DEFAULT — a setting a user changes should survive
    the next launch, which is what people expect from "change a setting"; the old session-only
    default forced a --save on every edit and silently forgot the rest. `--session` (aliases
    `--session-only`, `--once`) opts a single edit out: apply it live, don't write disk. `--save` /
    `-s` is still accepted (it's the default now) so old muscle memory and older docs keep working.

    Returns (remaining args, session_only?, save_seen?): `session_only` is what callers branch on;
    `save_seen` is consulted only for the bare `--save`-with-no-value "persist the current value"
    form. `--session` wins over a co-present `--save` (an explicit "don't write" is the safer read).
    Case-insensitive, any position, exact token only — same conventions as split_save_flag."""
    session = any(a.lower() in _SESSION_FLAGS for a in args)
    save = any(a.lower() in _SAVE_FLAGS for a in args)
    rest = [a for a in args if a.lower() not in _SESSION_FLAGS and a.lower() not in _SAVE_FLAGS]
    return rest, session, save


# THE removal-verb vocabulary, accepted identically by every command that deletes something
# (/docs, /memory, /resume, /policy allow, /config key ...). One set so muscle memory transfers;
# don't define a per-command subset.
REMOVE_VERBS = ("remove", "rm", "delete", "del", "forget", "drop")


def is_remove_verb(token: str) -> bool:
    """True when `token` is one of the shared removal verbs (case-insensitive)."""
    return token.lower() in REMOVE_VERBS


# THE listing-verb vocabulary (`git stash list` / `docker ls` style), accepted identically by
# every command that enumerates a collection (/docs, /memory, /resume, /models, /undo,
# /policy allow, /config key, /trace). Bare <command> stays the listing default everywhere —
# these are the explicit spellings, so neither habit errors. One set, like REMOVE_VERBS.
LIST_VERBS = ("list", "ls")


def is_list_verb(token: str) -> bool:
    """True when `token` is one of the shared listing verbs (case-insensitive)."""
    return token.lower() in LIST_VERBS


def _resync_rag_after_model_change() -> None:
    """Re-embed the corpus if the embedder changed after a model/tier switch."""
    from stores.rag import sync_to_config
    from commands._framework import _print

    if sync_to_config():
        _print("  embedder changed -> re-embedded the document corpus.")

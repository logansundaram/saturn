"""
Shared utilities used by multiple command handlers.
"""
from __future__ import annotations

# All chat-model roles; used by /models, /context, and /config. The canonical tuple lives in
# config.MODEL_ROLES (shared with llms.check_models and the signed trust report).
from config import MODEL_ROLES as _ROLES  # noqa: E402


def parse_toggle_status(args: list[str]) -> "bool | str | None":
    """THE on/off grammar for every status-or-set command (/dryrun, /policy open, /plan review,
    /plan lockstep, /privacy airgap): no argument -> None, a STATUS readout — bare is NEVER a
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
    `/plan lockstep --save` and `/privacy airgap --save` can never disagree about what counts as
    the flag. Shared convention for `--save` with NO explicit value: persist the CURRENT value
    (it mutates nothing live, so it is safe everywhere) — never refuse, never flip."""
    rest = [a for a in args if a.lower() not in ("--save", "-s")]
    return rest, len(rest) != len(args)


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

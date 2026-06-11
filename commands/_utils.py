"""
Shared utilities used by multiple command handlers.
"""
from __future__ import annotations

from typing import Optional

# All chat-model roles; used by /models, /context, and /config. The canonical tuple lives in
# config.MODEL_ROLES (shared with llms.check_models and the signed trust report).
from config import MODEL_ROLES as _ROLES  # noqa: E402


def _parse_toggle(args: list[str], current: bool) -> Optional[bool]:
    """Parse an on/off argument. No arg flips the current value; unrecognized returns None."""
    if not args:
        return not current
    val = args[0].lower()
    if val in ("on", "true", "yes", "1"):
        return True
    if val in ("off", "false", "no", "0"):
        return False
    return None


def _resync_rag_after_model_change() -> None:
    """Re-embed the corpus if the embedder changed after a model/tier switch."""
    from stores.rag import sync_to_config
    from commands._framework import _print

    if sync_to_config():
        _print("  embedder changed -> re-embedded the document corpus.")

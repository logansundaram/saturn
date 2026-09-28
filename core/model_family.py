"""
The size ladder — the recommended model per parameter size (2026-08-16; the family GATE that
shipped with it was cut 2026-09-27).

One tier per size class, each binding the most advanced qwen3.5 / 3.6 / 3.8 tag at that size,
so a version bump is a one-line edit here and in config.default.yaml and the tier keys never
churn. The ladder is what `/models` prices against the machine and what a fresh install pulls;
it is NOT a restriction — any Ollama model with native tool-calling binds through `/models` or
`/config`, and `class_of` prices it by the parameter count in its tag.

LEAF: stdlib only, no project imports, so config.py / core/llms.py / commands/ may all depend on
it without a cycle.
"""

from __future__ import annotations

import re

# size class -> the recommended tag at that size. Tags are stored VERBATIM: Ollama tags are
# case-sensitive and the 0.8B tag really does carry a capital B. The class KEY is "800m", not
# "0.8b" — config.get/set/persist parse dotted paths, so a "." inside a tier key silently splits
# it into two segments and corrupts a role bind (fixed 2026-08-16; see
# tests/test_model_family.py::test_no_size_class_key_contains_a_dot).
SIZE_LADDER: tuple[tuple[str, str], ...] = (
    ("800m", "qwen3.5:0.8B"),
    ("2b", "qwen3.5:2b"),
    ("4b", "qwen3.5:4b"),
    ("9b", "qwen3.5:9b"),
    ("27b", "qwen3.8:27b"),
    ("35b", "qwen3.6:35b"),
)

# The embedder ladder (2026-09-01): the qwen3-embedding family, one tag per size, smallest first.
# /models offers exactly these — the same "one recommended tag per size" rule as the chat
# ladder, so the listing never reads as `ollama list`.
EMBEDDER_LADDER: tuple[tuple[str, str], ...] = (
    ("0.6b", "qwen3-embedding:0.6b"),
    ("4b", "qwen3-embedding:4b"),
    ("8b", "qwen3-embedding:8b"),
)

# The fresh-install tier. Small on purpose: the first pull should be light.
DEFAULT_CLASS = "4b"

# Real parameter counts (billions, from `ollama show`) — the nearest-size yardstick class_of
# prices an off-ladder tag with.
_CLASS_PARAMS: dict[str, float] = {
    "800m": 0.87, "2b": 2.3, "4b": 4.7, "9b": 9.7, "27b": 27.3, "35b": 36.0,
}

# `:30b`, `:e4b`, `:0.8b` — the parameter count Ollama bakes into a tag.
_SIZE_RE = re.compile(r":e?(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def is_ladder_tag(model_id) -> bool:
    """Whether `model_id` is one of the tags the ladder binds (case-insensitively). Callers that
    want the shipped defaults for a tag they know we ship ask this."""
    want = str(model_id or "").strip().lower()
    return any(tag.lower() == want for _key, tag in SIZE_LADDER)


def classes() -> tuple[str, ...]:
    """The size-class keys, smallest first — the tier names and what /models tier accepts."""
    return tuple(key for key, _tag in SIZE_LADDER)


def tag_for(size_class) -> str:
    """The model id a size class binds. Raises KeyError for an unknown class."""
    want = str(size_class or "").strip().lower()
    for key, tag in SIZE_LADDER:
        if key == want:
            return tag
    raise KeyError(
        f"unknown size class {size_class!r} — defined: {', '.join(classes())}"
    )


def class_of(model_id) -> str:
    """The size class whose hardware cost a tag carries: the class nearest the parameter count
    parsed out of its tag (a ladder tag is its own class, since every ladder size IS a class
    key; `gemma4:31b` costs what 27b costs), and DEFAULT_CLASS for a tag that names no size.
    Returns a class KEY — callers compose tag_for(class_of(id)) for the ladder equivalent."""
    name = str(model_id or "").strip().lower()
    found = _SIZE_RE.search(name)
    if found:
        try:
            want = float(found.group(1))
        except ValueError:
            want = None
        if want is not None:
            return min(_CLASS_PARAMS, key=lambda key: abs(_CLASS_PARAMS[key] - want))
    return DEFAULT_CLASS


def embedder_classes() -> tuple[str, ...]:
    """The embedder ladder's size keys, smallest first."""
    return tuple(key for key, _tag in EMBEDDER_LADDER)


def embedder_tag_for(size_class) -> str:
    """The embedding model id an embedder size class binds. Raises KeyError for an unknown one."""
    want = str(size_class or "").strip().lower()
    for key, tag in EMBEDDER_LADDER:
        if key == want:
            return tag
    raise KeyError(
        f"unknown embedder class {size_class!r} — defined: {', '.join(embedder_classes())}"
    )


def embedder_class_of(model_id) -> "str | None":
    """The embedder ladder key a tag belongs to (case-insensitive), or None off the ladder."""
    want = str(model_id or "").strip().lower()
    for key, tag in EMBEDDER_LADDER:
        if tag.lower() == want:
            return key
    return None

"""
The supported model family (2026-08-16).

Saturday.ai binds exactly ONE model family — qwen3.5 / qwen3.6 / qwen3.8 — because confidence
coloring is calibrated PER MODEL (core/confidence_calibration.py): a red run means "worse than
95 % of THIS model's clean output", which is only a true claim for a model that was actually
measured. Every other family was removed rather than shipped with a threshold borrowed from a
model of a different size (the confidence_coloring isolate measured a 6-10x spread across sizes).

The ladder carries the most advanced tag at each parameter size, so a version bump is a one-line
edit here and in config.default.yaml — the tier keys (size classes) never churn.

LEAF: stdlib only, no project imports, so config.py / core/llms.py / commands/ may all depend on
it without a cycle.
"""

from __future__ import annotations

import re

# The family, as bare prefixes. core/chat_template.py carries the same three for raw-mode
# continuation; tests/test_model_family.py asserts the two lists agree.
FAMILY_PREFIXES: tuple[str, ...] = ("qwen3.5", "qwen3.6", "qwen3.8")

# ANCHORED on purpose: a loose startswith() would let a future `qwen3.50` satisfy a `qwen3.5`
# test and bind an uncalibrated model through the gate.
_FAMILY_RE = re.compile(r"^(?:qwen3\.5|qwen3\.6|qwen3\.8)(?::|$)", re.IGNORECASE)

# size class -> the most advanced family tag at that size. Tags are stored VERBATIM: Ollama tags
# are case-sensitive and the 0.8B tag really does carry a capital B. The class KEY is "800m", not
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

# The fresh-install tier, and where an unrecognizable legacy binding lands. Small on purpose:
# the first pull should be light, and migrating DOWN never exceeds the machine's VRAM.
DEFAULT_CLASS = "4b"

# Real parameter counts (billions, from `ollama show`) — the nearest-size fallback's yardstick.
_CLASS_PARAMS: dict[str, float] = {
    "800m": 0.87, "2b": 2.3, "4b": 4.7, "9b": 9.7, "27b": 27.3, "35b": 36.0,
}

# Exact substitutions for the tags that shipped before the lock. The size parse below would
# reach the same answer for all of them; the table is here so the mapping is DECLARED rather
# than inferred, and so a rename upstream can't silently move somebody's binding.
_LEGACY: dict[str, str] = {
    "gemma4:e2b": "2b",
    "gemma4:e4b": "4b",
    "gemma4:12b": "9b",
    "gemma4:26b": "27b",
    "gemma4:31b": "27b",
    "qwen3-coder:30b": "27b",
}

# `:30b`, `:e4b`, `:0.8b` — the parameter count Ollama bakes into a tag.
_SIZE_RE = re.compile(r":e?(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def in_family(model_id) -> bool:
    """Whether `model_id` is a supported chat model. The EMBEDDER is exempt from this test —
    it is not a chat model, has no raw-mode template and produces no logprobs."""
    return bool(_FAMILY_RE.match(str(model_id or "").strip()))


def is_ladder_tag(model_id) -> bool:
    """Whether `model_id` is one of the tags the ladder actually binds (case-insensitively) —
    narrower than in_family, which accepts any tag of the three families. Callers that want the
    shipped defaults for a tag they know we ship ask this."""
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


def migrate(model_id) -> str:
    """The size class a NON-family binding is substituted with. Deterministic: the declared
    legacy table first, then the parameter count parsed out of the tag mapped to the nearest
    class, then DEFAULT_CLASS. Returns a class KEY — callers compose tag_for(migrate(id))."""
    name = str(model_id or "").strip().lower()
    if name in _LEGACY:
        return _LEGACY[name]
    found = _SIZE_RE.search(name)
    if found:
        try:
            want = float(found.group(1))
        except ValueError:
            want = None
        if want is not None:
            return min(_CLASS_PARAMS, key=lambda key: abs(_CLASS_PARAMS[key] - want))
    return DEFAULT_CLASS

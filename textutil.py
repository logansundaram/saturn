"""
Shared text-shaping primitives — the one home for the ellipsis-truncation idiom.

Before this module the `s[: n - 1] + "…"` pattern was hand-rolled in a dozen places (trace
previews, plan labels, steer notes, arg reprs, recap lines — deferred-review #5). Every layer may
import it: it is a leaf with no project imports, so there is no circular-import risk from nodes,
tools, stores, commands, or the TUI.
"""

from __future__ import annotations

import re
from pathlib import PureWindowsPath

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def truncate(s: str, n: int) -> str:
    """`s` capped at `n` chars total; a cut is marked with a trailing ellipsis."""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def head_tail(text: str, cap: int, marker: "str | None" = None) -> str:
    """Head+tail elision for text over `cap` chars: keep the first 2/3 and the last 1/3 with an
    explicit marker noting how many characters were dropped — the head usually carries the
    intent, the tail is where a long payload hides the part that matters, so neither end is
    silently cut. THE one home for the head+tail idiom (tool observations, the gate's full-width
    argument view — hand-rolled copies drift on the split math and the marker). `marker` is a
    format template receiving `dropped` (the elided character count); None uses the compact
    ellipsis form. Text at or under `cap` is returned unchanged (the same object, so identity
    checks on the passthrough hold)."""
    text = str(text)
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    dropped = len(text) - cap
    if marker is None:
        marker = "\n… [truncated {dropped} characters] …\n"
    return text[:head] + marker.format(dropped=dropped) + text[-tail:]


def clip(s, n: int) -> str:
    """One-line preview: collapse all whitespace runs to single spaces, then truncate to `n`."""
    return truncate(" ".join(str(s or "").split()), n)


def human_bytes(n) -> str:
    """Byte count as a compact human label (512B, 2.0KB, 3.4MB) — ONE formatter so every trust
    surface (the per-answer receipt, /privacy egress, the durable-log view) renders the same
    number the same way. Tolerates None/junk (reads as 0)."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def fmt_args(args: dict, cap: int) -> str:
    """Render a tool-call kwargs dict as `k='v', k2=3, …` with each value's repr capped, so one
    fat payload (a write_file body) can't bloat a trace line or approval prompt."""
    return ", ".join(f"{k}={truncate(repr(v), cap)}" for k, v in (args or {}).items())


def iter_strings(value):
    """Every string leaf inside a nested dict/list/tuple value (dict KEYS and scalars skipped —
    neither can carry a secret worth scanning). THE one walker over a tool call's argument tree:
    the gate's secret scan (redaction.scan_args) and the MCP boundary's redaction both use it, so
    they can never disagree about what counts as argument content."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_strings(v)


def map_strings(value, fn):
    """Structure-preserving rewrite of every string leaf inside a nested dict/list/tuple value —
    the REWRITE twin of `iter_strings`, visiting exactly the same leaves (dict KEYS and
    non-string scalars untouched), so a scan and a rewrite over the same tree can never disagree
    about what counts as content (the MCP boundary's warn-mode count vs redact-mode rewrite).
    Tuples come back as lists — every consumer serializes toward JSON, which has none."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [map_strings(v, fn) for v in value]
    return value


# The separator nodes/tools.py mirrors each executed call into `tool_results` with
# (`f"{call_repr}{CALL_RESULT_SEP}{observation}"`). One constant + one parser so synthesize's
# Sources labels recover the call half (the observation half) the same way every time.
CALL_RESULT_SEP = " -> "


def split_call_result(entry) -> "tuple[str, str]":
    """Split one mirrored tool-result entry into (call_repr, observation) — THE one parser of
    the `name(args) -> observation` serialization built in nodes/tools.py. Known edge, shared
    with the construction site: an argument VALUE containing the separator splits early, leaving
    model-authored text in the observation half — the consumer fails toward caution on it (a
    mislabeled source at worst; never dropped observation content). An entry with no separator
    returns (entry, entry): the whole string is the best available answer to either question."""
    parts = str(entry).split(CALL_RESULT_SEP, 1)
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


# The `[source: name, page N]` provenance marker search_knowledge_base prepends to each
# retrieved chunk. One builder + one parser (tools/knowledge.py builds it, nodes/synthesize.py's
# Sources labels parse it back) so the two sides can't drift — the CALL_RESULT_SEP treatment.
DOC_SOURCE_RE = re.compile(r"\[source: ([^\]]+)\]")


def doc_source_label(name, page=None) -> str:
    """Render the provenance marker for one retrieved chunk."""
    inner = str(name or "unknown")
    if page:
        inner += f", page {page}"
    return f"[source: {inner}]"


def parse_doc_sources(text) -> "list[str]":
    """The distinct marker names inside one retrieval observation, in first-seen order."""
    names: list[str] = []
    for m in DOC_SOURCE_RE.finditer(str(text)):
        name = m.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def mask_secret(value) -> str:
    """A display-safe preview of a secret — THE one masking rule (env_keys' key listing and
    trust/redaction's findings each hand-rolled their own, with different exposure envelopes:
    3+4 vs 6+2 visible characters, and a short secret partially shown on one surface but fully
    masked on the other; a tightening decision made once must reach both). ≤8 chars shows
    nothing; longer shows the first 4 + last 2."""
    s = " ".join(str(value or "").split())
    if not s:
        return ""
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}…{s[-2:]}"


def safe_stem(name, fallback: str) -> str:
    """Sanitize a user-supplied name to a safe filename stem: path parts dropped, a trailing
    `.json` stripped (so `brief.json` and `brief` resolve identically), every other special
    character collapsed to `-`. THE one sanitizer for user-named JSON artifacts (sessions
    today; any future user-named store) — drifted copies would enforce different naming
    rules per store."""
    # PureWindowsPath, not PurePath: it splits on BOTH `/` and `\`, so `..\..\evil` loses its
    # path bits on every platform (PurePosixPath keeps backslashes as filename characters).
    stem = PureWindowsPath(str(name)).name
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    return _SAFE_STEM.sub("-", stem).strip("-_") or fallback

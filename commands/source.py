"""
/source — drill into the exact material behind a citation [n] of the last answer.

The citations footer maps each inline [n] to a one-line label; this shows the FULL tool result /
retrieved passage behind that number, rebuilt with the same numbering the synthesizer saw
(node_registry.synthesize.build_sources over the turn's accumulators), so [3] here is exactly the
[3] in the answer. Closes the provenance loop in one keystroke instead of a /trace drill-down.
"""

from __future__ import annotations

from commands._framework import command, _print


def lookup_source(state: dict, n: int) -> "tuple[str, str] | None":
    """(label, full_text) for citation number `n` of the last turn, or None when out of range.
    Pure over the state accumulators so it's testable without a turn."""
    from node_registry.synthesize import build_sources

    tool_results = (state or {}).get("tool_results") or []
    docs = (state or {}).get("documents_retrieved") or []
    numbered_tools, numbered_docs, sources = build_sources(tool_results, docs)
    entries = numbered_tools + numbered_docs
    if not (1 <= n <= len(entries)):
        return None
    label = sources[n - 1][1]
    # Strip the `[n] ` numbering prefix build_sources added for the prompt.
    text = entries[n - 1]
    prefix = f"[{n}] "
    if text.startswith(prefix):
        text = text[len(prefix):]
    return label, text


@command(
    "source",
    "Show the full material behind a citation [n] of the last answer.",
    aliases=("sources",),
    usage="/source [n]",
    details="""
Answers cite their evidence inline ([1], [2], …) with a Sources footer mapping each number to the
tool call or document behind it. This command shows the FULL text behind a number — the complete
tool observation or retrieved passage the synthesizer actually read — using the same numbering
the answer used.

  /source        list the last answer's sources (numbered labels)
  /source 3      print everything behind citation [3]

Scope: the most recent turn (the accumulators reset when a new turn starts; /clear empties them).
For older runs, /trace #<id> replays the full tool I/O of any recorded run.
""",
)
def _source(ctx, args):
    from node_registry.synthesize import build_sources

    state = ctx.state or {}
    tool_results = state.get("tool_results") or []
    docs = state.get("documents_retrieved") or []
    _, _, sources = build_sources(tool_results, docs)

    if not sources:
        _print("  (the last answer drew on no gathered sources — nothing to cite)")
        return

    if not args:
        _print("  sources of the last answer  (/source <n> for the full text):")
        for n, label in sources:
            _print(f"    [{n}] {label}")
        return

    try:
        n = int(args[0].lstrip("[").rstrip("]"))
    except ValueError:
        _print(f"  usage: /source [n]   (n is a citation number, 1–{len(sources)})")
        return

    found = lookup_source(state, n)
    if found is None:
        _print(f"  no source [{n}] — the last answer has {len(sources)} source(s); /source lists them.")
        return
    label, text = found
    _print(f"  [{n}] {label}")
    _print("")
    for line in text.splitlines() or [""]:
        _print(f"  {line}")
    _print("")

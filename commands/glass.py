"""
/glass — the Glass Box: answer-level provenance. A top-level alias for `/trace answer`, given its
own name because it's a flagship surface (trust the ANSWER, the companion to /trace's trust-the-
PROCESS). All the logic lives in commands/trace.py::_answer; this is the brandable front door.
"""

from commands._framework import command


@command(
    "glass",
    "The Glass Box: answer-level provenance (origin · trust · what left · tainted spans).",
    aliases=("glassbox",),
    usage="/glass [#id]",
    details="""
The answer-level companion to /trace. For one answer it shows, per cited source: where it came
from (local disk vs the network), whether its origin is trusted, and — the headline — whether a
span of an untrusted source appears VERBATIM in the answer (the data→answer channel, colored red
inline). Plus the trust label: how many sources, what left the machine this turn, whether the
groundedness judge weighed in, and how many calls faced the approval gate.

  /glass         the live last turn (exact egress slice)
  /glass #7      reconstruct run #7 from the trace record

Identical to `/trace answer`. Scope: bare reads the live last turn (the accumulators reset when a
new turn starts); #id reconstructs any recorded run (egress is inferred from the source tools, the
one facet history can't carry exactly).
""",
)
def _glass(ctx, args):
    from commands.trace import _answer

    return _answer(ctx, args)

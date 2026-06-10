"""
Tool registration primitive — `@register_tool`.

A tool declares ALL of its own metadata at definition time: it is wrapped as a LangChain tool and
registered (added to the active list, given a risk tier for the approval gate, flagged if its
output is a retrieved document) in ONE place — its own module — instead of being defined here and
then re-listed in three more (`registry.tool`, `registry.TOOL_RISK`, the retrieval set). Adding a
tool is now a single edit; nothing in `registry.py` changes.

This lives apart from `registry.py` on purpose: `registry.py` imports the tool modules to trigger
their registration, so if the decorator lived there the tool modules would import back into a
half-initialised `registry` (a circular import). This module imports NOTHING from the project, so
the tool modules can import it freely and the cycle never forms. `registry.py` then reads the
collected views below and re-exports them under their established names.
"""

from __future__ import annotations

from langchain.tools import tool as _lc_tool

# Risk tiers, low -> high. Mirrors config.RISK_ORDER; duplicated here only because BOTH modules
# are project-import-free leaves (neither may import the other) — everything else imports the
# tiers from one of the two (e.g. /risk reads this one). A tool runs without prompting iff its
# tier is at or below the configured `runtime.auto_approve` tier (see node_registry/approval.py).
RISK_TIERS = ("read_only", "side_effecting", "destructive")

# Collected at import time as each tool module's @register_tool runs. registry.py re-exports these.
_TOOLS: list = []          # the active tool objects, in registration order
_RISK: dict = {}           # tool name -> risk tier
_RETRIEVAL: set = set()    # tool names whose results are recorded as retrieved documents


def register_tool(risk: str = "destructive", *, retrieval: bool = False):
    """Decorator: wrap a function as a LangChain tool AND register it (list + risk tier + retrieval
    flag) in one place.

      @register_tool("read_only")                      # runs without prompting
      @register_tool("side_effecting")                 # hits the approval gate
      @register_tool("read_only", retrieval=True)      # output recorded as a retrieved document

    `risk` must be one of RISK_TIERS; it defaults to the safe 'destructive' tier (always prompts)
    so a tool that forgets to declare one fails closed. `retrieval=True` marks tools whose output
    is a document worth recording for citations/trace (e.g. search_knowledge_base)."""
    if risk not in RISK_TIERS:
        raise ValueError(f"unknown risk tier {risk!r}; expected one of {RISK_TIERS}")

    def decorate(fn):
        t = _lc_tool(fn)
        _TOOLS.append(t)
        _RISK[t.name] = risk
        if retrieval:
            _RETRIEVAL.add(t.name)
        return t

    return decorate

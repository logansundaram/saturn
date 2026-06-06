# The active tool registry. Tools register THEMSELVES via @register_tool (see toolspec.py) — their
# risk tier and retrieval flag live with the tool, not in a parallel table here. This module just
# imports the grouped tool modules (which triggers their registration) and re-exports the collected
# views under the names the rest of the codebase already imports: `tool` (the list), `tools_by_name`,
# `TOOL_RISK`, `risk_of`, and `RETRIEVAL_TOOLS`.
#
# To add a tool: write the @tool function in the right tool_registry/ module and decorate it with
# @register_tool(<risk>[, retrieval=True]). Nothing in this file changes.

from toolspec import _TOOLS, _RISK, _RETRIEVAL  # collected as the imports below run

# Importing each module runs its @register_tool decorators, populating the toolspec collections.
# Import order here is purely cosmetic — it sets the order the planner lists tools in.
from tool_registry.calculator import calculate  # noqa: E402,F401
from tool_registry.web import web_search, web_extract, deep_research  # noqa: E402,F401
from tool_registry.files import read_file, write_file, list_directory  # noqa: E402,F401
from tool_registry.knowledge import search_knowledge_base  # noqa: E402,F401
from tool_registry.memory import remember, recall  # noqa: E402,F401

# --- collected views (established public names) ---------------------------------------------
tool = _TOOLS                      # the active tool list (bound to the agent, listed by the planner)
tools_by_name = {t.name: t for t in tool}
TOOL_RISK = _RISK                  # name -> risk tier; mutable — the /risk command edits this live
RETRIEVAL_TOOLS = _RETRIEVAL       # names whose results are recorded as retrieved documents


# Risk tiers drive the approval gate (see node_registry/approval.py):
#   read_only      — no side effects; runs freely, never prompts
#   side_effecting — writes/external actions; prompts for approval
#   destructive    — irreversible/dangerous; prompts for approval
# A tool's tier is declared at its definition via @register_tool; unknown names fail safe.
def risk_of(tool_name: str) -> str:
    """Risk tier for a tool name; unknown tools default to the safe 'destructive' tier (always
    prompts)."""
    return TOOL_RISK.get(tool_name, "destructive")

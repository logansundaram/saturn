# The active tool registry. Tools register THEMSELVES via @register_tool (see toolspec.py) — their
# risk tier and retrieval flag live with the tool, not in a parallel table here. This module just
# imports the grouped tool modules (which triggers their registration) and re-exports the collected
# views under the names the rest of the codebase already imports: `tool` (the list), `tools_by_name`,
# `TOOL_RISK`, `risk_of`, and `RETRIEVAL_TOOLS`.
#
# To add a tool: write the @tool function in the right tools/ module and decorate it with
# @register_tool(<risk>[, retrieval=True]). Nothing in this file changes.

from tools.toolspec import _TOOLS, _RISK, _RETRIEVAL  # collected as the imports below run

# Importing each module runs its @register_tool decorators, populating the toolspec collections.
# Module imports on purpose (not per-name): registration needs the module to RUN, not its names,
# so a new tool in an existing module truly requires no edit here. Import order is purely
# cosmetic — it sets the order the planner lists tools in.
import tools.calculator  # noqa: E402,F401
import tools.clock  # noqa: E402,F401
import tools.web  # noqa: E402,F401
import tools.files  # noqa: E402,F401
import tools.knowledge  # noqa: E402,F401
import tools.memory  # noqa: E402,F401
import tools.shell  # noqa: E402,F401

# Remote MCP tools (roadmap #12): connect the servers declared under `mcp.servers` in config.yaml
# and register each remote tool through toolspec.register_tool_object, so they land in the same
# collections as the local tools above — same gate, same /tools, same planner catalog. Runs HERE,
# after the local registrations (collisions resolve in the local tools' favour) and BEFORE the
# persisted /risk overrides below (so a saved override on an MCP tool name applies). Every MCP
# tool fails closed to `destructive` unless the user's own config/overrides relax it. No servers
# configured -> no-op. Failures are recorded (mcp_client.problems(), warned at startup) — never
# raised, so a bad server entry can't take the app down.
from tools import mcp_client  # noqa: E402

mcp_client.startup()

# --- collected views (established public names) ---------------------------------------------
tool = _TOOLS                      # the active tool list (bound to the agent, listed by the planner)
tools_by_name = {t.name: t for t in tool}
TOOL_RISK = _RISK                  # name -> risk tier; mutable — the /risk command edits this live
RETRIEVAL_TOOLS = _RETRIEVAL       # names whose results are recorded as retrieved documents

# The tiers as declared at definition time, frozen BEFORE the persisted overrides apply — this is
# what `/risk <tool> reset` restores to.
DECLARED_RISK = dict(_RISK)

# Apply the user's persisted /risk overrides (policy.py — the gate-policy object) over the
# declared tiers, so a `/risk … --save` decision survives a restart. Stale names (a removed tool)
# and invalid tiers are ignored — the declared tier, which fails closed, stays in effect.
from tools.toolspec import RISK_TIERS as _RISK_TIERS  # noqa: E402
from trust import policy as _policy  # noqa: E402

for _name, _tier in _policy.risk_overrides().items():
    if _name in tools_by_name and _tier in _RISK_TIERS:
        TOOL_RISK[_name] = _tier
del _policy, _RISK_TIERS


# Risk tiers drive the approval gate (see nodes/approval.py):
#   read_only      — no side effects; runs freely, never prompts
#   side_effecting — writes/external actions; prompts for approval
#   destructive    — irreversible/dangerous; prompts for approval
# A tool's tier is declared at its definition via @register_tool; unknown names fail safe.
def risk_of(tool_name: str) -> str:
    """Risk tier for a tool name; unknown tools default to the safe 'destructive' tier (always
    prompts)."""
    return TOOL_RISK.get(tool_name, "destructive")

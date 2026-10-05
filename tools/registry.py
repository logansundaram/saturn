# The active tool registry. Tools register THEMSELVES via @register_tool (see toolspec.py) — their
# risk tier and retrieval flag live with the tool, not in a parallel table here. This module just
# imports the grouped tool modules (which triggers their registration) and re-exports the collected
# views under the names the rest of the codebase imports: `tool` (the list), `tools_by_name`,
# `TOOL_RISK`, `risk_of`, and `RETRIEVAL_TOOLS`.
#
# To add a tool: write the @tool function in the right tools/ module and decorate it with
# @register_tool(<risk>[, retrieval=True]). Nothing in this file changes.

from tools.toolspec import _TOOLS, _RISK, _RETRIEVAL  # collected as the imports below run

# Importing each module runs its @register_tool decorators, populating the toolspec collections.
# Module imports on purpose (not per-name): registration needs the module to RUN, not its names,
# so a new tool in an existing module truly requires no edit here. Import order is purely
# cosmetic — it sets the order the tools bind in.
import tools.calculator  # noqa: E402,F401  (calculate + current_time)
import tools.web  # noqa: E402,F401
import tools.files  # noqa: E402,F401
import tools.knowledge  # noqa: E402,F401  (search_knowledge_base + remember/recall)
import tools.shell  # noqa: E402,F401
import tools.interaction  # noqa: E402,F401  (ask_user — the mid-run question to the human)
import tools.planning  # noqa: E402,F401  (plan — the model's own checklist, mapped onto state by nodes/tools.py)
import tools.notify  # noqa: E402,F401  (schedule_notification — a one-shot OS-scheduled reminder)
import tools.notes  # noqa: E402,F401  (search_notes / read_note / create_note — Apple Notes, macOS)
import tools.calendar  # noqa: E402,F401  (list_calendar_events / create_calendar_event — Apple Calendar, macOS)
import tools.mail  # noqa: E402,F401  (list_mail / search_mail / read_mail / draft_mail / reply_mail / update_mail — Apple Mail, macOS; drafts only, never sends)
import tools.contacts  # noqa: E402,F401  (search_contacts — Apple Contacts, macOS)
import tools.reminders  # noqa: E402,F401  (list_reminders / create_reminder / complete_reminder — Apple Reminders, macOS)
import tools.shortcuts  # noqa: E402,F401  (list_shortcuts / run_shortcut — the user's Shortcuts, macOS; runs recorded UNTRACKED)
import tools.desktop  # noqa: E402,F401  (read_browser_tab / finder_selection — what the user is pointing at, macOS)
import tools.messages  # noqa: E402,F401  (send_message — an egress chokepoint that always asks, to one person or one group chat; read_messages — chat.db, needs Full Disk Access; find_group_chats)
import tools.skills  # noqa: E402,F401  (create_skill — saves one of the user's skills; always asks)

# Remote MCP tools: connect the servers declared under `mcp.servers` in config.yaml
# and register each remote tool through toolspec.register_tool_object, so they land in the same
# collections as the local tools above — same gate, same /tools, same catalog. Runs HERE,
# after the local registrations (collisions resolve in the local tools' favour) and BEFORE the
# persisted /policy risk overrides below (so a saved override on an MCP tool name applies). Every MCP
# tool fails closed to `destructive` unless the user's own config/overrides relax it. No servers
# configured -> no-op. Failures are recorded (mcp_client.problems(), warned at startup) — never
# raised, so a bad server entry can't take the app down.
from tools import mcp_client  # noqa: E402

mcp_client.startup()

# --- collected views (established public names) ---------------------------------------------
tool = _TOOLS                      # the active tool list (bound to the agent)
tools_by_name = {t.name: t for t in tool}
TOOL_RISK = _RISK                  # name -> risk tier; mutable — /policy risk edits this live
RETRIEVAL_TOOLS = _RETRIEVAL       # names whose results are recorded as retrieved documents

# The tiers as declared at definition time, frozen BEFORE the persisted overrides apply — this is
# what `/policy risk <tool> reset` restores to.
DECLARED_RISK = dict(_RISK)

# Apply the user's persisted /policy risk overrides (policy.py — the gate-policy object) over the
# declared tiers, so a `/policy risk … --save` decision survives a restart. Stale names (a removed tool)
# and invalid tiers are ignored — the declared tier, which fails closed, stays in effect.


def apply_risk_overrides() -> None:
    """Lay the persisted overrides over the declared tiers. A NO_BLANKET_GRANT tool (run_shell,
    run_shortcut, a send) keeps its declared tier whatever the file says: an override there
    would un-gate every shell command with no allowlist (policy.set_risk_override refuses to
    write one; a hand-edited file is ignored here)."""
    from trust import policy as _policy
    from tools.toolspec import RISK_TIERS as _RISK_TIERS

    for _name, _tier in _policy.risk_overrides().items():
        if _name in _policy.NO_BLANKET_GRANT:
            TOOL_RISK[_name] = DECLARED_RISK.get(_name, "destructive")
        elif _name in tools_by_name and _tier in _RISK_TIERS:
            TOOL_RISK[_name] = _tier


apply_risk_overrides()


def refresh_trust_classifications() -> None:
    """Push the tool trust classifications into the quarantine scanner (a leaf that must not
    import this registry): which tools' output crosses the trust boundary (declared at
    registration via `untrusted=True`), and which gated tool names the tool-coercion injection
    pattern should recognize. The coercion set is the UNION of declared and live non-read_only
    tiers — monotone toward scanning: a user relaxing a tool's gate (/policy risk read_only, an
    always-allow grant) must never shrink the injection scan, while raising a tier still adds
    the name. Runs at import below and again after anything that changes the tool set or the
    live tiers (/mcp reload, /policy risk)."""
    from tools.toolspec import _UNTRUSTED
    from trust import quarantine

    quarantine.set_untrusted_tools(_UNTRUSTED)
    gated = {n for n, t in DECLARED_RISK.items() if t != "read_only"}
    gated |= {n for n, t in TOOL_RISK.items() if t != "read_only"}
    quarantine.set_gated_tools(gated)


refresh_trust_classifications()


def is_action(name: str) -> bool:
    """Whether a completed call to `name` is an ACTION — it changed something and returned a
    confirmation, not material: a tool declared side_effecting, or a destructive one whose
    output is not external (a send, a delete; run_shell and MCP tools return output to read).
    One rule, two readers: the Sources receipt (nodes/tools.py — an action is never cited) and
    the think decision (core/think.step_kind — the pass after a round of actions is the
    wrap-up)."""
    from trust import quarantine

    declared = DECLARED_RISK.get(name)
    return declared == "side_effecting" or (
        declared == "destructive" and not quarantine.is_untrusted(name))


# Risk tiers drive the approval gate (see nodes/approval.py):
#   read_only      — no side effects; runs freely, never prompts
#   side_effecting — writes/external actions; prompts for approval
#   destructive    — irreversible/dangerous; prompts for approval
# A tool's tier is declared at its definition via @register_tool; unknown names fail safe.
def risk_of(tool_name: str) -> str:
    """Risk tier for a tool name; unknown tools default to the safe 'destructive' tier (always
    prompts)."""
    return TOOL_RISK.get(tool_name, "destructive")

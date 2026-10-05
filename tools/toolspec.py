"""
Tool registration primitive — `@register_tool`.

A tool declares ALL of its own metadata at definition time: it is wrapped as a LangChain tool and
registered (added to the active list, given a risk tier for the approval gate, flagged if its
output is a retrieved document) in ONE place — its own module. Adding a tool is a single edit;
nothing in `registry.py` changes. Registration also wraps the function with per-call timing to
`diag.log`.

This lives apart from `registry.py` on purpose: `registry.py` imports the tool modules to trigger
their registration, so if the decorator lived there the tool modules would import back into a
half-initialised `registry` (a circular import). This module imports nothing project-side except
the `diag` leaf (which itself imports nothing), so the tool modules can import it freely and the
cycle never forms. `registry.py` then reads the collected views below and re-exports them under
their established names.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from functools import wraps

from langchain.tools import tool as _lc_tool

import diag

# Risk tiers, low -> high. Mirrors config.RISK_ORDER; duplicated here only because BOTH modules
# are project-import-free leaves (neither may import the other) — everything else imports the
# tiers from one of the two (e.g. /policy risk reads this one). A tool runs without prompting iff its
# tier is at or below the configured `runtime.auto_approve` tier (see nodes/approval.py).
RISK_TIERS = ("read_only", "side_effecting", "destructive")


@dataclass(frozen=True)
class Toolkit:
    """A group of tools the user turns on or off as one (/tools). `core` cannot be turned off;
    `managed_by` names the command that owns a toolkit /tools only lists (an MCP server)."""

    label: str
    about: str
    core: bool = False
    managed_by: str = ""


# THE toolkits, in the order /tools lists them. The key is what the user types. A toolkit is an
# app or a capability, so it lines up with the macOS permission it needs. An MCP server's tools
# land under `mcp:<server>`, added by add_toolkit when the server connects.
TOOLKITS: "dict[str, Toolkit]" = {
    "core": Toolkit("Core", "checklist, questions, memory, calculator, clock", core=True),
    "files": Toolkit("Files", "read, write, edit, move, delete and search files"),
    "web": Toolkit("Web", "search the web, read a page"),
    "shell": Toolkit("Shell", "run a shell command"),
    "knowledge": Toolkit("Knowledge base", "search your documents (the RAG corpus), add and remove them"),
    "notes": Toolkit("Notes", "Apple Notes: search, read, create, append"),
    "calendar": Toolkit("Calendar", "Apple Calendar: list, create, update, delete events"),
    "mail": Toolkit("Mail", "Apple Mail: list, search, read, draft, reply, file"),
    "contacts": Toolkit("Contacts", "Apple Contacts: look a person up"),
    "reminders": Toolkit("Reminders", "Apple Reminders: list, create, complete"),
    "messages": Toolkit("Messages", "iMessage: find a group chat, read, send"),
    "shortcuts": Toolkit("Shortcuts", "list and run your Shortcuts"),
    "desktop": Toolkit("Desktop", "the front browser tab, the Finder selection"),
    "notifications": Toolkit("Notifications", "schedule a one-off alert"),
    "skills": Toolkit("Skills", "save a procedure as a skill"),
}

# Collected at import time as each tool module's @register_tool runs. registry.py re-exports these.
_TOOLS: list = []          # the BOUND tool objects, in registration order (registry filters it
                           # to the toolkits that are on — registry.apply_toolkits)
_ALL: list = []            # every registered tool object, on or off, in registration order
_TOOLKIT_OF: dict = {}     # tool name -> toolkit key
_RISK: dict = {}           # tool name -> risk tier
_RETRIEVAL: set = set()    # tool names whose results are recorded as retrieved documents
_UNTRUSTED: set = set()    # tool names whose OUTPUT crosses the trust boundary (quarantine scans)


class ToolError(Exception):
    """A tool call that did not do its job — raised, never returned, so the tools node stamps
    the round `error`: the adaptive think wakes on it and the answer's incidents note tells the
    user. The message is written for the model; the node hands it back as the observation."""


# Whether the call now executing was approved by a human at the gate — set by the tools node
# around each call from state["gate_events"], read by a tool whose record depends on it
# (`remember` stamps a fact by=user only for a call a person said yes to). False anywhere else:
# an auto-approved call, a direct invoke.
_HUMAN_APPROVED: contextvars.ContextVar = contextvars.ContextVar("human_approved", default=False)


def human_approved() -> bool:
    return bool(_HUMAN_APPROVED.get())


# Whether the call now executing skipped the gate because the user typed every word of it —
# set by the tools node from core/auto_memory.qualifies (the same check the approval node
# exempted the call on), read by `remember` to stamp the fact by=user src=said. False anywhere
# else.
_USER_STATED: contextvars.ContextVar = contextvars.ContextVar("user_stated", default=False)


def user_stated() -> bool:
    return bool(_USER_STATED.get())


def toolkit_of(name: str) -> "str | None":
    """The toolkit a registered tool belongs to; None for a name that is not registered."""
    return _TOOLKIT_OF.get(name)


def add_toolkit(key: str, toolkit: Toolkit) -> None:
    """Add a toolkit that is not known until runtime (an MCP server's, when it connects)."""
    TOOLKITS[key] = toolkit


def _record(t, risk: str, toolkit: str, retrieval: bool, untrusted: bool):
    """The one place a tool object enters the collections."""
    _TOOLS.append(t)
    _ALL.append(t)
    _TOOLKIT_OF[t.name] = toolkit
    _RISK[t.name] = risk
    if retrieval:
        _RETRIEVAL.add(t.name)
    if untrusted:
        _UNTRUSTED.add(t.name)
    return t


def register_tool(risk: str = "destructive", *, retrieval: bool = False, untrusted: bool = False,
                  toolkit: "str | None" = None):
    """Decorator: wrap a function as a LangChain tool AND register it (list + risk tier + retrieval
    flag + trust classification) in one place.

      @register_tool("read_only")                      # runs without prompting
      @register_tool("side_effecting")                 # hits the approval gate
      @register_tool("read_only", retrieval=True)      # output recorded as a retrieved document
      @register_tool("read_only", untrusted=True)      # output is EXTERNAL content — quarantine scans it

    `risk` must be one of RISK_TIERS; it defaults to the safe 'destructive' tier (always prompts)
    so a tool that forgets to declare one fails closed. `retrieval=True` marks tools whose output
    is a document worth recording for citations/trace (e.g. search_knowledge_base).
    `untrusted=True` marks tools whose output arrives from OUTSIDE the trust boundary (the web, a
    remote server, an ingested corpus) — the prompt-injection quarantine scans and fences those
    observations (registry.py pushes this set into trust/quarantine at startup). Declare it here,
    next to the risk tier, so a new external-fetch tool can't silently land inside the boundary.
    `toolkit` is the group the user turns on or off as one (TOOLKITS); left out, it is the
    tool module's own name (tools/mail.py -> "mail"), so only a tool that lives outside its
    toolkit's module says it."""
    if risk not in RISK_TIERS:
        raise ValueError(f"unknown risk tier {risk!r}; expected one of {RISK_TIERS}")

    def decorate(fn):
        kit = toolkit or fn.__module__.rsplit(".", 1)[-1]
        if kit not in TOOLKITS:
            raise ValueError(f"unknown toolkit {kit!r} for {fn.__name__}; expected one of "
                             f"{tuple(TOOLKITS)}")

        # functools.wraps keeps the name/docstring/signature, so the LangChain schema (and the
        # model-facing tool description) is exactly what the undecorated function would produce.
        @wraps(fn)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                diag.log(f"{fn.__name__} : {time.perf_counter() - start:.4f}s")

        return _record(_lc_tool(timed), risk, kit, retrieval, untrusted)

    return decorate


def register_tool_object(t, risk: str = "destructive", *, toolkit: str, retrieval: bool = False,
                         untrusted: bool = False):
    """Register an ALREADY-CONSTRUCTED LangChain tool object (list + risk tier + retrieval flag).

    The dynamic-source counterpart of @register_tool: a tool that can't be written as a decorated
    local function — e.g. a remote MCP tool built at runtime from a server's listing
    (mcp_client.py) — registers through here and flows into the exact same collections, so the
    approval gate, /tools, /policy risk, and the tool catalog treat it like any local tool.

    Unlike @register_tool (a developer-facing decorator, where an unknown tier is a programming
    error worth crashing on), `risk` here may originate from user config or a remote source, so an
    invalid value FAILS CLOSED to 'destructive' instead of raising — a dynamically-sourced tool
    must never end up ungated by a typo. Callers wanting to surface the downgrade should validate
    before calling. A tool must NEVER self-declare its tier (a remote server claiming read_only is
    exactly the attack the gate exists for); only the user's own config/overrides may relax it.
    `untrusted=True` marks the tool's output as external content for the quarantine scanner —
    mcp_client passes it for every remote tool. `toolkit` must already be in TOOLKITS
    (add_toolkit)."""
    if risk not in RISK_TIERS:
        risk = "destructive"
    if toolkit not in TOOLKITS:
        raise ValueError(f"unknown toolkit {toolkit!r} for {t.name}")
    return _record(t, risk, toolkit, retrieval, untrusted)

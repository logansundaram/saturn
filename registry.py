# import all tools here to register them in the tool registry

from tool_registry.calculator import calculate
from tool_registry.web_search import web_search
from tool_registry.web_extract import web_extract
from tool_registry.read_file import read_file
from tool_registry.write_file import write_file
from tool_registry.list_directory import list_directory
from tool_registry.deep_research import deep_research
from tool_registry.search_knowledge_base import search_knowledge_base
from tool_registry.remember import remember
from tool_registry.recall import recall
from tool_registry.run_python import run_python

tool = [
    calculate,
    web_search,
    web_extract,
    read_file,
    write_file,
    list_directory,
    deep_research,
    search_knowledge_base,
    remember,
    recall,
    run_python,
]

tools_by_name = {t.name: t for t in tool}


# Risk tiers drive the approval gate (see node_registry/approval.py):
#   read_only      — no side effects; runs freely, never prompts
#   side_effecting — writes/external actions; prompts for approval
#   destructive     — irreversible/dangerous; prompts for approval
# Unknown tools default to "destructive" (fail safe).
TOOL_RISK = {
    "calculate": "read_only",
    "web_search": "read_only",
    "web_extract": "read_only",  # fetches external pages but no side effects
    "read_file": "read_only",
    "list_directory": "read_only",
    "search_knowledge_base": "read_only",
    "recall": "read_only",
    "deep_research": "side_effecting",  # many external calls, slow/costly
    "write_file": "side_effecting",
    "remember": "side_effecting",  # persists across sessions; low risk but a real write
    "run_python": "destructive",  # arbitrary code execution — always gated (sandbox isn't a security boundary)
}


def risk_of(tool_name: str) -> str:
    return TOOL_RISK.get(tool_name, "destructive")

# import all tools here to register them in the tool registry. Tools are grouped by domain in
# tool_registry/: web.py (web_search/web_extract/deep_research), files.py (read/write/list),
# memory.py (remember/recall), knowledge.py (search_knowledge_base), calculator.py (calculate).

from tool_registry.calculator import calculate
from tool_registry.web import web_search, web_extract, deep_research
from tool_registry.files import read_file, write_file, list_directory
from tool_registry.knowledge import search_knowledge_base
from tool_registry.memory import remember, recall

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
    "web_extract": "read_only",
    "read_file": "read_only",
    "list_directory": "read_only",
    "search_knowledge_base": "read_only",
    "recall": "read_only",
    "deep_research": "side_effecting",  # many external calls, slow/costly
    "write_file": "side_effecting",
    "remember": "side_effecting",  # persists across sessions; low risk but a real write
}


def risk_of(tool_name: str) -> str:
    return TOOL_RISK.get(tool_name, "destructive")

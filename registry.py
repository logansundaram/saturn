# import all tools here to register them in the tool registry

from tool_registry.calculator import calculate
from tool_registry.web_search import web_search
from tool_registry.read_file import read_file
from tool_registry.write_file import write_file
from tool_registry.list_directory import list_directory
from tool_registry.deep_research import deep_research

tool = [calculate, web_search, read_file, write_file, list_directory, deep_research]

tools_by_name = {t.name: t for t in tool}

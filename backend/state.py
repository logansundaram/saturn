from typing import List, Any, Optional
from langchain.messages import ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_query: str
    current_response: str
    tools_called: List[str]
    tool_results: List[Any]
    context: List[str]
    tools_necessary: bool
    rag_necessary: bool

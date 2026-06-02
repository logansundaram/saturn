from typing import List, Any, Optional
from langchain.messages import ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_query: str  # may need to cahnge to human message
    current_response: str  # may need to cahnge to ai message
    tools_called: List[str]
    tool_results: List[Any]
    documents_retrieved: List[Any]
    context: str
    tools_necessary: bool
    rag_necessary: bool
    messages_relevant: bool

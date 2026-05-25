from typing import List, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    initial_query: List[str]
    route_decision: Optional[str]
    verification: Optional[Any]

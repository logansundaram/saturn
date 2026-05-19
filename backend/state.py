from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# define custom agent state, needs work and correct typings
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    # should contain the generated tool call adn the ouput of that tool call

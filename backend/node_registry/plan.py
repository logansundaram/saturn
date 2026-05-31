import time
from state import AgentState
from llms import llm
from messages import plan_freeform_system_msg
from pydantic import BaseModel, Field
from typing import List


class PlanStep(BaseModel):
    action: str = Field(description="retrieve | call_tool | reason | synthesize")
    description: str = Field(description="What to do and why")
    depends_on: str = Field(description="Which prior step this depends on, or 'none'")


# class AgentState(TypedDict):
#     messages: Annotated[List[Any], add_messages]
#     current_query: str
#     current_response: str
#     tools_called: List[str]
#     tool_results: List[Any]
#     context: List[str]
#     tools_necessary: bool
#     rag_necessary: bool


# might not be as relevant
class PlanOutput(BaseModel):
    tools_necessary: bool = Field(description="determine if the query needs tools")
    rag_necessary: bool = Field(
        description="determine if the query needs sepcific local docs"
    )


# need to alter this node to access to relevant context
def plan_node(state: AgentState):
    # plan node does not have access to tools avaible nor document metadata in the rag previous messages
    # should have context -> plan
    start = time.perf_counter()
    llm_with_structued_output = llm.with_structured_output(PlanOutput)
    llm_response = llm_with_structued_output.invoke(state["context"])
    print(f"plan_node : {time.perf_counter() - start:.4f}s")
    if llm_response.tools_necessary:
        print("tools_needed")
    return {
        "tools_necessary": llm_response.tools_necessary,
        "rag_necessary": llm_response.rag_necessary,
    }

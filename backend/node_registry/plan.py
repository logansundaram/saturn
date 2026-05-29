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


class PlanOutput(BaseModel):
    steps: List[PlanStep]


def plan_node(state: AgentState):
    start = time.perf_counter()
    result = llm.invoke(state["messages"] + [plan_freeform_system_msg])
    print(f"plan_node : {time.perf_counter() - start:.4f}s")
    return {"messages": result}

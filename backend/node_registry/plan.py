from state import AgentState
from llms import llm
from messages import plan_system_msg
from pydantic import BaseModel, Field
from typing import List


class PlanStep(BaseModel):
    action: str = Field(description="retrieve | call_tool | reason | synthesize")
    description: str = Field(description="What to do and why")
    depends_on: str = Field(description="Which prior step this depends on, or 'none'")


class PlanOutput(BaseModel):
    steps: List[PlanStep]


def plan_node(state: AgentState):
    planner = llm.with_structured_output(PlanOutput)
    result = planner.invoke(state["messages"] + [plan_system_msg])
    return {"messages": result}

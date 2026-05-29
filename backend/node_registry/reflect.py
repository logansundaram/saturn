import time
from state import AgentState
from llms import llm
from messages import reflect_system_msg
from pydantic import BaseModel, Field


class ReflectOutput(BaseModel):
    needs_revision: bool = Field(description="True if the output should be revised")
    critique: str = Field(description="Specific, actionable critique of the output")


MAX_REFLECTIONS = 2


def reflect_node(state: AgentState):
    start = time.perf_counter()
    reflector = llm.with_structured_output(ReflectOutput)
    result = reflector.invoke(state["messages"] + [reflect_system_msg])
    print(f"reflect_node : {time.perf_counter() - start:.4f}s")
    return {
        "reflection": result,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }


def should_revise(state: AgentState):
    if state["reflection_count"] >= MAX_REFLECTIONS:
        return False
    return state["reflection"].needs_revision

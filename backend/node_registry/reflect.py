    from state import AgentState
    from llms import llm
    from messages import reflect_system_msg
    from pydantic import BaseModel, Field

    class ReflectOutput(BaseModel):
        needs_revision: bool = Field(description="True if the output should be revised")
        critique: str = Field(description="Specific, actionable critique of the output")

    def reflect_node(state: AgentState):
        reflector = llm.with_structured_output(ReflectOutput)
        result = reflector.invoke(state["messages"] + [reflect_system_msg])
        return {
            "reflection": result,
            "reflection_count": state.get("reflection_count", 0) + 1
        }

    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
        initial_query: List[str]
        reflection: Optional[ReflectOutput]      # last critique
        reflection_count: int                     # loop guard

    MAX_REFLECTIONS = 2

    def should_revise(state: AgentState):
        if state["reflection_count"] >= MAX_REFLECTIONS:
            return False
        return state["reflection"].needs_revision
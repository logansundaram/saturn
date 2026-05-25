from state import AgentState
from llms import llm
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage
from typing import List, Tuple, Callable


class RouteDecision(BaseModel):
    choice: str = Field(description="The selected route name")


def build_router(
    routes: List[str], system_prompt: str
) -> Tuple[Callable, Callable]:
    """
    Factory that returns (router_node, routing_fn) for a given set of routes.

    router_node  — LangGraph node; calls the LLM and writes choice to state.
    routing_fn   — Pure conditional edge function; reads choice from state.

    Usage in a workflow:
        router_node, routing_fn = build_router(
            routes=["light", "moderate", "complex"],
            system_prompt=complexity_router_msg.content,
        )
        builder.add_node("router", router_node)
        builder.add_conditional_edges("router", routing_fn, {r: r for r in routes})
    """
    router_llm = llm.with_structured_output(RouteDecision)
    system_message = SystemMessage(content=system_prompt)

    def router_node(state: AgentState) -> dict:
        decision = router_llm.invoke(state["messages"] + [system_message])
        choice = decision.choice if decision.choice in routes else routes[0]
        return {"route_decision": choice}

    def routing_fn(state: AgentState) -> str:
        return state["route_decision"]

    return router_node, routing_fn
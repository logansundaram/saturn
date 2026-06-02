import time
from state import AgentState
from llms import llm
from messages import agent_verifier_msg
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage


class VerifierOutput(BaseModel):
    valid: bool = Field(
        description="True if the response fully and correctly answers the initial query"
    )
    feedback: str = Field(
        description="Specific actionable feedback on what is missing or wrong. Empty string if valid."
    )


verifier_llm = llm.with_structured_output(VerifierOutput)


def verifier_node(state: AgentState) -> bool:
    start = time.perf_counter()
    # implement later
    print(f"verifier_node : {time.perf_counter() - start:.4f}s")

    return False


def routing_fn(state: AgentState) -> str:
    return "pass" if state["verification"].valid else "fail"

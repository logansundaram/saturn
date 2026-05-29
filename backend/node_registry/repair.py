import time
from state import AgentState

from llms import llm

from langchain.messages import SystemMessage

from messages import generic_llm_call_msg


def repair_node(state: AgentState, system_prompt: SystemMessage = generic_llm_call_msg):
    start = time.perf_counter()
    llm_response = llm.invoke([system_prompt] + state["messages"])
    print(f"repair_node : {time.perf_counter() - start:.4f}s")
    return {"messages": [llm_response]}

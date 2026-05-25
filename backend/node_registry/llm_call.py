from state import AgentState

from llms import llm

from langchain.messages import SystemMessage

from messages import generic_llm_call_msg


# need a generic system prompt for llm call
def llm_call(state: AgentState, system_prompt: SystemMessage = generic_llm_call_msg):
    llm_response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [llm_response]}

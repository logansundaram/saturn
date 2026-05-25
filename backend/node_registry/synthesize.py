from state import AgentState
from llms import llm
from messages import synthesize_system_msg
from langchain.messages import SystemMessage

# def synthesize_node(state: AgentState):
#     extra = []

#     reflection = state.get("reflection")
#     if reflection and reflection.needs_revision:
#         extra.append(SystemMessage(
#             content=f"Your previous response was critiqued: {reflection.critique}\n\nAddress this directly in your revised
# response."
#         ))

#     response = llm.invoke(state["messages"] + extra + [synthesize_system_msg])
#     return {"messages": response}


def synthesize_node(
    state: AgentState, system_prompt: SystemMessage = synthesize_system_msg
):
    llm_response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [llm_response]}

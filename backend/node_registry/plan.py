import time
from langchain.messages import HumanMessage, SystemMessage
from state import AgentState
from llms import llm_with_structued_output

_plan_routing_msg = SystemMessage(content="""
You are a routing classifier. Given the session context and the user's current query, set each flag:

- tools_necessary: true if answering the query requires calling ANY tool — this includes math/calculation, current or real-time information, reading files, writing files, listing directories, or any external action. If the task cannot be completed from memory alone, set this to true.
- rag_necessary: true if the query asks about content that may exist in the local document knowledge base.
- messages_relevant: true ONLY if prior conversation turns (not the current query itself) contain context that is needed to answer the current query. If there are no prior turns, set this to false.

When in doubt about tools_necessary, prefer true over false.
""")


def plan_node(state: AgentState):
    start = time.perf_counter()
    llm_response = llm_with_structued_output.invoke(
        [_plan_routing_msg, HumanMessage(content="context: " + state["context"])]
    )
    print(f"plan_node : {time.perf_counter() - start:.4f}s")
    # need to add the plan to the state
    # plan = PlanStep.model_validate_json(llm_response.content)
    # plan = plan.model_dump()
    # print(plan)
    return {
        "tools_necessary": llm_response.tools_necessary,
        "rag_necessary": llm_response.rag_necessary,
        "messages_relevant": llm_response.messages_relevant,
    }

from langchain_ollama import ChatOllama
from registry import tool
from state import Plan


# Single shared local model for now. Phase 3 replaces this with a get_model(role)
# factory + hardware-tier config; until then these are the three handles the graph uses.
llm = ChatOllama(model="gemma4:e4b")

# Native tool-calling handle for the agent loop. Tools come from the registry.
llm_with_tools = llm.bind_tools(tool)

# Structured-output handle for the planner and the plan-updater. Emits a full Plan
# (list of PlanStep) — see state.py. method="json_schema" constrains generation to the
# schema at the Ollama level, which is far more reliable on small local models than
# parser-based modes (which let the model leak prose/markdown and fail to parse).
llm_with_plan = llm.with_structured_output(Plan, method="json_schema")

from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.messages import HumanMessage
from langchain_core.outputs import LLMResult

from agent import build_agent
from state import AgentState

_W = 64
_THICK = "━" * _W
_THIN = "─" * _W
_TRUNC_LEN = 500

# vibecoded need to review


def _trunc(s: str, n: int = _TRUNC_LEN) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n    ... [{len(s) - n} chars hidden]"


def _role(msg: Any) -> str:
    return type(msg).__name__.replace("Message", "")


def _fmt_msg(msg: Any) -> str:
    content = msg.content if hasattr(msg, "content") else str(msg)
    return f"  [{_role(msg)}] {_trunc(content)}"


class _DebugCallback(BaseCallbackHandler):
    """Intercepts LLM calls to print inputs and outputs."""

    def on_chat_model_start(
        self, serialized: dict, messages: list[list], **kwargs: Any
    ) -> None:
        name = (serialized.get("kwargs") or {}).get(
            "model", serialized.get("name", "llm")
        )
        print(f"\n{_THIN}")
        print(f"LLM INPUT  [{name}]")
        print(_THIN)
        for group in messages:
            for msg in group:
                print(_fmt_msg(msg))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"\n{_THIN}")
        print("LLM OUTPUT")
        print(_THIN)
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    print(_fmt_msg(msg))
                else:
                    print(f"  {_trunc(str(gen.text))}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"\n  [LLM ERROR] {error}")


def _print_node_updates(node_name: str, updates: dict) -> None:
    print(f"\n{_THICK}")
    print(f"NODE: {node_name.upper()}")
    print(_THICK)

    for key, value in updates.items():
        if key == "messages":
            msgs = value if isinstance(value, list) else [value]
            print(f"  messages  ({len(msgs)} appended)")
            for m in msgs:
                print(_fmt_msg(m))

        elif key == "context":
            print(f"  context:")
            print(f"  {_trunc(str(value))}")

        elif key in ("tools_necessary", "rag_necessary", "messages_relevant"):
            marker = "✓" if value else "✗"
            print(f"  {marker} {key}: {value}")

        elif key == "tool_results":
            print(f"  tool_results  ({len(value)} result(s))")
            for i, r in enumerate(value):
                print(f"  [{i}] {_trunc(str(r))}")

        elif key == "documents_retrieved":
            print(f"  documents_retrieved: {len(value)} doc(s)")
            for i, doc in enumerate(value):
                snippet = getattr(doc, "page_content", str(doc))
                print(f"  [{i}] {_trunc(snippet, 200)}")

        else:
            print(f"  {key}: {_trunc(str(value))}")

    # Call out routing decision after plan node
    if node_name == "plan":
        branches = []
        if updates.get("tools_necessary"):
            branches.append("tool")
        if updates.get("rag_necessary"):
            branches.append("rag")
        if not branches:
            branches.append("synthesize (direct)")
        print(f"\n  → routing to: {', '.join(branches)}")


def debug_run(
    query: str = "Explain the difference between a list and a tuple in Python.",
) -> None:
    graph = build_agent()

    state: AgentState = {
        "messages": [],
        "current_query": "",
        "current_response": "",
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
        "context": "",
        "tools_necessary": False,
        "rag_necessary": False,
        "messages_relevant": False,
    }

    state["messages"].append(HumanMessage(content=query))
    state["current_query"] = query

    print(f"\n{'═' * _W}")
    print(f"  DEBUG RUN")
    print(f"  Query: {query}")
    print(f"{'═' * _W}")

    config = {"callbacks": [_DebugCallback()]}

    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, updates in chunk.items():
            _print_node_updates(node_name, updates)

    print(f"\n{'═' * _W}\n")


def benchmark(query: str) -> None:
    graph = build_agent()

    state: AgentState = {
        "messages": [],
        "current_query": "",
        "current_response": "",
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
        "context": "",
        "tools_necessary": False,
        "rag_necessary": False,
        "messages_relevant": False,
    }

    state["messages"].append(HumanMessage(content=query))
    state["current_query"] = query
    state["context"] = ""
    state["tool_results"] = []

    state = graph.invoke(state)

    messages = state["messages"]
    last_msg = messages[-1]

    print(f"Assistant: {last_msg.content}")


if __name__ == "__main__":
    debug_run("Explain the difference between a list and a tuple in Python.")

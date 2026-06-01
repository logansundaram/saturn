import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from langchain.messages import HumanMessage

from agent import build_agent
from state import AgentState

SUITES: dict[str, list[str]] = {
    "llm": [
        "Explain the difference between a list and a tuple in Python.",
        "What are three pros and cons of microservices?",
    ],
    "calculator": [
        "What is 847 × 293 + 12,450?",
        "If I invest $5,000 at 7% annual interest for 10 years, what is the final value?",
    ],
    "web_search": [
        "What is the current price of Bitcoin?",
        "Who won the most recent Super Bowl?",
    ],
    "filesystem": [
        "List the files in the workspace.",
        "Read the file .manifest.md from the workspace and summarize it.",
        "Create a file called test_output.txt in the workspace with the text 'benchmark test'.",
    ],
    "rag": [
        "What documents are available in the knowledge base?",
    ],
    "multi_tool": [
        "Search the web for the latest news on LangGraph and save a summary to a file called langgraph_news.md in the workspace.",
        "Calculate 15% of 2,340 and then write the result to a file called calc_result.txt in the workspace.",
    ],
    "deep_research": [
        "Do a deep research report on the current state of local LLMs.",
    ],
}

SUITE_NAMES = list(SUITES.keys())


def _fresh_state() -> AgentState:
    return {
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


def run_query(graph, query: str) -> dict:
    state = _fresh_state()
    state["messages"].append(HumanMessage(content=query))
    state["current_query"] = query

    start = time.perf_counter()
    try:
        result = graph.invoke(state)
        elapsed = round(time.perf_counter() - start, 3)
        last_msg = result["messages"][-1]
        return {
            "status": "ok",
            "query": query,
            "response": last_msg.content,
            "latency_s": elapsed,
            "plan_flags": {
                "tools_necessary": result.get("tools_necessary"),
                "rag_necessary": result.get("rag_necessary"),
                "messages_relevant": result.get("messages_relevant"),
            },
            "tools_called": result.get("tools_called", []),
            "docs_retrieved": len(result.get("documents_retrieved", [])),
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 3)
        return {
            "status": "error",
            "query": query,
            "error": str(exc),
            "latency_s": elapsed,
        }


def run_suites(selected: list[str], output_path: Path | None = None) -> Path:
    print("Building agent...")
    graph = build_agent()

    print("Warming up model...")
    _warmup_state = _fresh_state()
    _warmup_state["messages"].append(HumanMessage(content="hi"))
    _warmup_state["current_query"] = "hi"
    graph.invoke(_warmup_state)

    total = sum(len(SUITES[s]) for s in selected)
    print(f"Running {total} queries across suite(s): {', '.join(selected)}\n")

    results: dict[str, list[dict]] = {}

    for suite_name in selected:
        queries = SUITES[suite_name]
        suite_results = []
        print(f"[{suite_name}] ({len(queries)} queries)")
        for query in queries:
            preview = query[:72] + "..." if len(query) > 72 else query
            print(f"  Q: {preview}")
            entry = run_query(graph, query)
            suite_results.append(entry)
            print(f"  → {entry['status']}  ({entry['latency_s']}s)")
        results[suite_name] = suite_results
        print()

    log_dir = Path(__file__).parent / "logging" / "benchmarks"
    log_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"benchmark_{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "suites_run": selected,
        "summary": {
            suite: {
                "total": len(r),
                "ok": sum(1 for e in r if e["status"] == "ok"),
                "error": sum(1 for e in r if e["status"] == "error"),
                "avg_latency_s": round(sum(e["latency_s"] for e in r) / len(r), 3)
                if r
                else 0,
            }
            for suite, r in results.items()
        },
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Results written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the AI agent across query suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available suites: {', '.join(SUITE_NAMES)}",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITE_NAMES + ["all"],
        default=["all"],
        metavar="SUITE",
        help=f"Suites to run: {{{', '.join(SUITE_NAMES + ['all'])}}} (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output JSON file path (default: logging/benchmarks/benchmark_<timestamp>.json)",
    )
    args = parser.parse_args()

    selected = SUITE_NAMES if "all" in args.suites else args.suites
    output_path = Path(args.output) if args.output else None

    run_suites(selected, output_path)


if __name__ == "__main__":
    main()

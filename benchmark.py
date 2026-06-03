import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain.messages import HumanMessage

from agent import build_agent, run_turn
from registry import risk_of
from state import AgentState

# Query suites for the living-plan ReAct loop (ground -> plan -> agent -> approval ->
# (tools -> update_plan -> agent)* -> synthesize). Each suite targets one capability of the
# loop so regressions are easy to localize. The harness auto-approves the approval gate so it
# measures capability, not human-in-the-loop latency.
SUITES: dict[str, list[str]] = {
    # Pure LLM reasoning — no tools, no retrieval. The agent should emit a final answer with
    # no tool calls, so the loop short-circuits agent -> synthesize.
    "llm": [
        "Explain the difference between a list and a tuple in Python.",
        "What are three pros and cons of microservices?",
        "What is the difference between concurrency and parallelism?",
        "Explain how garbage collection works in Python.",
        "What is the CAP theorem and why does it matter for distributed systems?",
        "Compare REST and GraphQL — when would you choose one over the other?",
    ],
    # Calculator tool — agent should call `calculate` (read-only, no approval) and report the
    # result. Tests numeric tool selection and faithful reporting of the returned value.
    "calculator": [
        "What is 847 × 293 + 12,450?",
        "If I invest $5,000 at 7% annual interest for 10 years, what is the final value?",
        "A rectangle is 47.3 meters long and 18.6 meters wide. What is its area and perimeter?",
        "What is 17.5% of 3,842?",
        "If a car travels 240 miles on 8 gallons of gas, what is its fuel efficiency in miles per gallon?",
        "Convert 98.6 degrees Fahrenheit to Celsius.",
    ],
    # Web search — agent should call `web_search` (read-only) and synthesize live results.
    "web_search": [
        "What is the current price of Bitcoin?",
        "Who won the most recent Super Bowl?",
        "What is the latest version of Python?",
        "What is the current weather in New York City?",
        "What are the top headlines in technology news today?",
        "What is the current USD to EUR exchange rate?",
    ],
    # Filesystem tools — `list_directory`/`read_file` are read-only; `write_file` is
    # side-effecting and trips the approval gate (auto-approved here). Tests tool selection,
    # workspace-sandboxed execution, and the gate routing.
    "filesystem": [
        "List the files in the workspace.",
        "Read the file .manifest.md from the workspace and summarize it.",
        "Create a file called test_output.txt in the workspace with the text 'benchmark test'.",
        "Create a file called notes.md in the workspace with a short note about Python lists.",
        "List the workspace files again to confirm test_output.txt was created.",
        "Read the file test_output.txt from the workspace and tell me what it says.",
    ],
    # RAG retrieval — agent should call `search_knowledge_base` (read-only) and ground its
    # answer in the retrieved chunks. The corpus is the synthetic RAG test pack under
    # database/documents/, so these target exact, gradeable facts (and the current-vs-deprecated
    # conflict between the handbook and router_config_v0_1_deprecated.txt). The expected answer
    # is noted after each query for manual grading.
    "rag": [
        "According to the Saturday.ai handbook, what is the default dashboard port?",  # 4173 (NOT the deprecated 5173)
        "What is the emergency rollback phrase defined in the handbook?",  # "blue lantern protocol"
        "Which embedding model does the handbook specify for RAG indexing?",  # nomic-embed-text:v1.5-lab
        "What target chunk size and overlap does the handbook mandate for chunking?",  # 420 tokens, 60 token overlap
        "What are the three agent workflow modes defined in the handbook?",  # FAST, STANDARD, DEEP
        "The deprecated router config and the current handbook disagree on the dashboard port. Which value is current and why?",  # 4173 wins via precedence; 5173 is deprecated
    ],
    # Multi-tool chaining — sequential use of two tools in one turn, exercising the
    # tools -> update_plan -> agent loop. Each ends in a `write_file` (gated, auto-approved).
    "multi_tool": [
        "Search the web for the latest news on LangGraph and save a summary to a file called langgraph_news.md in the workspace.",
        "Calculate 15% of 2,340 and then write the result to a file called calc_result.txt in the workspace.",
        "Search the web for the current price of Bitcoin and write it to a file called btc_price.txt in the workspace.",
        "Calculate the area of a circle with radius 12.5, then save the result to a file called circle_area.txt in the workspace.",
    ],
}

SUITE_NAMES = list(SUITES.keys())

# Deep research — exercises the `deep_research` tool (multi-source synthesis via Tavily's
# research API). DEFINED FOR REFERENCE BUT NOT RUN BY DEFAULT: each query is slow (minutes of
# polling) and costly (many external API calls). Opt in explicitly with --run-deep-research.
DEEP_RESEARCH_QUERIES: list[str] = [
    "Do a deep research report on the current state of local LLMs.",
    "Research the pros and cons of using LangGraph versus CrewAI for building AI agents.",
    "Do a deep research report on the best open-source embedding models available for local use.",
]
DEEP_RESEARCH_SKIP_REASON = (
    "deep_research is slow (minutes per query) and costly (many external API calls); "
    "queries are defined for reference and skipped unless --run-deep-research is passed."
)


def _fresh_state() -> AgentState:
    return {
        "messages": [],
        "current_query": "",
        "current_response": "",
        "context": "",
        "plan": [],
        "iteration": 0,
        "verified": False,
        "verifier_feedback": "",
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
    }


def run_query(graph, query: str) -> dict:
    state = _fresh_state()
    state["messages"].append(HumanMessage(content=query))
    state["current_query"] = query

    start = time.perf_counter()
    try:
        # Auto-approve gated tools so the benchmark measures capability without blocking on
        # the human approval gate. thread_id is required now that the graph is checkpointed.
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = run_turn(graph, state, config, approver=lambda _v: True)
        elapsed = round(time.perf_counter() - start, 3)
        last_msg = result["messages"][-1]
        tools_called = result.get("tools_called", [])
        return {
            "status": "ok",
            "query": query,
            "response": last_msg.content,
            "latency_s": elapsed,
            "plan": [
                {"label": s["label"], "status": s["status"]} for s in result.get("plan", [])
            ],
            "plan_steps": len(result.get("plan", [])),
            "iterations": result.get("iteration"),
            "tools_called": tools_called,
            # Tools that tripped the approval gate (anything not read-only) — surfaces how often
            # a suite exercises the safety gate.
            "gated_tools": [t for t in tools_called if risk_of(t) != "read_only"],
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


def run_suites(
    selected: list[str],
    output_path: Path | None = None,
    run_deep_research: bool = False,
) -> Path:
    print("Building agent...")
    graph = build_agent()

    print("Warming up model...")
    _warmup_state = _fresh_state()
    _warmup_state["messages"].append(HumanMessage(content="hi"))
    _warmup_state["current_query"] = "hi"
    # The graph is checkpointed, so even a bare invoke needs a thread_id in config.
    graph.invoke(_warmup_state, {"configurable": {"thread_id": str(uuid.uuid4())}})

    total = sum(len(SUITES[s]) for s in selected)
    if run_deep_research:
        total += len(DEEP_RESEARCH_QUERIES)
    print(f"Running {total} queries across suite(s): {', '.join(selected)}"
          + (", deep_research" if run_deep_research else "") + "\n")

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

    # Deep research: opt-in only. By default we record the queries as skipped rather than
    # spending the time/cost to run them.
    skipped: dict[str, dict] = {}
    if run_deep_research:
        suite_results = []
        print(f"[deep_research] ({len(DEEP_RESEARCH_QUERIES)} queries) — WARNING: slow + costly")
        for query in DEEP_RESEARCH_QUERIES:
            preview = query[:72] + "..." if len(query) > 72 else query
            print(f"  Q: {preview}")
            entry = run_query(graph, query)
            suite_results.append(entry)
            print(f"  → {entry['status']}  ({entry['latency_s']}s)")
        results["deep_research"] = suite_results
        print()
    else:
        skipped["deep_research"] = {
            "reason": DEEP_RESEARCH_SKIP_REASON,
            "queries": DEEP_RESEARCH_QUERIES,
        }
        print(f"[deep_research] SKIPPED ({len(DEEP_RESEARCH_QUERIES)} queries defined) — "
              "pass --run-deep-research to execute.\n")

    log_dir = Path(__file__).parent / "logging" / "benchmarks"
    log_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"benchmark_{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "suites_run": list(results.keys()),
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
        "skipped": skipped,
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Results written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the AI agent across query suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available suites: {', '.join(SUITE_NAMES)}\n"
        "deep_research is defined but skipped by default (slow + costly); "
        "use --run-deep-research to include it.",
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
        "--run-deep-research",
        action="store_true",
        help="Also run the deep_research queries (slow + costly; off by default).",
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

    run_suites(selected, output_path, run_deep_research=args.run_deep_research)


if __name__ == "__main__":
    main()

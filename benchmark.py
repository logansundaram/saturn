"""
Benchmark + eval harness (Phase 5).

Upgrades the old latency-only harness into a correctness/safety grader. One command runs the
suites, grades every case (routing, task success, LLM-as-judge, safety gate), measures the
pipeline overhead vs a direct model call, writes a scored JSON report, and diffs it against the
previous run so the output literally tells you whether the agent got better or worse
(SATURDAY_MVP_PLAN.md — Phase 5 exit criterion).

Usage:
    python benchmark.py                     # run all suites, grade, baseline, auto-diff vs last
    python benchmark.py --suites rag safety # a subset
    python benchmark.py --no-judge          # skip the LLM-as-judge (faster, deterministic only)
    python benchmark.py --no-baseline       # skip the overhead measurement
    python benchmark.py --diff A.json B.json # just diff two existing reports (no run)
    python benchmark.py --diff              # diff the two most recent reports
"""

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain.messages import HumanMessage

from agent import build_agent, run_turn
from config import get_config
from eval_cases import (
    SUITES,
    SUITE_NAMES,
    DEEP_RESEARCH_CASES,
    DEEP_RESEARCH_SKIP_REASON,
    EvalCase,
)
from registry import risk_of
import grading
from state import AgentState

LOG_DIR = Path(__file__).parent / "logging" / "benchmarks"


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
        "tok_per_sec": 0.0,
    }


def _requested_tools(messages: list) -> list[str]:
    """Every tool the model asked for across the turn (names from each AIMessage's tool_calls),
    whether or not it executed. Used for routing so a declined-at-the-gate call still scores."""
    names: list[str] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


def run_case(graph, case: EvalCase, workspace: Path) -> dict:
    """Run one case to completion and return the raw run record (pre-grading).

    The approver records whether the gate fired (the safety signal) and approves unless the
    case asks to decline (workflow #6). File-check / no-side-effect targets are deleted first so
    a stale file from a prior run can't produce a false pass."""
    for target in (
        case.file_check[0] if case.file_check else None,
        case.no_side_effect_file,
    ):
        if target:
            (workspace / target).unlink(missing_ok=True)

    state = _fresh_state()
    state["messages"].append(HumanMessage(content=case.query))
    state["current_query"] = case.query

    gate = {"fired": False}

    def approver(_interrupt_value) -> bool:
        gate["fired"] = True
        return not case.decline  # decline at the gate for safety cases, else approve

    start = time.perf_counter()
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = run_turn(graph, state, config, approver=approver)
        elapsed = round(time.perf_counter() - start, 3)
        last_msg = result["messages"][-1]
        tools_called = result.get("tools_called", [])
        return {
            "status": "ok",
            "query": case.query,
            "response": last_msg.content,
            "latency_s": elapsed,
            "iterations": result.get("iteration"),
            # Routing is scored on what the model *requested* (every tool_call it emitted), not
            # what executed — so a tool correctly chosen but declined at the gate (the safety
            # suite) still counts as correct routing.
            "tools_requested": _requested_tools(result["messages"]),
            "tools_called": tools_called,
            "gated_tools": [t for t in tools_called if risk_of(t) != "read_only"],
            "gate_fired": gate["fired"],
            "docs_retrieved": len(result.get("documents_retrieved", [])),
        }
    except Exception as exc:
        return {
            "status": "error",
            "query": case.query,
            "error": str(exc),
            "latency_s": round(time.perf_counter() - start, 3),
            "tools_called": [],
            "gate_fired": gate["fired"],
        }


def measure_baseline(queries: list[str]) -> dict:
    """Overhead baseline: time a direct single LLM call (no graph, no tools) per query. Compared
    against the full-pipeline latency, this isolates how much the agent loop costs over a raw
    model call (SATURDAY_MVP_PLAN.md — "Overhead baseline")."""
    from llms import get_model

    model = get_model("synthesizer")
    latencies = []
    for q in queries:
        start = time.perf_counter()
        try:
            model.invoke([HumanMessage(content=q)])
        except Exception:
            continue
        latencies.append(round(time.perf_counter() - start, 3))
    avg = round(sum(latencies) / len(latencies), 3) if latencies else None
    return {"queries": len(queries), "samples": len(latencies), "avg_direct_latency_s": avg}


def run_suites(
    selected: list[str],
    output_path: Path | None = None,
    run_deep_research: bool = False,
    use_judge: bool = True,
    use_baseline: bool = True,
) -> Path:
    print("Building agent...")
    graph = build_agent()
    workspace = get_config().path("workspace")

    judge_model = None
    if use_judge:
        from llms import get_model

        judge_model = get_model("judge")

    print("Warming up model...")
    warm = _fresh_state()
    warm["messages"].append(HumanMessage(content="hi"))
    warm["current_query"] = "hi"
    graph.invoke(warm, {"configurable": {"thread_id": str(uuid.uuid4())}})

    total = sum(len(SUITES[s]) for s in selected) + (len(DEEP_RESEARCH_CASES) if run_deep_research else 0)
    print(f"Running {total} queries across suite(s): {', '.join(selected)}"
          + (", deep_research" if run_deep_research else "")
          + (f"  [judge: {'on' if use_judge else 'off'}]\n"))

    results: dict[str, list[dict]] = {}

    def _run_one(case: EvalCase) -> dict:
        entry = run_case(graph, case, workspace)
        entry["grades"] = grading.grade_case(case, entry, workspace, judge_model)
        return entry

    for suite_name in selected:
        cases = SUITES[suite_name]
        print(f"[{suite_name}] ({len(cases)} queries)")
        suite_results = []
        for case in cases:
            preview = case.query[:72] + "..." if len(case.query) > 72 else case.query
            print(f"  Q: {preview}")
            entry = _run_one(case)
            suite_results.append(entry)
            print(f"  → {entry['status']}  ({entry['latency_s']}s)  {_case_grade_line(entry)}")
        results[suite_name] = suite_results
        print()

    skipped: dict[str, dict] = {}
    if run_deep_research:
        print(f"[deep_research] ({len(DEEP_RESEARCH_CASES)} queries) — WARNING: slow + costly")
        suite_results = []
        for case in DEEP_RESEARCH_CASES:
            preview = case.query[:72] + "..." if len(case.query) > 72 else case.query
            print(f"  Q: {preview}")
            entry = _run_one(case)
            suite_results.append(entry)
            print(f"  → {entry['status']}  ({entry['latency_s']}s)  {_case_grade_line(entry)}")
        results["deep_research"] = suite_results
        print()
    else:
        skipped["deep_research"] = {"reason": DEEP_RESEARCH_SKIP_REASON,
                                    "queries": [c.query for c in DEEP_RESEARCH_CASES]}
        print(f"[deep_research] SKIPPED ({len(DEEP_RESEARCH_CASES)} cases defined) — "
              "pass --run-deep-research to execute.\n")

    baseline = None
    if use_baseline and "llm" in results:
        print("Measuring overhead baseline (direct model calls)...")
        baseline = measure_baseline([e["query"] for e in results["llm"]])
        pipe = grading._mean([e["latency_s"] for e in results["llm"] if "latency_s" in e])
        if baseline["avg_direct_latency_s"] and pipe:
            baseline["pipeline_avg_latency_s"] = pipe
            baseline["overhead_factor"] = round(pipe / baseline["avg_direct_latency_s"], 2)
        print()

    scores = grading.score_results(results)
    if baseline:
        scores["overall"]["overhead_factor"] = baseline.get("overhead_factor")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = LOG_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "suites_run": list(results.keys()),
        "config": {
            "active_tier": get_config().active_tier,
            "judge": use_judge,
            "baseline": use_baseline,
        },
        "scores": scores,
        "baseline": baseline,
        "skipped": skipped,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print_report(scores, baseline)
    print(f"\nResults written to {output_path}")

    prev = _previous_report(output_path)
    if prev is not None:
        prev_scores = json.loads(prev.read_text(encoding="utf-8")).get("scores")
        if prev_scores:
            print(f"\nRegression diff vs previous run ({prev.name}):")
            print_diff(grading.diff_scores(prev_scores, scores))

    return output_path


def _case_grade_line(entry: dict) -> str:
    """One-line per-case grade summary for the live console."""
    g = entry.get("grades", {})
    bits = []
    if "routing" in g:
        bits.append("route " + ("ok" if g["routing"]["exact"] else f"f1={g['routing']['f1']}"))
    if "task" in g:
        bits.append("task " + ("PASS" if g["task"].get("passed") else "FAIL"))
    if "safety" in g:
        bits.append("gate " + ("PASS" if g["safety"].get("passed") else "FAIL"))
    if "judge" in g and g["judge"].get("score") is not None:
        bits.append(f"judge {g['judge']['score']}/5")
    return " | ".join(bits)


# --- reporting -------------------------------------------------------------------------
def _fmt(v) -> str:
    return "  -  " if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_report(scores: dict, baseline: dict | None = None) -> None:
    """Print the scored summary table (per suite + overall)."""
    cols = ["suite", "n", "route_f1", "exact", "task", "judge", "safety", "lat(s)"]
    print("=" * 78)
    print("SCORED SUMMARY")
    print("-" * 78)
    print(f"{cols[0]:<12}{cols[1]:>4}{cols[2]:>10}{cols[3]:>8}{cols[4]:>8}"
          f"{cols[5]:>8}{cols[6]:>9}{cols[7]:>10}")
    for suite, s in scores["suites"].items():
        print(
            f"{suite:<12}{s['n']:>4}{_fmt(s['routing_f1']):>10}{_fmt(s['routing_exact_rate']):>8}"
            f"{_fmt(s['task_success_rate']):>8}{_fmt(s['judge_avg']):>8}"
            f"{_fmt(s['safety_gate_rate']):>9}{_fmt(s['avg_latency_s']):>10}"
        )
    o = scores["overall"]
    print("-" * 78)
    print(
        f"{'OVERALL':<12}{'':>4}{_fmt(o['routing_f1']):>10}{_fmt(o['routing_exact_rate']):>8}"
        f"{_fmt(o['task_success_rate']):>8}{_fmt(o['judge_avg']):>8}"
        f"{_fmt(o['safety_gate_rate']):>9}{_fmt(o['avg_latency_s']):>10}"
    )
    print("=" * 78)
    print(f"task cases checked: {o['task_checked']}   judged: {o['judge_n']}   "
          f"gate cases: {o['safety_n']}   errors: {o['errors']}")
    if baseline and baseline.get("overhead_factor"):
        print(f"overhead: pipeline {baseline['pipeline_avg_latency_s']}s vs direct "
              f"{baseline['avg_direct_latency_s']}s = {baseline['overhead_factor']}x")


_ARROW = {"improved": "↑", "regressed": "↓", "unchanged": "=", "n/a": "·"}


def print_diff(diff: dict) -> None:
    print("-" * 60)
    print(f"{'metric':<20}{'old':>10}{'new':>10}{'delta':>10}  ")
    for r in diff["rows"]:
        arrow = _ARROW.get(r["direction"], " ")
        print(f"{r['metric']:<20}{_fmt(r['old']):>10}{_fmt(r['new']):>10}"
              f"{_fmt(r['delta']):>10}  {arrow} {r['direction']}")
    print("-" * 60)
    print(f"VERDICT: {diff['verdict']}  "
          f"(improved {diff['improved']}, regressed {diff['regressed']})")


def _previous_report(current: Path) -> Path | None:
    """The most recent benchmark report other than `current` (for the auto-diff)."""
    reports = sorted(LOG_DIR.glob("benchmark_*.json"))
    others = [p for p in reports if p.resolve() != current.resolve()]
    return others[-1] if others else None


def diff_command(paths: list[str]) -> None:
    """`--diff` entry: diff two reports (explicit paths, or the two most recent)."""
    if len(paths) >= 2:
        old, new = Path(paths[0]), Path(paths[1])
    elif len(paths) == 1:
        new = Path(paths[0])
        prev = _previous_report(new)
        if prev is None:
            print("No earlier report to diff against.")
            return
        old = prev
    else:
        reports = sorted(LOG_DIR.glob("benchmark_*.json"))
        if len(reports) < 2:
            print("Need at least two reports in logging/benchmarks/ to diff.")
            return
        old, new = reports[-2], reports[-1]

    old_scores = json.loads(old.read_text(encoding="utf-8")).get("scores")
    new_scores = json.loads(new.read_text(encoding="utf-8")).get("scores")
    if not old_scores or not new_scores:
        print("One of the reports has no `scores` block (pre-Phase-5 format); cannot diff.")
        return
    print(f"Diff: {old.name}  ->  {new.name}")
    print_diff(grading.diff_scores(old_scores, new_scores))


def main():
    parser = argparse.ArgumentParser(
        description="Run and grade the agent eval suites; diff against the previous run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available suites: {', '.join(SUITE_NAMES)}\n"
        "deep_research is defined but skipped by default (slow + costly).",
    )
    parser.add_argument("--suites", nargs="+", choices=SUITE_NAMES + ["all"], default=["all"],
                        metavar="SUITE", help=f"Suites to run: {{{', '.join(SUITE_NAMES + ['all'])}}} (default: all)")
    parser.add_argument("--run-deep-research", action="store_true",
                        help="Also run the deep_research cases (slow + costly; off by default).")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge grader (deterministic graders only).")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the direct-call overhead baseline.")
    parser.add_argument("--diff", nargs="*", default=None, metavar="REPORT",
                        help="Diff two reports (paths, or the two most recent) and exit; no run.")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output JSON path (default: logging/benchmarks/benchmark_<ts>.json)")
    args = parser.parse_args()

    if args.diff is not None:
        diff_command(args.diff)
        return

    selected = SUITE_NAMES if "all" in args.suites else args.suites
    run_suites(
        selected,
        Path(args.output) if args.output else None,
        run_deep_research=args.run_deep_research,
        use_judge=not args.no_judge,
        use_baseline=not args.no_baseline,
    )


if __name__ == "__main__":
    main()

"""The `plan` tool — the model's own checklist.

On a task that needs several tool calls, the model records what it intends to do here so the
user can follow along in the rail. It is intent, not record: the
tools node maps a successful call onto state["plan"] in the same step-dict shape every reader
already renders (the rail, the gate's step context, /trace why, replay, the headless --json
plan field), and nothing else keys on it — the answer trailers read the tool rounds that
actually ran. read_only: it changes nothing outside the turn's state, so it never faces the gate."""

from tools.toolspec import register_tool

PLAN_TOOL = "plan"
_STATUSES = ("pending", "done")


def to_plan(steps) -> list:
    """The tool's argument as state step dicts: blank labels dropped, unknown statuses read as
    pending, ids renumbered 1..N. A done item carries a result so core.state.current_step (the
    first item with `result is None`) points at the next pending one."""
    out: list = []
    for s in steps or []:
        if not isinstance(s, dict):
            s = {"label": str(s)}
        label = " ".join(str(s.get("label") or "").split())
        if not label:
            continue
        status = str(s.get("status") or "pending").strip().lower()
        if status not in _STATUSES:
            status = "pending"
        out.append({
            "step_id": len(out) + 1,
            "label": label,
            "status": status,
            "result": "done" if status == "done" else None,
        })
    return out


@register_tool("read_only", toolkit="core")
def plan(steps: list[dict]):
    """Record or update your checklist for a multi-step task so the user can follow along.
    `steps` is the FULL list in order, each {"label": "<what this step does>", "status":
    "pending" | "done"}. Call it again with updated statuses as steps complete. Use it only when
    the task needs several tool calls; never for a single lookup or a direct answer."""
    items = to_plan(steps)
    done = sum(1 for s in items if s["status"] == "done")
    return f"plan recorded: {len(items)} step(s), {done} done"

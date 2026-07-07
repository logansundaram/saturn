"""
User-interaction tool — ask_user.

The one tool whose "backend" is the human at the terminal. It pauses the RUNNING graph via
LangGraph's `interrupt()` — the same checkpoint machinery the approval gate, plan review, and the
freeze editor already ride — the loop's interrupt dispatcher renders the question at the prompt
(`ui.answer_question`), and the typed answer resumes the turn as this tool's observation. So the
answer lands on the plan step's `result` like any other tool output and flows to rectify/replan/
synthesize through the ordinary data bus.

Registered `read_only`: asking mutates nothing, so it never faces the approval gate — the
question IS the interaction. The answer is the user's own words: trusted input (never quarantined),
exactly like the request itself.

Headless (`-p`) there is no human to ask: the headless approver resolves the interrupt with a
bare True (noting the unanswered question on stderr), and the tool reports the absence honestly —
the model must proceed without the answer and disclose the gap, never fabricate one. The same
honest degradation covers an empty reply (the user pressing Enter / Ctrl-C at the prompt).

Determinism contract (same as plan_gate): a resumed `interrupt()` re-executes its node from the
top, so nothing here may mutate state before the interrupt — and tool_node's batches are
singletons (execute emits one call per step), so the re-run re-invokes only this tool.
"""

from langgraph.types import interrupt

from tools.toolspec import register_tool


@register_tool("read_only")
def ask_user(question: str):
    """Ask the human user ONE question and pause until they type an answer. Use when a needed
    value, choice, or confirmation is missing from the request and no file, note, or search can
    supply it — the alternative to asking is guessing. Not for presenting results or progress."""
    answer = interrupt({"type": "ask_user", "question": str(question or "").strip()})
    if isinstance(answer, str) and answer.strip():
        return f"The user answered: {answer.strip()}"
    # A non-string resume (the headless approver's bare True) or an empty reply: no answer
    # exists. Say so in the observation — downstream steps and the final answer must treat the
    # value as unknown, never invent one.
    return (
        "[no answer: the user did not provide one (running headless, or they submitted an "
        "empty reply). Proceed without it and state plainly what remains unknown or undone.]"
    )

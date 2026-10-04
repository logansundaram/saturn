"""
create_skill — the agent saves a procedure as one of the user's skills (core/skills.py; spec
docs/superpowers/specs/2026-10-03-user-skills-design.md).

A skill is text Saturn later follows as the user's own instructions, so the skills folder is a
persistence channel like memory, and the rule is memory's: nothing lands there without a
person's yes to that exact text.
  - `create_skill` is in `trust/policy.ALWAYS_ASKS`: no tier, open gate or always-allow lets it
    through, and headless refuses it even with --yolo. The tool checks `human_approved()` as
    well, so a save can never ride a policy mistake.
  - `draft()` is the ONE builder of (path, file text): the approval gate shows it
    (tui/ui/approval._render_skill_draft) and the tool writes it.
  - It writes the GLOBAL folder only. A workspace skill arrives with a project; the user edits
    those by hand. The file tools refuse both skills folders (tools/files._control_dirs).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from core import hooks, skills
from stores.snapshots import snapshot_file
from tools.toolspec import ToolError, human_approved, register_tool


def _builtin(key: str) -> bool:
    from commands._framework import resolves  # lazy: the framework imports nothing from tools/

    return resolves(key)


def _origin() -> str:
    """The provenance pointer written into the file: the run that drafted it, and the day."""
    from stores.trace import current_run_id

    run = current_run_id()
    today = date.today().isoformat()
    return f"saturn run={run} {today}" if run else f"saturn {today}"


def draft(args: dict) -> "tuple[Path, str]":
    """(the file a create_skill call writes, its exact text). The approval gate renders this
    and the tool writes this, so what the user approves is what lands on disk."""
    name = skills.draft_name(args.get("name"))
    return skills.target_path(name), skills.render(name, args.get("description") or "",
                                                   args.get("steps"), origin=_origin())


@register_tool("side_effecting")
def create_skill(name: str, description: str, steps: str, replace: bool = False):
    """Creates a skill: a reusable procedure the user runs later by typing /name. Use ONLY when the user asks to create, save or change a skill ("save that as a skill called weekly-review"). name: lowercase letters, digits and hyphens. description: one line saying what it does and when to use it. steps: the procedure as a numbered markdown list, written as instructions to yourself. To change a skill that exists, call with replace=true and the COMPLETE new steps. The user reads the whole skill before it is saved."""
    if not human_approved():
        raise ToolError("A skill is saved only after the user reads it and approves this exact "
                        "call at the prompt; nothing was saved.")
    name = skills.draft_name(name)
    steps = skills.steps_text(steps)
    problem = skills.draft_problem(name, description, steps, bool(replace), builtin=_builtin)
    if problem:
        raise ToolError(problem)
    target, text = draft({"name": name, "description": description, "steps": steps})
    refusal = hooks.before_write(target, "create_skill")
    if refusal:
        raise ToolError(refusal)
    existed = target.exists()
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(target)      # /undo takes a save back: a new skill removed, an old one restored
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)    # atomic: a reader never sees half a skill
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ToolError(f"could not save /{name}: {exc}; nothing was changed.") from exc
    hooks.run("after-write", file=str(target), tool="create_skill")
    return (f"{'Replaced' if existed else 'Saved'} the skill /{name} ({target}). "
            f"The user runs it by typing /{name}.")

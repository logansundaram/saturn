"""
/skills — the user's own procedures (core/skills.py, pivot #8): list them, read one, create
one, delete one. Running one is typing its name — `/weekly-review` in the REPL (app/repl.py) or
`saturn -p "/weekly-review"` headless.
"""

from __future__ import annotations

from commands._framework import command, resolves, _print

# `create` is the spelling in --help; `new` and `add` are accepted so neither habit errors.
_CREATE_VERBS = ("create", "new", "add")

TEMPLATE = """\
---
name: {name}
description: One line saying what this does and when to use it
# disable-model-invocation: true   # uncomment so only typing /{name} runs it
---

Steps Saturn follows when you type /{name}. Anything you type after the name is the request
these steps are applied to.

1. Say what to look at or gather first.
2. Say what to do with it.
3. Finish with a short summary: what was done and what is left.
"""


@command(
    "skills",
    "Your own procedures: list them, read one, create or delete one.",
    usage="/skills [show <name> | create <name> | delete <name>]",
    details="""
A skill is a procedure you write once in markdown and run by typing its name. Saturn follows
its steps for that request; every action it leads to still asks for approval as usual.

  /skills                 list your skills and where they live
  /skills show <name>     print one skill
  /skills create <name>   write a template to ~/.saturn/skills/<name>/SKILL.md
  /skills delete <name>   move a skill to the Trash (asks first)
  /<name> [request]       run a skill ("/weekly-review focus on work")
  /<name> --help          show it instead of running it

Where they live (a folder's own skill wins over a global one of the same name):
  ~/.saturn/skills/<name>/SKILL.md  or  ~/.saturn/skills/<name>.md
  <this folder>/.saturn/skills/<name>/SKILL.md

Ask Saturn to save one ("save that as a skill called weekly-review"): it shows you the whole
skill and asks before it writes, every time, whatever /policy says. Otherwise edit the files in
your editor; Saturn's file tools never touch these folders. A skill named like a built-in
command (/help, /memory, …) never runs — rename its file. Keys other than name, description,
disable-model-invocation and origin (allowed-tools, …) are ignored: a skill never changes what
asks first.

Example — ~/.saturn/skills/weekly-review/SKILL.md:

  ---
  name: weekly-review
  description: Friday review — what got done, what slipped, what is next
  ---
  1. List this week's calendar events and the reminders completed or overdue.
  2. Read my note titled "This week" if there is one.
  3. Answer in three short lists: done, slipped, next week.
""",
)
def _skills(ctx, args):
    from commands._utils import is_list_verb, is_remove_verb
    from core import skills, workspace
    from tui import ui

    sub = args[0].lower() if args else "list"
    rest = args[1:]
    if is_list_verb(sub) and not rest:
        _list(skills, workspace, ui)
    elif sub == "show" and len(rest) == 1:
        _show(skills, workspace, rest[0])
    elif sub in _CREATE_VERBS and len(rest) == 1:
        _create(skills, workspace, rest[0])
    elif is_remove_verb(sub) and len(rest) == 1:
        _delete(skills, workspace, ui, rest[0])
    else:
        _print("  usage: /skills [show <name> | create <name> | delete <name>] — "
               "/skills --help explains")


def _list(skills, workspace, ui) -> None:
    found = skills.discover()
    ui.section("skills", "run one by typing /<name> · /skills show <name> · /skills create <name>")
    rows = []
    for name in sorted(found):
        skill = found[name]
        notes = []
        if resolves(name):
            notes.append(f"the built-in /{name} wins — rename the file")
        if skill.manual_only:
            notes.append("only when you type it")
        text = skill.description + (f"  ({'; '.join(notes)})" if notes else "")
        rows.append(("/" + name, (text, "dim"),
                     (f"{skill.scope} · {skills.written_by(skill)}", "dim")))
    if rows:
        ui.table(rows)
    else:
        ui.note("no skills yet — /skills create <name> writes one in "
                + workspace.display(skills.global_dir()) + ", or ask Saturn to save one")
    _print(f"  folders: {workspace.display(skills.global_dir())} · "
           f"{workspace.display(skills.workspace_dir())} (this folder's; wins on a shared name)")
    for problem in skills.problems():
        ui.warn(problem)


def _show(skills, workspace, name: str) -> None:
    skill = skills.get(name)
    if skill is None:
        _print(f"  no skill named {name} — /skills lists them")
        return
    _print(f"  /{skill.name} · {workspace.display(skill.path)} ({skill.scope})")
    _print(f"  {skill.description}")
    if skill.origin:
        run = skills.written_by(skill).partition("#")[2]
        _print(f"  drafted by Saturn ({skill.origin})"
               + (f" — /trace why #{run} shows the turn" if run else ""))
    if resolves(skill.name):
        _print(f"  typing /{skill.name} runs the built-in command, not this skill — rename the file")
    if skill.manual_only:
        _print("  runs only when you type its name")
    if skill.extra_keys:
        _print(f"  ignored: {', '.join(skill.extra_keys)} — a skill never changes what asks first")
    if skill.path.name == skills.SKILL_FILE:
        others = sorted(p.name for p in skill.path.parent.iterdir() if p.name != skills.SKILL_FILE)
        if others:
            _print(f"  not used: {', '.join(others)} — Saturn reads only SKILL.md and runs "
                   "nothing from this folder")
    _print("")
    for line in skill.body.splitlines():
        _print(f"    {line}" if line.strip() else "")
    _print("")


def _create(skills, workspace, name: str) -> None:
    key = name.lstrip("/").lower()
    if not skills.valid_name(name.lstrip("/")):
        _print(f"  {name}: a skill name is lowercase letters, digits and hyphens (up to 64), "
               "e.g. weekly-review")
        return
    if resolves(key):
        _print(f"  /{key} is a built-in command — pick another name")
        return
    existing = skills.get(key)
    if existing is not None:
        _print(f"  /{key} already exists: {workspace.display(existing.path)} — edit it there")
        return
    path = skills.global_dir() / key / skills.SKILL_FILE
    # A file the loader skipped (unclosed frontmatter, a folder spelled Weekly-Review on a disk
    # that ignores case) is not in skills.get(), but it is still the user's text — and the
    # startup warning sends them straight here. Never write over it.
    taken = skills.unloaded_file(key)
    if taken is not None:
        _print(f"  {workspace.display(taken)} is already there but did not load as a skill "
               "(/skills says why) — fix or remove that file first; nothing was written")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as f:    # "x": create, never replace
            f.write(TEMPLATE.format(name=key))
    except OSError as exc:
        _print(f"  could not write {workspace.display(path)}: {exc} — nothing was written")
        return
    _print(f"  wrote {workspace.display(path)} — open it in your editor and replace the steps; "
           f"/{key} runs it")


def _delete(skills, workspace, ui, name: str) -> None:
    """Move the skill `/name` runs to the Trash, after a y/N. The user's own action, so it is
    not an /undo entry (/undo is the record of what the agent changed): the Trash is the way
    back. The agent has no way to do this — the file tools refuse the skills folders."""
    import shutil

    from commands import _utils
    from tools import files

    skill = skills.get(name)
    if skill is None:
        _print(f"  no skill named {name.lstrip('/')} — /skills lists them")
        return
    # A folder skill goes as its folder (SKILL.md and whatever sits beside it), a flat one as
    # its file — never the skills folder itself.
    target = skill.path.parent if skill.path.name == skills.SKILL_FILE else skill.path
    _print(f"  /{skill.name} · {workspace.display(target)} ({skill.scope})")
    if target.is_dir():
        others = sorted(p.name for p in target.iterdir() if p.name != skills.SKILL_FILE)
        if others:
            _print(f"  the folder also holds: {', '.join(others)} — it goes too")
    if not _utils._stdin_is_tty():
        _print("  deleting a skill needs a person at the keyboard to say yes — nothing was deleted")
        return
    if ui.ask("  move it to the Trash? [y/N] ").lower() not in ("y", "yes"):
        _print(f"  kept /{skill.name}")
        return
    try:
        trash = files._trash_dir()
        trash.mkdir(parents=True, exist_ok=True)
        slot = files._trash_slot(trash, target.name)
        shutil.move(str(target), str(slot))
        files._trash_info(slot, target)
    except OSError as exc:
        _print(f"  could not delete /{skill.name}: {exc} — nothing was changed")
        return
    _print(f"  moved /{skill.name} to the Trash ({workspace.display(slot)}) — restore it from there")
    left = skills.get(skill.name)   # a global skill the deleted workspace one was shadowing
    if left is not None:
        _print(f"  the {left.scope} /{left.name} now runs: {workspace.display(left.path)}")

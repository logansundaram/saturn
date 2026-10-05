"""Skills — the user's own procedures, written as markdown (pivot #8).

A skill is a file the user writes, in the shape Claude Code and other agents share: optional
YAML frontmatter (`name`, a one-line `description`, `disable-model-invocation`) and a markdown
body of steps. Typing `/<name> [request]` runs it: the body rides that turn's DYNAMIC grounding
(nodes/ground.py), so a skill costs no extra model call and never touches the cached prefix.

Where they live — a workspace skill wins over a global one of the same name:
  $SATURN_HOME/skills/<name>/SKILL.md   or   $SATURN_HOME/skills/<name>.md   (~/.saturn/skills)
  <workspace>/.saturn/skills/<name>/SKILL.md   or   …/<name>.md

The name is the file name (the folder's, for SKILL.md): what the user types. Broken files are
named by `problems()` (the startup warning, /skills), never fatal. A skill is the user's own
instruction text, so the file tools refuse to write into these folders (tools/files
`_control_dirs`), and a skill never changes what the approval gate asks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from textutil import clip, unseen_chars

SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
BODY_CAP = 6000          # the same budget as a SATURN.md (nodes/ground._INSTRUCTIONS_CAP)
DESCRIPTION_CAP = 200
MANIFEST_CAP = 1500      # Phase 2: the stable-grounding list of skills the model may load
DRAFT_CAP = 3000         # steps Saturn itself saves (tools/skills.create_skill): what a person
                         # will read, whole, at the approval prompt — refused beyond, never cut

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_KNOWN_KEYS = {"name", "description", "disable-model-invocation", "origin"}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    scope: str               # "global" ($SATURN_HOME/skills) | "workspace" (<root>/.saturn/skills)
    manual_only: bool        # disable-model-invocation: true
    extra_keys: tuple        # frontmatter keys Saturn ignores (allowed-tools, model, …)
    origin: str = ""         # "saturn run=<id> <date>" when create_skill wrote it; "" = the user


def global_dir() -> Path:
    from config import saturn_home

    return saturn_home() / SKILLS_DIR


def workspace_dir() -> Path:
    from core import workspace

    return workspace.root() / ".saturn" / SKILLS_DIR


def _folders() -> "list[tuple[str, Path]]":
    """(scope, folder), global first. Launched from ~ the two are one folder: listed once."""
    out: list = []
    seen: set = set()
    for scope, folder in (("global", global_dir()), ("workspace", workspace_dir())):
        try:
            key = folder.resolve()
        except OSError:
            key = folder
        if key in seen:
            continue
        seen.add(key)
        out.append((scope, folder))
    return out


def control_dirs() -> "list[Path]":
    """The folders the file tools must never write into (tools/files): a skill is an instruction
    the model follows as the user's own, so a file planted there is one the user never wrote."""
    return [folder for _scope, folder in _folders()]


def linked_targets() -> "list[Path]":
    """Where skills that are symlinked INTO a skills folder really live (a dotfiles folder, a
    ~/.claude/skills entry): the real folder of a linked `<name>/`, the real file of a linked
    `SKILL.md` or `<name>.md`. The loader follows those links, so the file tools must refuse
    the real paths too (tools/files) — else a write to `dotfiles/standup/SKILL.md` rewrites a
    skill without ever naming the skills folder."""
    out: list = []
    for _scope, folder in _folders():
        try:
            base = folder.resolve()
        except OSError:
            continue
        for path in _candidates(folder):
            for entry in ((path.parent, path) if path.name == SKILL_FILE else (path,)):
                try:
                    real = entry.resolve()
                except OSError:
                    continue
                if not real.is_relative_to(base) and real not in out:
                    out.append(real)
    return out


def valid_name(name: str) -> bool:
    return bool(_NAME.match(str(name or "")))


def _first_line(body: str) -> str:
    for line in body.splitlines():
        text = line.strip().lstrip("#").strip().lstrip("-*").strip()
        if text:
            return text
    return ""


def _parse(path: Path, scope: str) -> "tuple[Skill | None, str | None]":
    """(skill, problem) for one file; a problem with no skill means the file was skipped."""
    name = path.parent.name if path.name == SKILL_FILE else path.stem
    if not valid_name(name):
        return None, (f"{path}: {name!r} is not a skill name — lowercase letters, digits and "
                      "hyphens, up to 64 — skipped")
    try:
        # errors="replace": a hand-edited file with a stray non-UTF-8 byte still loads.
        text = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
    except OSError as exc:
        return None, f"{path}: unreadable, skipped ({exc.strerror or exc})"
    meta: dict = {}
    body = text
    m = _FRONTMATTER.match(text)
    if m:
        try:
            import yaml

            meta = yaml.safe_load(m.group(1)) or {}
            if not isinstance(meta, dict):
                raise ValueError("expected key: value lines")
        except Exception as exc:
            return None, f"{path}: frontmatter unreadable, skipped ({' '.join(str(exc).split())})"
        # YAML aliases let a few hundred bytes stand for billions of items; a value is only
        # ever used as one line of text, so anything else is refused BEFORE it is turned into
        # a string (a workspace skill arrives with a folder, and this runs at every prompt).
        for key in ("name", "description", "origin"):
            if not isinstance(meta.get(key), (str, int, float, bool, type(None))):
                return None, f"{path}: frontmatter `{key}` must be one line of text, skipped"
        body = text[m.end():]
    elif text.startswith("---"):
        return None, f"{path}: frontmatter has no closing --- line, skipped"
    body = body.strip()
    if not body:
        return None, f"{path}: no steps after the frontmatter, skipped"
    if len(body) > BODY_CAP:
        body = body[:BODY_CAP] + f"\n… ({path.name} truncated at {BODY_CAP} characters — keep a skill short)"
    problem = None
    declared = str(meta.get("name") or "").strip()
    if declared and declared != name:
        problem = f"{path}: its name: line says {declared!r}; the file name wins — type /{name}"
    description = " ".join(str(meta.get("description") or "").split()) or _first_line(body)
    extra = tuple(sorted(k for k in map(str, meta) if k not in _KNOWN_KEYS))
    return Skill(name=name, description=clip(description, DESCRIPTION_CAP), body=body, path=path,
                 scope=scope, manual_only=meta.get("disable-model-invocation") is True,
                 extra_keys=extra, origin=" ".join(str(meta.get("origin") or "").split())), problem


def _candidates(folder: Path) -> "list[Path]":
    """One folder's skill files: every <name>/SKILL.md first, then every flat <name>.md."""
    try:
        nested = sorted(p for p in folder.glob(f"*/{SKILL_FILE}") if p.is_file())
        flat = sorted(p for p in folder.glob("*.md") if p.is_file() and p.name != SKILL_FILE)
    except OSError:
        return []
    return nested + flat


def _scan() -> "tuple[dict[str, Skill], list[str]]":
    found: dict = {}
    problems: list = []
    for scope, folder in _folders():
        here: dict = {}
        for path in _candidates(folder):
            skill, problem = _parse(path, scope)
            if problem:
                problems.append(problem)
            if skill is None:
                continue
            if skill.name in here:
                problems.append(f"{path}: /{skill.name} is also {here[skill.name].path} — that one runs")
                continue
            here[skill.name] = skill
        found.update(here)  # this folder's skills replace the global ones of the same name
    return found, problems


def discover() -> "dict[str, Skill]":
    """Every loadable skill by name. Read from disk on each call (a handful of small files):
    a skill written in the editor runs without a restart."""
    return _scan()[0]


def problems() -> "list[str]":
    """What the skill files got wrong — one sentence each, naming the file."""
    return _scan()[1]


def get(name: str) -> "Skill | None":
    return discover().get(str(name or "").strip().lstrip("/").lower())


def invocation(line: str, builtin: "Callable[[str], bool]" = lambda key: False
               ) -> "tuple[Skill, str] | None":
    """(skill, request text) for a `/name …` line, or None: not a slash line, a name
    `builtin(name)` claims (Saturn's own commands always win), or no skill by that name."""
    text = str(line or "").strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(None, 1)
    if not parts:
        return None
    key = parts[0].lower()
    if builtin(key):
        return None
    skill = discover().get(key)
    if skill is None:
        return None
    return skill, (parts[1].strip() if len(parts) > 1 else "")


_HOW = {
    "typed": ("The user ran it by typing /{name}. The text after the name in their message is "
              "what to apply it to; with nothing after it, run it as written."),
    "matched": "You loaded it because the request matches it. Apply it to the request.",
}


def block(skill: Skill, how: str = "typed") -> str:
    """The skill as a grounding block (a typed /name) or a use_skill observation (matched)."""
    return (f"### Skill /{skill.name} — the user's own procedure: {skill.description}\n"
            + _HOW[how].format(name=skill.name)
            + " Follow its steps in order; every action still asks for approval as usual.\n\n"
            + skill.body)


# ── a skill Saturn drafts (tools/skills.create_skill) ────────────────────────────────────────
# Pure: nothing here writes. The tool writes exactly what `render` returns, the approval gate
# shows exactly what `render` returns, and `draft_problem` is asked twice — by the agent's
# hygiene before the gate, and by the tool at the write.

_ORIGIN_RUN = re.compile(r"\brun=(\w+)")
_NUMBERED = re.compile(r"\d+[.)]\s")


def written_by(skill: Skill) -> str:
    """Who wrote a skill, for /skills: `you`, or `saturn · #<run>` (`/trace why #<run>` shows
    the turn). Read from the `origin` key — a pointer, not a security claim: a hand edit may
    keep or drop it."""
    if not skill.origin.startswith("saturn"):
        return "you"
    m = _ORIGIN_RUN.search(skill.origin)
    return f"saturn · #{m.group(1)}" if m else "saturn"


def draft_name(name) -> str:
    """A drafted name as the file name it becomes: `/Weekly Review` -> `weekly-review`. Applied
    before the gate, so the human reads the name that will be written."""
    return re.sub(r"[\s_]+", "-", str(name or "").strip().lstrip("/").strip().lower())


def steps_text(steps) -> str:
    """Drafted steps as the markdown the file holds. A small model sometimes sends a JSON list
    where the schema says a string: each item becomes a numbered line."""
    if isinstance(steps, (list, tuple)):
        items = [str(s).strip() for s in steps if str(s).strip()]
        return "\n".join(s if _NUMBERED.match(s) else f"{i}. {s}" for i, s in enumerate(items, 1))
    return str(steps or "").strip()


def render(name: str, description: str, steps, origin: str = "") -> str:
    """The exact text of a skill file. The frontmatter goes through yaml.safe_dump, so a colon,
    a quote or a leading dash in the description cannot break it; terminal controls are made
    visible, so a saved skill holds no live escape. `_parse(render(...))` gives the same name,
    description and body back (tests/test_skills.py pins it)."""
    import yaml

    from textutil import visible_controls

    meta = {"name": str(name), "description": " ".join(visible_controls(description).split())}
    if origin:
        meta["origin"] = str(origin)
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False,
                           width=10_000)
    return f"---\n{front}---\n\n{visible_controls(steps_text(steps))}\n"


def in_scope(scope: str) -> "dict[str, Skill]":
    """ONE folder's loadable skills by name, unmerged: "global" or "workspace". `discover()`
    merges the two (the workspace one wins); a draft needs them apart, because Saturn writes
    only the global folder. Launched from ~ the two are one folder, listed as "global"."""
    found: dict = {}
    for s, folder in _folders():
        if s != scope:
            continue
        for path in _candidates(folder):
            skill, _problem = _parse(path, s)
            if skill is not None:
                found.setdefault(skill.name, skill)   # SKILL.md before a flat file, as in _scan
    return found


def target_path(name: str) -> Path:
    """The file a draft named `name` is written to: the global skill of that name where it
    already is (folder or flat), else `<global>/<name>/SKILL.md`. A skill LINKED into the
    folder is written where it really lives — the gate names that file, and the link (a
    dotfiles checkout, say) keeps pointing at the text that runs."""
    current = in_scope("global").get(name)
    if current is None:
        return global_dir() / name / SKILL_FILE
    try:
        real = current.path.resolve()
        plain = global_dir().resolve() / current.path.relative_to(global_dir())
    except (OSError, ValueError):
        return current.path
    return current.path if real == plain else real


def unloaded_file(name: str) -> "Path | None":
    """A file sitting where the global skill `name` would be written that did NOT load as that
    skill (unclosed frontmatter, a folder spelled Weekly-Review on a disk that ignores case, a
    dangling link), or None. It is still the user's text: nothing Saturn does — /skills create
    or a create_skill draft, `replace` or not — writes over it."""
    if name in in_scope("global"):
        return None
    return next((p for p in (global_dir() / name / SKILL_FILE, global_dir() / f"{name}.md")
                 if os.path.lexists(p)), None)


def existing_text(skill: Skill) -> str:
    """What the model is told when it drafts over a skill without `replace`: the current text,
    which it has no other way to read, and how to change it."""
    return (f"/{skill.name} already exists ({skill.path}); nothing was changed. Its current "
            f"text:\n\ndescription: {skill.description}\n\n{skill.body}\n\n"
            "To change it, call create_skill again with replace=true and the COMPLETE new steps.")


def draft_problem(name, description, steps, replace: bool = False,
                  builtin: "Callable[[str], bool]" = lambda key: False) -> "str | None":
    """Why a drafted skill cannot be saved — one sentence for the model — or None. Checked in
    this order: the name, a built-in command's name, the description, the steps, a workspace
    skill of that name, a file of that name the loader skipped, a global skill of that name
    without `replace`. Nothing is truncated to fit: what the user is shown is what is saved."""
    name = str(name or "")
    if not valid_name(name):
        return (f"{name!r} is not a skill name: use lowercase letters, digits and hyphens, up "
                "to 64 characters, starting with a letter or digit (e.g. weekly-review).")
    if builtin(name):
        return (f"/{name} is a built-in command, so a skill by that name would never run. "
                "Pick another name.")
    description = " ".join(str(description or "").split())
    if not description:
        return "The skill needs a one-line description: what it does and when to use it."
    if len(description) > DESCRIPTION_CAP:
        return (f"The description is {len(description)} characters; keep it to one line of at "
                f"most {DESCRIPTION_CAP} characters.")
    steps = steps_text(steps)
    if not steps:
        return "The skill needs its steps: the procedure as a numbered markdown list."
    if len(steps) > DRAFT_CAP:
        return (f"The steps are {len(steps)} characters; a skill Saturn saves is at most "
                f"{DRAFT_CAP} characters, so the user can read all of it before approving. "
                "Shorten the steps.")
    unseen = unseen_chars(f"{description}\n{steps}")
    if unseen:
        return ("The skill holds characters a person cannot see at the approval prompt "
                f"({', '.join(unseen[:5])}); remove them — a skill is saved only as text the "
                "user can read in full.")
    local = in_scope("workspace").get(name)
    if local is not None:
        return (f"/{name} is this folder's own skill ({local.path}); Saturn saves skills only in "
                f"{global_dir()}. Ask the user to edit that file by hand, or pick another name.")
    taken = unloaded_file(name)
    if taken is not None:
        return (f"{taken} is already there but did not load as a skill, so it cannot be "
                "replaced from here; nothing was changed. Tell the user: /skills says why it "
                "did not load, and they fix or remove that file by hand — or pick another name.")
    current = in_scope("global").get(name)
    if current is not None and not replace:
        return existing_text(current)
    return None

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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from textutil import clip

SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
BODY_CAP = 6000          # the same budget as a SATURN.md (nodes/ground._INSTRUCTIONS_CAP)
DESCRIPTION_CAP = 200
MANIFEST_CAP = 1500      # Phase 2: the stable-grounding list of skills the model may load

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_KNOWN_KEYS = {"name", "description", "disable-model-invocation"}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    scope: str               # "global" ($SATURN_HOME/skills) | "workspace" (<root>/.saturn/skills)
    manual_only: bool        # disable-model-invocation: true
    extra_keys: tuple        # frontmatter keys Saturn ignores (allowed-tools, model, …)


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
                 extra_keys=extra), problem


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

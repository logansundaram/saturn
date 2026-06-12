"""
User-defined slash commands — markdown templates that become /name (roadmap Tier-3 #9).

Drop `<name>.md` into `database/commands/` (paths.user_commands; user data, gitignored, survives
/update) and `/name` exists: invoking it expands the template into an agent turn. Optional YAML
frontmatter supplies the /help summary; `$ARGUMENTS` in the body is replaced with everything
typed after the command (no `$ARGUMENTS` → the arguments append to the end).

    ---
    summary: Summarize a file in three bullets
    ---
    Summarize @$ARGUMENTS in exactly three bullet points.

Then `/brief notes.md` runs the expanded text as a normal turn — same plan, same gates, same
trace. A template is a PROMPT, never code: it can't call tools or skip the gate; everything it
triggers goes through the loop's existing trust machinery. Names that collide with a built-in
command (or its aliases) are skipped and reported. Loaded templates are listed in /help (tagged
"(user)"); templates are picked up automatically — startup loads them, and an unknown /name at
dispatch triggers one rescan — no management command needed.
"""

from __future__ import annotations

from pathlib import Path

from commands._framework import (
    COMMANDS,
    _ALIASES,
    _RENAMED,
    SlashCommand,
    command_completions,
)

# Names this module registered (so a reload can drop them before re-scanning, and a user file
# can never shadow a built-in: built-ins are whatever is in COMMANDS that ISN'T ours).
_REGISTERED: set[str] = set()
_PROBLEMS: list[str] = []


def _commands_dir() -> Path:
    """The user-command template directory (`paths.user_commands`, default database/commands).
    Tolerates a config.yaml predating the key — installed-mode users upgrade in place."""
    from config import get_config

    cfg = get_config()
    rel = cfg.get("paths.user_commands", "database/commands")
    p = Path(rel)
    if not p.is_absolute():
        p = cfg.path("database").parent / p
    return p


def parse_template(text: str) -> tuple[dict, str]:
    """Split optional `---`-fenced YAML frontmatter from a template. Returns (meta, body);
    malformed frontmatter degrades to an empty meta + the full text as body."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text.strip()
    lines = stripped.splitlines()
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            front = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1:])
            try:
                import yaml

                meta = yaml.safe_load(front)
            except Exception:
                meta = None
            return (meta if isinstance(meta, dict) else {}), body.strip()
    return {}, text.strip()


def expand_arguments(body: str, args: list[str]) -> str:
    """Substitute the invocation's arguments into the template: every `$ARGUMENTS` is replaced;
    a template without the placeholder gets them appended (so `/name extra context` still
    carries the extra context)."""
    arg_str = " ".join(args).strip()
    if "$ARGUMENTS" in body:
        return body.replace("$ARGUMENTS", arg_str)
    return body + (f"\n\n{arg_str}" if arg_str else "")


def _make_handler(body: str):
    def handler(ctx, args):
        # The expansion runs as the next agent turn (the same requeue seam /retry full uses)
        # — through the normal plan/gate/trace machinery, never around it.
        ctx.requeue = expand_arguments(body, args)

    return handler


def load_user_commands() -> tuple[int, list[str]]:
    """(Re)scan the template directory and register each `*.md` as a slash command. Returns
    (count_registered, problems). Safe to call repeatedly — previously loaded user commands are
    dropped first, so deletes and renames take effect on reload."""
    global _PROBLEMS
    problems: list[str] = []

    for name in _REGISTERED:
        COMMANDS.pop(name, None)
    _REGISTERED.clear()

    d = _commands_dir()
    if d.exists():
        for f in sorted(d.glob("*.md")):
            name = f.stem.lower().replace(" ", "-")
            if not name:
                continue
            if name in COMMANDS or name in _ALIASES:
                problems.append(f"/{name} ({f.name}): collides with a built-in command — rename the file")
                continue
            # Legacy names too: dispatch resolves COMMANDS before consulting _RENAMED, so a
            # template named egress.md/why.md/save.md would otherwise hijack the documented
            # "/egress moved — use /privacy egress" redirect with an arbitrary agent turn.
            if name in _RENAMED:
                problems.append(
                    f"/{name} ({f.name}): shadows a renamed built-in "
                    f"(/{name} redirects to /{_RENAMED[name]}) — rename the file"
                )
                continue
            try:
                meta, body = parse_template(f.read_text(encoding="utf-8"))
            except OSError as exc:
                problems.append(f"{f.name}: {exc}")
                continue
            if not body:
                problems.append(f"{f.name}: empty template")
                continue
            summary = str(meta.get("summary") or "").strip() or f"user command ({f.name})"
            COMMANDS[name] = SlashCommand(
                name=name,
                summary=summary + "  (user)",
                handler=_make_handler(body),
                usage=f"/{name} [arguments]",
                details=(
                    f"User-defined command from {f.name} (under {d}).\n"
                    "Invoking it expands the template into a normal agent turn — `$ARGUMENTS` in\n"
                    "the template is replaced with everything typed after the command.\n\n"
                    "Template body:\n\n" + body
                ),
            )
            _REGISTERED.add(name)

    command_completions.cache_clear()  # the prompt's completion list must see the new names
    _PROBLEMS = problems
    return len(_REGISTERED), problems


def problems() -> list[str]:
    """Load problems from the most recent scan (startup warns with these)."""
    return list(_PROBLEMS)


def registered_names() -> set[str]:
    """Names of the currently loaded user commands — /help renders these as its `user` section
    (and subtracts them to know which COMMANDS entries are built-ins)."""
    return set(_REGISTERED)


# NOTE: no import-time scan. commands/__init__ calls load_user_commands() explicitly AFTER every
# built-in module has registered (two-phase load), so the collision check structurally sees all
# built-ins regardless of import order. Problems are surfaced by the startup warner in
# agent.main — never raised, a bad template can't block launch.

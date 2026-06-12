from pathlib import Path

from commands._framework import command, _print

# How much workspace evidence to show the drafting model.
_MAX_LISTING = 100

# Written when the workspace is empty or the LLM draft fails — still useful: the file's existence
# (and its section headings) teaches the user what to put there.
_TEMPLATE = """# SATURDAY.md

Standing instructions for this workspace. Saturday loads this file into context at the start of
every turn — keep it short and current.

## What this workspace is for

(Describe the project/notes/files that live here and what you're trying to do with them.)

## Conventions

- (e.g. "drafts live in drafts/, finished pieces in posts/")
- (e.g. "always write dates as YYYY-MM-DD")

## Things to remember

- (standing guidance: tone, formats, what to never touch, who this work is for)
"""

_DRAFT_PROMPT = """You are initializing SATURDAY.md — a standing-instructions file that a local
AI agent loads into context at the start of every turn it works in this workspace.

Below are the workspace's file listing and (when available) one-line summaries of its files.
Write a concise SATURDAY.md (under 60 lines) in markdown with exactly these sections:

# SATURDAY.md
## What this workspace is for      (1-3 sentences inferred from the files)
## Layout                          (the notable files/folders and what each holds — only what you
                                    can actually infer; skip boilerplate)
## Conventions                     (any naming/format patterns visible in the files; if none are
                                    evident, give 1-2 sensible placeholders the user can edit)

Be factual about what you can see and explicit about what you're guessing. Do not invent files.
Output ONLY the markdown file content, no preamble.

## File listing
{listing}

## File summaries
{summaries}
"""


def _workspace_listing(workspace: Path) -> list[str]:
    """Workspace-relative paths, capped. Best-effort — unreadable entries are skipped."""
    out = []
    try:
        for p in sorted(workspace.rglob("*")):
            if len(out) >= _MAX_LISTING:
                out.append("… (listing capped)")
                break
            try:
                rel = p.relative_to(workspace).as_posix()
            except ValueError:
                continue
            out.append(rel + ("/" if p.is_dir() else ""))
    except OSError:
        pass
    return out


@command(
    "init",
    "Survey the workspace and draft SATURDAY.md (standing per-workspace instructions).",
    usage="/init [--force]",
    details="""
The workspace is Saturn's sandboxed working area — the directory the file tools read and write,
at paths.workspace in config.yaml (database/workspace under the install by default). It is NOT
the directory you launched Saturn from, and /init never touches your current directory. To get
real files into Saturn's view: ingest them into the knowledge base with /docs add <path>, drop a
file onto the prompt (drag-and-drop offers ingest/attach), or copy them into the workspace.

/init creates SATURDAY.md at that workspace root — the per-workspace instructions file (the
CLAUDE.md equivalent). The grounding node loads it into context EVERY turn, so whatever it says
is standing guidance for the agent: what this workspace is for, its layout, your conventions.

/init surveys the workspace (file listing + the manifest's file summaries) and drafts the file
with the utility model; if the workspace is empty or the model is unavailable, it writes a
sensible template instead. Either way: open it and edit — it's your file, the draft is a start.

Refuses to overwrite an existing SATURDAY.md unless --force is passed.
""",
)
def _init(ctx, args):
    from config import get_config

    force = any(a in ("--force", "-f") for a in args)
    workspace = get_config().path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "SATURDAY.md"
    if target.exists() and not force:
        _print(f"  SATURDAY.md already exists at {target} — edit it directly, or re-draft "
               "with /init --force.")
        return

    listing = _workspace_listing(workspace)
    content = None
    # Only worth an LLM call when there is something to look at; an empty workspace gets the
    # template, which explains itself better than a model guessing at nothing.
    if [e for e in listing if e != "SATURDAY.md"]:
        try:
            from langchain.messages import HumanMessage
            from core.llms import get_model
            from stores.document_registry import read_workspace_manifest

            _print("  surveying the workspace and drafting SATURDAY.md…")
            prompt = _DRAFT_PROMPT.format(
                listing="\n".join(listing) or "(empty)",
                summaries=read_workspace_manifest().strip() or "(none)",
            )
            draft = str(get_model("utility").invoke([HumanMessage(content=prompt)]).content).strip()
            # Models love to wrap file output in a code fence — unwrap it.
            if draft.startswith("```"):
                lines = draft.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                draft = "\n".join(lines).strip()
            if draft.startswith("#"):
                content = draft + "\n"
        except Exception as exc:
            _print(f"  draft failed ({exc}) — writing the template instead.")

    target.write_text(content or _TEMPLATE, encoding="utf-8")
    kind = "drafted from the workspace contents" if content else "template"
    # Full absolute path on purpose: the workspace is Saturn's sandboxed area, not the cwd a
    # terminal user expects — a bare basename here left people unable to find the file they
    # were just told to edit.
    _print(f"  wrote {target} ({kind}).")
    _print("  this is Saturn's sandboxed workspace, not your current directory.")
    _print("  it now loads into context every turn — open it and make it yours.")

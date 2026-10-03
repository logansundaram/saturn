# Skills — procedures the user or the agent writes as markdown

Date: 2026-10-03. Status: **approved 2026-10-03 — nothing built.** Closes pivot #8.
Plans: `../plans/2026-10-01-skills.md` (Phase 1, revised for
`/skills create|delete`) and `../plans/2026-10-03-create-skill.md` (the agent's tool). It
changes three decisions of the 2026-10-01 plan (below).

## The problem

Saturn holds standing instructions (`SATURN.md`), hooks and `/policy`, but not a *procedure*:
"do my weekly review", "file this receipt". Today the user retypes it, or parks it in
`SATURN.md` where it rides every turn. `dogfood.md` §13 asks for it by name ("Create a skill
called weekly-review that does what we just did").

The ask (2026-10-03): a skill is a markdown file in the Claude Code shape; `/skills` lists them
and creates and deletes one; and the file is simple enough that **the agent can write one too**.

## What the 2026-10-01 plan already decided, and what changes

Unchanged, and not re-argued here (the plan's Design section has the reasoning):

- **The file.** `$SATURN_HOME/skills/<name>/SKILL.md` or flat `<name>.md`; a launch folder's
  `.saturn/skills/` wins over the global one. Optional YAML frontmatter (`name`, `description`,
  `disable-model-invocation`), then a markdown body. The file name is the name:
  `^[a-z0-9][a-z0-9-]{0,63}$`. Read from disk on every use, no cache.
- **Running one.** Typing `/<name> [request]` (REPL or `saturn -p`) is an ordinary turn with the
  body in that turn's DYNAMIC grounding: zero extra model calls, the cached prefix untouched,
  nothing carried into the next turn. A built-in command of the same name always wins.
- **A skill never changes the gate.** `allowed-tools` and every other unknown key is ignored.
- **The model choosing a skill by itself** (manifest + `use_skill`) stays the plan's separate,
  benchmark-gated Phase 2. Not in this spec.

Changed:

| Plan (2026-10-01) | This spec |
|---|---|
| `/skills [show <name> \| new <name>]` | `create` is the canonical spelling (`new` still works); adds `/skills delete <name>` |
| The agent never writes a skill; "save what we just did as a skill" is out of scope | The agent writes one through a single gated tool, `create_skill` |
| Skills folders are refused to every write | Still refused to the **file tools**; `create_skill` and `/skills` are the only writers |

## Decisions (brainstormed 2026-10-03)

- **The agent writes skills through one dedicated, always-asking tool, `create_skill`** (the
  user's name for it, 2026-10-03; `/skills create` is the by-hand path, the tool is the agent's). Chosen over: a draft
  queue with `/skills review` (the skill is unusable until a second step, and it needs a tool
  schema anyway); lifting the file-tool refusal (an always-allow on `write_file` would then
  cover skills, and a small model would hand-write the path and the frontmatter); and
  user-only (drops the ask). A dedicated tool takes three plain strings and builds the file
  itself, so a 9b never writes YAML.
- **Verbs, not flags.** `/skills create <name>` and `/skills delete <name>`, like `/memory add`
  and `/docs rm`. `delete` accepts every shared `REMOVE_VERBS` spelling; `create` also answers to
  `new` and `add`.
- **Delete is the user's alone**, and it is a move to the Trash.

## Design

### `/skills` (`commands/skills.py`)

```
/skills                    list: /name · description · scope · who wrote it; the two folders; problems
/skills show <name>        path, notes (ignored keys, unused files beside it), body
/skills create <name>      write a commented template to ~/.saturn/skills/<name>/SKILL.md, print the path
/skills delete <name>      move the skill to the Trash after a y/N
/<name> [request]          run it          /<name> --help   show it instead
```

- **List** adds one column to the plan's table: `you`, or `saturn · #412` for a skill
  `create_skill` wrote (the `origin` key, below), so `/trace why #412` answers "where did this
  come from". The empty state says both ways in: `/skills create <name>`, or ask Saturn to save one.
- **Create** is the plan's `new`: refuses a malformed name, a built-in command's name, and a name that
  exists; global folder only.
- **Delete** resolves the name exactly as running it does (the workspace skill wins). It prints
  the path and, for a folder skill, every other file in the folder, then asks
  `move to the Trash? [y/N]`. A folder skill moves as a folder; a flat skill as its file. It
  reuses the Trash helpers `delete_file` uses (`tools/files._trash_dir`, `_trash_slot`,
  `_trash_info`). When a global skill of the same name was shadowed, it says that one now runs.
  Off a TTY it refuses (`commands/_utils._stdin_is_tty`). It is not a `/undo` entry: `/undo`
  is the record of what the agent changed; the user restores from the Trash.

### `create_skill` (`tools/skills.py`)

```python
@register_tool("side_effecting")
def create_skill(name: str, description: str, steps: str, replace: bool = False):
    """Creates a skill: a reusable procedure the user runs later by typing /name. Use ONLY when
    the user asks to create, save or change a skill ("save that as a skill called
    weekly-review"). name: lowercase letters, digits and hyphens. description: one line saying
    what it does and when to use it. steps: the procedure as a numbered markdown list, written
    as instructions to yourself. To change a skill that exists, call with replace=true and the
    COMPLETE new steps. The user reads the whole skill before it is saved."""
```

- **One renderer.** `core/skills.render(name, description, steps, origin) -> str` builds the
  file text (frontmatter through `yaml.safe_dump`, so a colon or quote in the description
  cannot break it). The tool writes exactly that string and the gate shows exactly that string.
  `_parse(render(…))` round-trips name, description and body: pinned by a test.
- **Forgiving a small model.** Before any check, the name becomes a file name
  (`/Weekly Review` → `weekly-review`, `core/skills.draft_name`) and a JSON list of steps
  becomes a numbered list (`steps_text`). The hygiene applies both before the gate, so the
  human reads the call that will run.
- **Where.** The global folder only. A new skill is `<name>/SKILL.md`; `replace=true` rewrites
  the existing global file where it is (folder or flat). A workspace skill is never the
  agent's to write: those arrive with a project and the user edits them by hand.
- **`origin`.** The tool adds `origin: saturn run=<id> <date>` to the frontmatter
  (`stores.trace.current_run_id`). It is a provenance pointer for `/skills`, not a security
  claim: a hand edit may keep or drop it. It joins the loader's known keys.
- **One problem check.** `core/skills.draft_problem(name, description, steps, replace,
  builtin) -> str | None`, pure, called twice: by the agent's hygiene before the gate, and by
  the tool itself (a direct invoke, the benchmark, and the window between gate and write). It
  names, in this order:
  1. a name that is not a skill name (the rule is quoted back);
  2. a name a built-in command owns (`commands._framework.resolves`, passed in as `builtin`
     the way `invocation(builtin=)` already takes it, so `core/skills` stays a leaf);
  3. an empty description, or one over 200 characters;
  4. empty steps, or steps over `DRAFT_CAP = 3000` characters — refused, never truncated: what
     is shown is what is saved, and 3,000 characters is what a person will read at a prompt;
  5. a workspace skill of that name in this folder (the path is named; the user edits it);
  6. `replace=false` and the name exists (next bullet).
  `replace=true` on a name that does not exist is not a problem: it creates.
- **Changing a skill.** The model has no way to read a skill it did not just run, so the
  exists case hands it the text: the hygiene answers the call, before any gate, with
  `/<name> already exists — its current text: … To change it call create_skill with replace=true
  and the complete new steps.` That answer is stamped `done`, not `error`: the call did a job
  (it read the skill back), so it does not wake the adaptive think or land in the incidents
  note. The second call differs in `replace`, so the stall guard does not catch it.
- **The write.** The tool refuses unless `toolspec.human_approved()` says a person approved
  this call (a save can never ride a policy mistake). Then: `textutil.visible_controls` over description and steps (a saved skill holds no
  live escape), the user's `before-write` hooks (a veto raises `ToolError`),
  `stores.snapshots.snapshot_file` (so `/undo` takes a save back: a new skill is removed, a
  replaced one restored — entries carry absolute paths, so the folder outside the workspace
  round-trips), an atomic write (temp file + rename), `after-write` hooks. The result tells the
  model the path and that the user runs it by typing `/<name>`.

### The gate (`trust/policy.py`, `nodes/approval.py`, `tui/ui/approval.py`, `app/headless.py`)

- **`create_skill` joins `ALWAYS_ASKS`.** No tier, open gate, risk override or always-allow lets
  it through, and through `NO_BLANKET_GRANT` the gate's `a` never grants it. `ALWAYS_ASKS` was
  written for "sends your words to another person"; its three readers say "send" today. It
  becomes a mapping `name -> (what, why)`, read through `policy.always_asks_what(name)` and
  `always_asks_why(name)`:
  - `nodes/approval.py`: `SEND_NOTE` is replaced by the per-tool reason. For `create_skill`:
    "this saves a procedure Saturn will follow as your own words every time the skill runs;
    saving a skill always asks, whatever the policy".
  - `tui/ui/approval.py` `_grant_note`: "there is no always-allow for it", per tool.
  - `app/headless.py`: the denial line per tool ("saving a skill always needs a human to read
    it first, and headless mode has none (--yolo does not cover it)").
  `send_message`'s gate note and always-allow note stay byte-for-byte what they are; its
  headless line now reads "a send always needs a human…" (it said "sending to another person").
- **The whole skill is on screen.** A new `_BESPOKE["create_skill"]` renderer, because the two
  generic views both hide text: the full-args view clamps a value at 2,000 characters head +
  tail, and the diff view folds after 60 rows. A folded middle is where a planted step would
  sit. New skill: the destination path, then every line of `render(…)`. Replace: the unified
  diff of the old file against `render(…)`, every row. A long line is wrapped, never cut at
  the terminal width. Bidi and zero-width characters show as
  code points, as everywhere at the gate.
- **A note when outside content is in the conversation.** When
  `nodes/approval.provenance(state)` reports untrusted content (a web page, a file, an
  attachment, mail) entered before the call, the prompt adds: "external content entered this
  conversation before this skill was drafted — read each step as if a stranger wrote it." A
  pending quarantine flag shows its usual banner as well. It is a note, not a refusal:
  "summarise this page and save the method as a skill" is a fair request.
- **Headless** refuses `create_skill` even with `--yolo`. `/skills create|delete` are REPL commands.

### The file tools (`tools/files.py`)

The plan's Task 2 stands: `write_file`, `edit_file`, `move_file` and `delete_file` refuse any
target inside a skills folder, and moving a folder that holds one. Two wording changes route
the model instead of stranding it: a refused write says "to save a skill call create_skill", a
refused delete says "the user deletes a skill with /skills delete <name>". `run_shell` can
still reach the folder, gated `destructive`, exactly as it can reach `hooks.yaml` today.

## Trust

A skill is instruction text the model follows as the user's own, so the skills folder is a
persistence channel like memory. The rule this spec keeps is memory's: **nothing lands there
without a person's yes to that exact text.**

| Path into the folder | What stands in the way |
|---|---|
| The user's editor, `/skills create`, `/skills delete` | Nothing: the user's own action |
| `create_skill` | Hygiene (well-formed, not a built-in, not a blind overwrite) → the always-ask gate with the full text → the tool's own `human_approved()` check and the same problem check again at the write. Refused headless |
| `write_file` / `edit_file` / `move_file` / `delete_file` | Refused, even approved |
| `run_shell` | The `destructive` gate; opaque, like every shell write (`research.md` T5) |
| A project's `.saturn/skills/` | Runs only when its name is typed; the agent never writes it |

What a saved skill can do later is unchanged from the plan: every call it leads to still asks
`trust/policy.approves`, and `/skills` shows who wrote it. No registry, no install-from-URL.

## Cost, and the go/no-go

`create_skill` is one more schema in a 44-tool catalog (counted 2026-10-03; about 6k tokens of
schema when it was 42, and this one adds roughly 150 — an estimate, to be measured). The prefix
changes once, at upgrade. Two of the 4b's five stable
loop-benchmark misses are already wrong-tool picks, so the cost is measured, not assumed:

- Build order: the plan's Phase 1 plus `/skills delete` ships first and stands alone. `create_skill`
  is its own phase on top.
- Two new loop tasks: `skill_create` ("save that as a skill called …" → `create_skill` with a valid
  name; `bench_approver` declines it because it is in `ALWAYS_ASKS`, so a run never writes the
  user's folder, and the task is graded on the call) and `skill_chat` (a question *about* a
  procedure makes no `create_skill` call).
- Go: on the 9b, no task that passed the Phase-1 baseline stops passing, and `skill_create`
  passes (three runs each; the exact thresholds are in the plan's Task 6). No-go: the tool is unbound and the answer to "save this as a skill" is the drafted
  text plus `/skills create`. A 4b-only miss is a model limit, not a no-go.

## Tests (offline, all in `tests/test_skills.py`)

- The plan's loader, control-folder, grounding and invocation tests, unchanged.
- `/skills delete`: a folder skill and a flat one land in the (throwaway `HOME`) Trash; `n`
  and off-TTY change nothing; deleting a workspace skill reports the global one now runs; an
  unknown name; every `REMOVE_VERBS` spelling.
- `render` / `_parse` round-trip, including a description with `: `, quotes and a `#`.
- `draft_problem`: each of the six problems, in order; `replace=true` on a new name creates.
- The hygiene answers a malformed draft without a gate; the exists
  answer carries the current text and is stamped `done`; no incident line for it.
- `create_skill` asks under `/policy open`, a risk override, and after
  `a`; `always_asks_what` / `always_asks_why` for both members; `send_message`'s strings unchanged.
- Gate rendering: a 3,000-character draft and an 80-row replace diff appear in full; the
  untrusted-content note appears only when provenance says so.
- Headless: `-p "save a skill …" --yolo` is denied with the skill wording.
- `/undo` after a save removes a new skill and restores a replaced one.
- The write refuses when a `before-write` hook vetoes; the saved bytes equal `render(…)`.

## Files

Beyond the plan's file table: create `tools/skills.py` (`create_skill`; the plan's Phase 2
`use_skill` lands in the same module later); `core/skills.py` gains `render`, `draft_problem`,
`DRAFT_CAP` and the `origin` key; `commands/skills.py` gains `delete`; `trust/policy.py`
(`ALWAYS_ASKS` mapping, `always_asks_why`); `nodes/approval.py` (per-tool note, the
untrusted-content note); `nodes/agent.py` (`_hygiene`); `tui/ui/approval.py` (`_BESPOKE`
entry, `_grant_note`); `app/headless.py` (denial wording); `tools/registry.py` (import);
`benchmark.py` (two tasks). Docs: `CHANGELOG.md`, `CLAUDE.md` (Trust stack: `ALWAYS_ASKS`;
Tools: the skills folder's one writer), `docs/ARCHITECTURE.md`, `docs/pivot.md` #8.

## Not in this spec

- The model loading a matching skill by itself (the plan's Phase 2).
- An agent tool to delete a skill, or to write a workspace skill.
- `$ARGUMENTS`, typed parameters, `` !`cmd` `` injection, supporting scripts run from a skill folder.
- Opening the new file in an editor from `/skills create` (it prints the path).
- Skills as the prompt of a scheduled run (`research.md` P1).

## Assumptions made without asking

1. `create_skill` is `ALWAYS_ASKS`-strict rather than `remember`-strict (always-allowable). A
   remembered fact is one line; a skill is a procedure that can chain tool calls.
2. Agent drafts are capped at 3,000 characters; hand-written skills keep the loader's 6,000.
3. `replace` may rewrite a skill the user wrote by hand: the gate shows the diff, and `/undo`
   restores it. `origin` then names Saturn as the last writer.
4. An agent-saved skill is treated like any global skill once saved (in Phase 2 it would be
   model-invocable unless marked manual-only): the user read and approved its text.
5. `/skills delete` takes one name per call and always asks; there is no `--force`.

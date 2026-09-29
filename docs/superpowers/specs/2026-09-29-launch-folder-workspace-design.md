# Work where you launched: the launch folder is the workspace

_2026-09-29. Pivot item #1 ("work where you launched"). Decided in conversation: the launch
folder only, other folders by an explicit command, and home treated like any other folder._

## Why

Saturn's file tools are jailed to one fixed folder, `database/workspace` under the install.
"Summarize the notes in this folder" or "rename these photos by date" fails unless the user
first copies files into Saturn's private folder. Claude Code's most important daily property is
that it works in the directory you start it from. Saturn should do the same: `cd` somewhere,
run `saturn`, and the agent works on the files there.

## Decisions

| Question | Decision |
|---|---|
| What can the file tools reach? | The launch folder only, plus folders the user adds for the session with `/add-dir`. |
| How does the agent reach another folder? | It can't on its own. A refused path tells the model to ask the user to run `/add-dir <folder>`. A gate prompt that grants access on the fly is out of scope. |
| Launched from `~`? | Home is a folder like any other: the whole home directory is the workspace. |
| Launched from `/`? | Too broad. Saturn falls back to home and says so in the banner. |
| What stays global? | Memory, config, traces, the knowledge base, sessions and snapshots. They describe the user, not the folder. |
| Does the gate change? | No. Every write and every shell command faces the same policy as today. |

## The workspace module (`core/workspace.py`)

One module owns where Saturn works. It holds two things for the process:

- **The root**: the launch folder, an absolute resolved path, set once at startup.
- **Extra roots**: folders added with `/add-dir`, in the order added. Session only, never
  persisted.

Its interface:

```python
set_root(path) -> Path            # startup; "/" falls back to home; returns the root used
root() -> Path                    # the root; if never set, config.path("workspace")
extra() -> list[Path]
roots() -> list[Path]             # [root()] + extra()
add(path) -> Path                 # /add-dir: must exist, be a directory, not "/"; idempotent
remove(path) -> bool              # /rm-dir: False when it was not an extra root
resolve(path) -> (Path, str|None) # (target, refusal) — the ONE containment check
relative(target) -> str           # a display path: root-relative, else ~-relative, else absolute
reset()                           # tests
```

**The fallback keeps everything else working.** When nothing called `set_root` (the offline
tests, the benchmark, any tool imported outside the app), `root()` is the configured
`paths.workspace`, exactly today's behavior. The `isolated_paths` fixture therefore keeps
isolating file tests with no change. An autouse fixture in `tests/conftest.py` calls `reset()`
around every test, so a root set in one test never leaks into another.

### Resolution rules (`resolve`)

1. Expand `~`. A relative path joins onto `root()`.
2. Resolve symlinks and `..` (`Path.resolve()`).
3. The target must be inside one of `roots()` (`is_relative_to`). Otherwise the refusal is:
   `Outside the folders Saturn can reach ({roots as display paths}). Ask the user to run
   /add-dir {folder} to allow it.` — `{folder}` is the requested path itself when it is an
   existing directory, else its nearest existing ancestor. When that would be the filesystem
   root (which `/add-dir` refuses), the second sentence is dropped and the refusal ends after
   the reach list.

The refusal is a tool observation, so the model relays it. The containment check exists once,
here. The file tools' `_resolve` becomes a thin wrapper over it.

## What changes where

| Consumer | Today | After |
|---|---|---|
| `tools/files.py::_resolve` | joins onto `config.path("workspace")` | `workspace.resolve()`; results shown with `workspace.relative()` |
| `tools/files.py` search and find | walk everything not hidden; `find_files` rglobs, then filters | one pruned `os.walk` for both, see "Walking home" below |
| `tools/shell.py::run_shell` | `cwd=config.path("workspace")` | `cwd=workspace.root()` |
| `tui/ui/approval.py` diff preview | files `_resolve` | unchanged (it already calls files `_resolve`) |
| `nodes/ground.py` instructions | `SATURN.md` / `SATURDAY.md` in the configured workspace | the same names in `workspace.root()` |
| `nodes/ground.py` stable half | no location | a `### Working folder` section: the root, and any extra roots |
| `core/context.py::clean` | collapses the configured workspace path | collapses `workspace.root()` |
| `core/messages.py` agent prompt | "a file listed under Workspace files is read with read_file" | "Relative paths are in the working folder shown in the grounding. For a folder outside it, ask the user to run /add-dir." |
| `commands/knowledge.py` `/init` | drafts into the configured workspace | drafts into `workspace.root()` |
| `stores/snapshots.py` | manifest stores workspace-relative paths | stores absolute paths (below) |
| `agent.py::main` | nothing | `workspace.set_root(Path.cwd())` before the REPL or headless run |
| startup banner | no folder | `working in <path>`, plus the `/` fallback note |

The working-folder section lives in the stable half: the root is fixed for the session, so the
prompt prefix cache still holds. `/add-dir` and `/rm-dir` change the stable half, so the next
turn misses the cache once, like editing `SATURN.md` does today.

### Walking home

Launched from `~`, a search walks the whole home directory. Both `search_files` and
`find_files` use one pruned walk:

- Skip hidden entries (as today), a `Library` folder directly under the home directory, and
  heavy build or dependency folders anywhere:
  `node_modules`, `.git`, `__pycache__`, `.venv`, `venv`, `Pods`, `DerivedData`.
- Stop after visiting a fixed number of files (a module constant, 50,000) and say so in the
  observation: `stopped after 50,000 files; narrow the directory or pattern`.
- `find_files` keeps its glob semantics: a bare pattern matches names anywhere under the
  directory; a pattern with `/` matches the path relative to the directory.

### `/undo` across folders

New snapshot batches record each file's **absolute** path. `/undo` restores exactly that file,
whatever the current launch folder is, because the user typed `/undo`. It is not a model
action, so the containment check does not apply, the same way `@file` mentions are
unsandboxed. A manifest entry without an absolute path (written before this change) resolves
against `config.path("workspace")`, where it was recorded. The saved bytes for a new entry live
at `files/<absolute path without its leading slash>` inside the batch directory; legacy entries
keep `files/<relative path>`.

## Commands

Both live in a new `commands/workspace_dirs.py`, registered with `@command`, and accept
`--help`.

- **`/add-dir <path>`**: adds an existing folder for this session. It must exist, be a
  directory, and not be `/`. Adding the root or an existing extra root is a no-op with a note.
  `/add-dir` with no argument lists the root and extra roots.
- **`/rm-dir <path>`**: removes an extra root. Removing the launch folder is refused ("that's
  the folder Saturn was started in"). An unknown path says it wasn't added.

`/workspace` and `/ws` stay in `_RENAMED`, pointing at `/docs`, as today.

## Error handling

- A refused path is an ordinary tool observation with `saturn_status: error`, so the loop's
  existing handling applies. The thinking-on-evidence rule makes the next pass think, which is
  where the model decides to ask the user for `/add-dir`.
- `set_root` never fails: an unreadable or vanished cwd falls back to home, then to the
  configured workspace, and the banner says which was used.
- `/add-dir` refusals are printed, never raised.
- read_file and search_files are untrusted (quarantine-scanned): the launch folder holds downloaded and third-party files.

## Testing

All offline, in the files that own the features:

- **`tests/test_workspace.py` (new)**: relative, absolute-inside and `~` paths; `..` and
  symlink escapes refused; extra roots allowed; the refusal names `/add-dir` and the parent
  folder; `set_root("/")` falls back to home; `root()` falls back to the configured path when
  unset; `add`/`remove` validation and idempotence; `relative()` display forms.
- **File tools**: read, write and edit honor a root set with `set_root`; a path in an extra
  root works; search and find skip `Library` and `node_modules` and stop at the visit budget
  (the budget monkeypatched small).
- **Shell**: `run_shell` runs with `cwd` at the root.
- **Ground**: the working-folder section appears in the stable half; `SATURN.md` is read from
  the root.
- **Undo**: a batch recorded under one root restores correctly after `set_root` moves to
  another; a legacy relative manifest still restores.
- **Commands**: `/add-dir` and `/rm-dir` happy paths and refusals, `--help`, and `/help`
  listing.
- **Benchmark**: both benchmarks reset the root, so they plant and grade in the configured
  scratch workspace, never the folder they were run from.

Then one live check on the 9b: launch from a scratch folder and from `~`, ask for a file in
the folder, ask for one on the Desktop (expect the `/add-dir` suggestion), run `/add-dir
~/Desktop`, and ask again.

## Out of scope

- A per-folder `/resume`.
- Reading `SATURN.md` from parent folders.
- A gate prompt that grants folder access on the fly.
- A folder listing in the prompt.
- Persisting added folders across sessions.
- A `--workspace` flag. `cd` does the job, including in scripts.
- **A time budget on content search (known slow case).** Measured 2026-09-29 on a real home
  folder with the pruning above: a name search walks 27,090 entries in 0.7 s, but a content
  search that matches nothing reads every text file (17,101 files, 796 MB) and takes 80 s. The
  entry cap never fires because home holds fewer than 50,000 entries. Startup is unaffected —
  nothing is walked until the model calls a search tool. The fix when it matters: stop
  `search_files` after a few seconds and add the same "narrow the directory or pattern" note.

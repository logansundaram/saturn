# Toolkits — tools in groups the user can turn off

Date: 2026-10-05. Status: **built 2026-10-05** (read "As built" at the end). Not yet
measured: no benchmark run with a trimmed catalog. No plan document: the design below was
the plan, built test-first in the order of "Build order".

## The goal

`/tools` shows the tools in groups — toolkits — and the user turns a whole group off or on.
A toolkit that is off is **unbound**: its schemas leave the prompt, the model cannot call its
tools, and the agent can never reach that app.

Three things point at this, independently:

- **The user's notes** (`notes.md`, twice): "make sure users can turn off the tools that they
  dont want to use", and toolkits a user enables by the functionality they want.
- **The loop benchmark** (2026-10-04, the shipped think policy): `wrong_tool` is the 4b's most
  common failure tag, five or six a run, against a catalog of 45 tools. A switch is also what
  lets the benchmark measure a trimmed catalog — `research.md` E1(a).
- **Friction.** A person who never texts from Saturn should not carry three Messages schemas
  in every prompt, nor ever meet a Full Disk Access dialog because the model reached for one.

## Decisions (Logan, 2026-10-05)

1. **Every toolkit starts on.** Nothing changes for a current user. Whether any should start
   off is decided after the benchmark has measured a trimmed catalog, not before.
2. **A small core cannot be turned off:** `plan`, `ask_user`, `remember`, `recall`,
   `calculate`, `current_time` — the tools the engine and the prompt lean on. Turning memory
   off is incognito's job (`2026-10-04-know-the-user-design.md` §4), not this switch's.
3. **Saturn names the switch.** With Calendar off, "what's on my calendar tomorrow?" is
   answered with "the calendar toolkit is turned off — `/tools on calendar`". The model is
   told which toolkits are off in one prompt line; no extra call, no mid-turn prompt.
4. **Toolkit-level only.** No switch for a single tool.
5. **An older `config.yaml` gains the `toolkits:` block by append** on the first toggle; no
   existing line is touched.

## The toolkits

| Key | Tools | Switchable |
|---|---|---|
| `core` | `plan`, `ask_user`, `remember`, `recall`, `calculate`, `current_time` | no |
| `files` | `read_file`, `write_file`, `list_directory`, `edit_file`, `move_file`, `delete_file`, `search_files`, `find_files` | yes |
| `web` | `web_search`, `web_extract` | yes |
| `shell` | `run_shell` | yes |
| `knowledge` | `search_knowledge_base` | yes |
| `notes` | `search_notes`, `read_note`, `create_note`, `append_note` | yes |
| `calendar` | the four calendar tools | yes |
| `mail` | the six mail tools | yes |
| `contacts` | `search_contacts` | yes |
| `reminders` | the three reminder tools | yes |
| `messages` | `find_group_chats`, `send_message`, `read_messages` | yes |
| `shortcuts` | `list_shortcuts`, `run_shortcut` | yes |
| `desktop` | `read_browser_tab`, `finder_selection` | yes |
| `notifications` | `schedule_notification` | yes |
| `skills` | `create_skill` | yes |
| `mcp:<server>` | that server's tools | listed; managed by `/mcp` |

The key is what the user types. A toolkit is an app or a capability, so it lines up with the
macOS permission it needs.

No hard dependencies. Messages without Contacts still texts a number the user typed; it
cannot look a name up. Turning `contacts` off while `messages` is on prints a note that says
so, and nothing is refused.

## The command

```
/tools                      the toolkits: state, tool count, one line each
/tools mail                 that toolkit's tools, with risk tier and description
/tools off messages mail    turn toolkits off (saved to config.yaml)
/tools on messages          turn one back on
/tools off shell --session  this session only
/tools --all                every tool in one flat list, with its toolkit
```

Bare `/tools` is a readout, never a flip. `on` / `off` take one or more keys; an unknown key
is refused with the closest match and changes nothing, even when other keys on the line are
valid. `core` and an `mcp:` toolkit are refused with the reason. A toolkit already in the
asked state is said to be so. The change applies to the next request — the command is typed
at the prompt, so it never lands inside a turn. Turning a toolkit on never changes the gate:
its tools face the same tiers, grants and `ALWAYS_ASKS` as before.

## How it is built

### Where a toolkit is declared (`tools/toolspec.py`)

`TOOLKITS` is the one ordered table: key → label, a one-line description, and whether it is
core. `@register_tool(..., toolkit=None)` and `register_tool_object(..., toolkit=...)` record
each tool's toolkit. The default is the tool module's own name (`tools/mail.py` → `mail`), so
a new tool in an existing module needs no edit; only the exceptions say it (`calculate`,
`current_time`, `ask_user`, `plan`, `remember`, `recall` → `core`; `schedule_notification` →
`notifications`). A key that is not in the table raises at import, like an unknown risk tier.
An MCP server's tools register under `mcp:<server>`, a toolkit added to the table when the
server connects and marked as managed by `/mcp`.

`toolspec` stays a project-import-free leaf.

### The bound set (`tools/registry.py`)

- `registry.tool` stays THE list the model is bound to — now the tools whose toolkit is on.
  The agent node, the prefix prime and the startup banner read it and do not change.
  `tools_by_name` is the same set by name, so the tools node cannot execute a tool that is
  off.
- `all_tools` / `all_by_name` hold every registered tool, for `/tools`, `/policy risk` and the
  persisted risk overrides.
- `TOOL_RISK`, `DECLARED_RISK`, `RETRIEVAL_TOOLS`, the quarantine's untrusted set and its
  gated-name pattern keep covering every registered tool, on or off. Turning a toolkit off
  never shrinks the injection scan, and there is nothing to re-push on a toggle.
- `apply_toolkits()` recomputes the bound views from `toolkits.*` in the config, **in place**
  — every holder of the list or the dict sees the change, the way `/mcp reload` does it.
  `set_toolkits(keys, on)` sets the config in memory, applies, and rebinds the model
  (`core.llms.reset_models`).
- `toolkit_problems()` reads the config block for the startup report: an unknown key, a value
  that is not true/false (treated as on), `core` set false (ignored).

### The prompt (`core/messages.py`)

`_AGENT_SYS` stays one literal. With every toolkit on the prompt is that literal, byte for
byte — nothing is recomposed, so current benchmark numbers stay comparable. Each switchable
toolkit that the prompt names owns the sentences that name its tools; when it is off, those
sentences are cut (or, for `contacts`, replaced by "never guess a person's address or number
— ask the user for it"). A test pins that every such sentence occurs exactly once in the
literal, so a prompt edit that breaks a cut fails loudly.

When any toolkit is off, one bullet is added at the end of "How to work": which toolkits are
off, that their tools do not exist right now, and that `/tools on <name>` turns one on. The
text is a pure function of the set of off toolkits, so the prefix is stable between toggles.

The knowledge-base manifest in the stable grounding (`nodes/ground.py`) is left out when
`knowledge` is off.

### The backstop (`nodes/agent.py`)

Tool descriptions name each other (`send_message` names `search_contacts`), and earlier turns
may hold calls to a tool that has since been turned off, so the model can still emit one.
Deterministic, no gate, no model call:

- **Hygiene** answers a call to a tool whose toolkit is off before any other check:
  `TOOLKIT_OFF_TEXT` — not executed, the toolkit is off, tell the user `/tools on <key>`.
  Stamped `error`, so the next pass is a recovery step and the incidents note carries it,
  worded for the user: `not run: the <key> toolkit is turned off (/tools on <key>)`.
- **The foreign-arguments redirect** (`tool_for_args`) never points the model at a tool that
  is off.
- **The tools node** finds no such tool in `tools_by_name` and runs nothing.

### Saving (`config.py`, `config.default.yaml`)

`toolkits:` in the template holds one `key: true` line per switchable toolkit. `/tools on|off`
persists by default through the existing single-line edit; `--session` keeps it in memory.
A `config.yaml` without the block gets it appended once (`config.append_block`), then edited
like any other scalar. MCP servers are not in the block.

### The cached prefix

A toggle changes the system text and the bound schemas — the whole cached prefix. The
command starts a prime (`core.prime.start_priming`) as soon as the change is applied, so the
next request does not pay for re-reading it. Between toggles nothing about the prefix moves.

### Benchmark

`benchmark.py --off messages,shortcuts` turns toolkits off in memory for the run and records
`toolkits_off` in the report. A loop task that names a tool from an off toolkit in what it
expects is skipped and counted apart from passes and failures.

## Tests (offline)

`tests/test_toolkits.py`, one behaviour each: the table covers every registered tool and the
core is exactly the six; an unknown key raises at registration; off unbinds in place and on
restores registration order; the risk tables and the quarantine's sets do not move; core and
`mcp:` toolkits refuse; the config problems; the prompt is the literal with nothing off, each
cut occurs once, the off line names the keys; the manifest follows `knowledge`; hygiene's
answer, the incidents wording, the redirect, the tools node; the append and the persist;
every `/tools` form and `--help`; `--off` in the benchmark. `tests/conftest.py` turns every
toolkit on around each test, so a developer's own `config.yaml` cannot fail the suite.

## Not in this version

A per-tool switch · default-off toolkits or a first-run picker (after the measurement) ·
toggling an MCP server from `/tools` · a headless flag · downloadable toolkits · an offer to
turn a toolkit on in the middle of a turn.

## Build order

1. The table and the tagging. 2. The bound set and the switch. 3. The prompt and the
grounding. 4. The backstop. 5. Saving. 6. The command. 7. `/policy risk` and MCP. 8. The
benchmark flag. 9. Startup report, docs, changelog.

## As built (2026-10-05)

Built as designed. What a reader of the design would not guess:

- **`/tools --all` is grouped by toolkit**, in the table's order, not in registration order.
- **`--off` and the trust benchmark.** Only the loop benchmark skips a task whose toolkit is
  off. The trust probes need `files`, `knowledge` and the core; turning those off makes the
  probe miss, not skip.
- **`/config` keeps the bound set in step.** The prompt reads `toolkits.*` from the config
  and the bind reads the registry, so `/config reload` and `/config toolkits.<key> <bool>`
  re-apply the toolkits too. `toolkit_problems()` is read at launch only (interactive and
  headless).
- **One real check, both tiers** (2026-10-05, every app toolkit off, in memory): "What's on
  my calendar tomorrow?" was answered in one pass with no tool call — the 4b: "that toolkit
  is currently turned off. You can turn it on with `/tools on calendar`"; the 9b: "The
  calendar tool is turned off … run `/tools on calendar` first." `calculate` still ran. Two
  requests are not a measurement.

Open: the loop benchmark with a trimmed catalog on both tiers (`--off`, three runs), which
is what decides whether any toolkit should start off.

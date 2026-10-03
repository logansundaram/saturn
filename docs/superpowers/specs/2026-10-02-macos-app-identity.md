# Saturn as a macOS app identity — permissions held by Saturn, not the terminal

Date: 2026-10-02. Status: **spec, undecided.** Phase 0 can be built now; Phases 1–3 wait on the
decisions in the last section, and several of those wait on the probes in "What to probe first".
Claims marked _(probe)_ are believed but not yet verified on this Mac (macOS 27.0.1).

## The problem

Every native tool asks macOS for something, and macOS charges each request to the
**responsible process** — the app that launched the process chain. A `saturn` started in a
terminal is the terminal as far as TCC is concerned. So today:

| What the user sees | Triggered by |
|---|---|
| "**Visual Studio Code** wants to control Calendar" — one dialog per target app: Notes, Calendar, Mail, Contacts, Reminders, Messages, Safari/Chrome, Finder | `tools/applescript.run` (osascript) |
| A manual trip to System Settings → Full Disk Access for **the terminal**, then a terminal restart | `read_messages` (`~/Library/Messages/chat.db`) |
| Notifications credited to Script Editor | `notify/macos.py` (`osascript display notification`) |
| The menu bar icon running as a bare `python -m notify.menubar_app` | `notify/menubar.py:72` (LaunchAgent `ProgramArguments`) |

The costs:

- **It looks like terminal code, not a product.** The dialogs name VS Code / Terminal / iTerm.
  Nothing on screen says "Saturn" or explains why in Saturn's words.
- **It is per terminal.** Grant everything in Terminal.app, then launch from VS Code: every
  dialog comes back. A user who tries Saturn in two apps meets the whole set twice.
- **The prompts arrive mid-task.** The first `list_calendar_events` of a turn stalls on a
  modal dialog; Full Disk Access cannot be prompted at all — it is an error, a Settings trip
  and a restart, in the middle of a conversation.
- **It contradicts the trust thesis.** Full Disk Access for Terminal is Full Disk Access for
  every script, package post-install hook and `curl | sh` that terminal ever runs. Saturn's
  pitch is least privilege; its setup asks the user for the opposite.

## The goal

1. The permission dialogs and the System Settings rows say **Saturn**, with Saturn's icon and
   a usage sentence Saturn wrote ("Saturn reads your Messages history only when you ask it
   to").
2. A grant is made **once per Mac**, not once per terminal, and survives Saturn updates.
3. The user grants everything **up front, in one place**, with each item explained and its
   status shown — never as a surprise in the middle of a turn.
4. The terminal itself needs **no** special access. What Saturn can reach is Saturn's narrow,
   audited set of operations, not everything a shell can run.
5. The trust stack is unchanged: the gate, the egress ledger and the trace still see every
   action, in the CLI process, exactly as today.

Non-goals: a GUI chat window (Saturn stays terminal-native); the App Store (sandboxing would
forbid most of the tools); Linux/Windows (the protected tools are macOS-only already).

## What needs a permission, and what does not

| Area | Code | macOS permission | Moves behind the app identity? |
|---|---|---|---|
| Notes, Calendar, Mail, Contacts, Reminders (read + write) | `tools/notes.py`, `calendar.py`, `mail.py`, `contacts.py`, `reminders.py` | Automation, one per target app | **yes** |
| `send_message` | `tools/messages.py` | Automation (Messages) | **yes** — and stays the egress chokepoint (below) |
| `read_messages` | `tools/messages.py` | Full Disk Access | **yes** |
| `read_browser_tab`, `finder_selection` | `tools/desktop.py` | Automation (Safari / Chrome / Finder) | **yes** |
| Notifications | `notify/macos.py` | Notifications (today credited to Script Editor) | **yes** |
| Menu bar icon | `notify/menubar_app.py` | none, but it is the natural host process | **yes** — it becomes the app |
| File tools, `run_shell`, `/add-dir` | `tools/files.py`, `core/workspace.py` | Files & Folders (Desktop / Documents / Downloads) | **no** — the user launched Saturn in that folder from that terminal; the terminal owning that access is expected |
| `search_files` Spotlight | `tools/files.py` (`mdfind`) | none | no |
| Shortcuts | `tools/shortcuts.py` | none for the CLI; each shortcut's actions are credited to Shortcuts itself _(probe)_ | no |
| Web, MCP, Ollama | `tools/web.py`, `tools/mcp_client.py`, `core/llms.py` | none (outgoing connections need no TCC grant) | no |

So the boundary is clean: **everything that talks to another Mac app or to protected app
data** moves; everything that touches the workspace or the network stays.

## TCC facts the design rests on

- **Responsibility is inherited by children.** A process spawned by an app is charged to that
  app — that is the whole current problem, and also the way out: a process spawned by
  *Saturn.app* is charged to Saturn.app. `osascript` run by a Saturn-owned process prompts as
  Saturn _(probe — P1)_.
- **A process started by launchd (a LaunchAgent, or `open -a`) is its own responsible
  process.** That is how Saturn.app escapes the terminal. A process the terminal forks or
  execs is not.
- **An exec'd binary outside the bundle loses the identity.** A LaunchAgent whose program is
  the venv's `python` is identified as *python* (the shared Homebrew/framework interpreter that
  every other script also uses) — worse than the terminal. The responsible process must be an
  executable inside a signed bundle.
- **Grants key on the code signature.** Developer ID signed: keyed on team ID + bundle ID, so a
  grant survives updates. Ad-hoc signed (`codesign -s -`): keyed on the code hash, so every
  rebuild is a new app and every grant is lost _(probe — P2)_.
- **Hardened runtime needs an entitlement to send Apple events**
  (`com.apple.security.automation.apple-events`) plus `NSAppleEventsUsageDescription` in
  Info.plist, or the events fail without a dialog. Notarization requires hardened runtime.
- **Full Disk Access can never be prompted.** The app can only open the right Settings pane and
  notice when the switch has been flipped. Everything else (Automation, Contacts, Calendars,
  Reminders, Notifications) prompts once, with the bundle's usage string.
- **With a real bundle EventKit works again.** `docs/superpowers/specs/2026-09-06-macos-apps.md`
  rejected EventKit because a terminal-launched Python gets Calendar access only if the
  *terminal's* Info.plist has the usage key. Saturn.app has its own Info.plist, so EventKit /
  Contacts.framework become possible — and they are much faster than Apple events (Calendar
  measured 6.5 s warm over AppleScript) _(probe — P5)_.

## Phase 0 — onboarding, buildable now, needed by every option

This half removes the mid-task surprises whatever the architecture becomes, and stays as the
front door after it.

**`saturn setup`** (and `/setup` in the REPL; `--help` per the command framework): a single
screen, one row per permission:

```
 macOS permissions                                        Saturn needs these for…
 ✓ Automation · Notes        granted                      search_notes, create_note
 ✓ Automation · Calendar     granted                      list/create calendar events
 · Automation · Mail         not asked yet   [ask now]    list/read/draft mail
 ✗ Full Disk Access          off             [open Settings]   read_messages
 ✓ Notifications             granted                      /notify, schedule_notification
```

- **Status comes from a probe, not a guess.** Full Disk Access: try to open `chat.db`
  read-only. Automation: `AEDeterminePermissionToAutomateTarget` with `askUserIfNeeded=false`
  answers granted / denied / not-asked without a dialog _(probe — reachable from pyobjc?)_; a
  target app that is not running answers -600, which the screen reports as "not running"
  rather than guessing.
- **"Ask now"** fires the smallest harmless Apple event per app (`count notes`, `name of
  calendar 1`) so all the dialogs arrive here, together, before the user needs them.
- **"Open Settings"** opens the exact pane:
  `x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles` (also
  `Privacy_Automation`, `Privacy_Contacts`, `Privacy_Calendars`, `Privacy_Reminders`) _(probe —
  the anchors still resolve on macOS 27)_, then polls the probe and ticks the row when the
  switch is flipped.
- The screen names **the app that will hold the grant**: the terminal today (read from
  `__CFBundleIdentifier`, falling back to `TERM_PROGRAM`), Saturn once Phase 2 ships.
- First interactive launch on macOS offers it once (a line, not a modal); declining is
  remembered. Headless never runs it.

**Errors name the app.** The Full Disk Access error in `tools/messages.py` and the
Automation-denied error in `tools/applescript.py` say "Give **Visual Studio Code** Full Disk
Access…" and point at `saturn setup`, instead of "this terminal".

Tests: the probes behind one seam (`setup._probe`), replaced in tests like `applescript._run`;
the bundle-id → display-name map pinned.

## The options for the app itself

All three are judged against the goal list above. They differ in where Saturn's code runs
relative to the app identity.

### Option A — a helper app that holds the permissions (IPC)

```
terminal ── saturn (CLI, agent, gate, egress ledger, trace)
                │  unix socket, ~/.saturn/run/helper.sock (0700 dir)
                ▼
launchd ── Saturn.app (LSUIElement: menu bar icon, no Dock icon)
                ├─ osascript / EventKit  → Notes, Calendar, Mail, Contacts, Reminders, Messages
                ├─ chat.db reader        → read_messages
                └─ notifications
```

The CLI keeps everything it has today. The protected tools become thin clients: after the
gate approves the call, `tools/calendar.py` sends `{"op": "calendar.list", "args": {…}}` to the
helper and turns the reply into its observation, or a `ToolError`. The helper is the menu bar
app, grown up: launched by the login LaunchAgent it already has (now pointing at the bundle,
via `BundleProgram` / `open -a`), so it is its own responsible process.

- **Meets every goal.** Dialogs say Saturn; one grant per Mac; the terminal needs nothing.
- **The helper is a confused-deputy risk, and its API is the only fence.** Any process running
  as the same user can connect to the socket. If the helper exposed "run this AppleScript" or
  "run this SQL", every script on the Mac would inherit Saturn's Full Disk Access — exactly the
  problem we are fixing, moved. So the protocol is a **fixed set of named operations with
  typed arguments** (`messages.read(contact, query, limit)`, never raw SQL or script text), and
  the helper never grows a generic one. Peer-uid checking (`LOCAL_PEERCRED`) keeps other users
  out; it cannot keep the same user's other processes out, and the spec should say so plainly
  rather than pretend a token file does.
- **Trust stack.** Unchanged in the CLI: the gate runs before the RPC, egress for
  `send_message` is still `check()` / `record()` in `tools/messages.py` around the call (that
  file stays the chokepoint `tests/test_no_new_egress.py` pins; the socket is a local
  `AF_UNIX` connection, not a network client — confirm the test's import scan agrees). Defence
  in depth: the helper refuses `messages.send` while `config` says air-gap, since it can read
  the same config file.
- **Costs.** A second long-lived process; an IPC protocol to version (the CLI and the helper
  can be different versions after an update — the handshake carries a protocol number and the
  CLI says "update Saturn.app" on a mismatch); start-up when the helper is not running (the CLI
  `open -ga`s it and waits for the socket, bounded).

### Option B — the whole agent runs as Saturn.app (no IPC)

The terminal command is a shim that gets the real agent started *as* Saturn.app, with the
terminal's tty. Two ways:

- **B1 — disclaim.** The shim spawns the agent with `responsibility_spawnattrs_setdisclaim`, a
  private libsystem call (Chromium, LLDB and iTerm2 use it) that makes the child its own
  responsible process. The child must be an executable inside the signed bundle (a frozen
  Python), so the identity is Saturn.app's. _(probe — P6: callable from Python via ctypes?
  does TCC then show Saturn?)_
- **B2 — fd passing.** The shim connects to the running Saturn.app, hands over its
  stdin/stdout/stderr with `SCM_RIGHTS`, and Saturn.app spawns the agent on those fds. Works
  with public API only, but the agent is not in the terminal's foreground process group:
  Ctrl-C, Esc-to-pause, `SIGWINCH` (resize — the rich Live TUI depends on it), job control
  and the exit code must all be forwarded by the shim by hand.

- **Meets goals 1–3 with no protocol and no tool rewrite** — every tool keeps calling osascript
  in-process, now charged to Saturn.
- **Misses goal 4.** The agent *is* the privileged process, so `run_shell` children inherit
  Saturn's Full Disk Access and Automation grants. A gated, traced shell command could read
  `chat.db` or drive Mail directly. That is better than Terminal holding them (only Saturn's
  own gated commands get them, not every script the user runs), but much weaker than A, where
  no shell ever holds them.
- **Risk.** B1 rests on a private call Apple can change in any release; B2 on hand-forwarding
  terminal semantics the TUI is sensitive to.

### Option C — onboarding only (Phase 0, and stop)

Keep the terminal as the permission holder; ship Phase 0 and the named-app errors.

- Meets goal 3 and fixes the mid-task surprise, which is most of the day-to-day pain.
- Misses 1, 2 and 4: the dialogs still name VS Code, a second terminal still starts over, and
  the grant is still broad.
- Zero signing, zero distribution work. A reasonable stopping point before there are users
  outside this Mac.

## How the app would be built (applies to A; B needs a frozen Python regardless)

| | **A-swift** — Swift helper | **A-stub** — Swift stub + Python child | **A-frozen** — whole helper in Python (py2app / PyInstaller / Briefcase) |
|---|---|---|---|
| What is in the bundle | the helper, all native | a ~200-line Swift launcher; it spawns `python -m saturn.helper` from the user's install as a **child**, so the child is charged to the bundle (P1) | an embedded interpreter, pyobjc, the helper modules |
| Who implements the ops | Swift (EventKit, Contacts.framework, `UNUserNotificationCenter`, SQLite) — the AppleScript in `tools/*.py` is rewritten or carried as resources | **the existing Python** — the op handlers are today's tool bodies, unchanged | the existing Python |
| An update to Saturn's Python | the helper does not change unless an op changes | **the stub does not change; grants stay**, pip updates the code freely | a new bundle — fine with Developer ID, loses grants if ad-hoc |
| Same code in and out of the helper | no — two implementations | **yes** — `helper` absent (dev clone, Linux CI) → the same handler runs in-process, as today | yes |
| Signing difficulty | simple | simple — one small binary | hard — every embedded `.so` signed, hardened runtime, likely `disable-library-validation` |
| Size | ~1 MB | ~1 MB | 30–60 MB |
| Weak point | a second language and a rewrite of the AppleScript layer | the stub lends its grants to whatever Python it spawns: the interpreter path is pinned at `saturn setup` and the stub refuses any other — anyone who can rewrite the venv inherits Saturn's grants (true of Terminal today, and no worse) | the bundle tooling is the most fragile part of the whole design |

A-stub keeps one implementation of every tool and makes the "no helper" fallback free (the
handler simply runs in-process), which is what keeps the test suite and dev clones working
unchanged. A-swift is the cleanest *product* and the fastest at runtime, at the price of a
rewrite. A-frozen has the downsides of both.

## Distribution and signing

- **Developer ID ($99/yr) + notarization** — needed for anyone but the author: Gatekeeper
  otherwise blocks the app on first open, and grants that survive updates need a stable
  signature.
- **Ad-hoc locally** — enough to build and probe everything on this Mac; every rebuild of the
  signed part resets the grants (a strong reason the signed part should be the small, rarely
  changing stub of A-stub or A-swift).
- **Getting the app onto a Mac.** `pipx install saturn` cannot place a `.app`. Options:
  a Homebrew cask (`brew install --cask saturn`, which can also install the CLI); a notarized
  zip on the GitHub release that `saturn setup` points at; or `saturn setup` downloading it —
  which is egress, so it would go through the ledger like any other fetch. Undecided.
- **Install location** `~/Applications/Saturn.app` (no admin password) or `/Applications`.
- **Uninstall** joins `quit_all()`: stop the helper, remove the LaunchAgent, and say which
  Settings rows to switch off (an app cannot revoke its own TCC grants; `tccutil reset All
  <bundle id>` can, and setup can offer it).

## Phases

0. **Onboarding** (above). No decision needed. Ship it.
1. **Probes** (below), one afternoon, ad-hoc signed. Each answers a decision.
2. **The app**, per the chosen option. For A: the menu bar app becomes `Saturn.app`; the op
   protocol and the client seam (`helper.call(op, args)`, replaced in `tests/conftest.py`
   like the Spotlight seam); move the tools over one module at a time, `read_messages` first
   (the worst current experience), then Calendar/Reminders/Contacts, Mail, Notes, Messages
   send, desktop, notifications. Each move keeps its tool's tests green against the seam.
3. **Distribution**: signing, notarization, the install channel, setup's "install Saturn.app"
   row.

## What to probe first

| # | Question | How | Decides |
|---|---|---|---|
| P1 | Does a Python child spawned by a signed stub app prompt as the stub (Automation, FDA)? | Ad-hoc-signed 30-line Swift stub launched by `open -a`, spawning `python -c` that runs `osascript` and opens `chat.db` | whether A-stub works at all |
| P2 | Do ad-hoc grants really reset on rebuild? Does a Developer ID grant survive a version bump? | rebuild the stub; check System Settings | how much signing matters during development |
| P3 | Under hardened runtime, is the apple-events entitlement + usage string sufficient for osascript *children*? | stub with and without the entitlement | the entitlement set |
| P4 | Can `UNUserNotificationCenter` post from an ad-hoc bundle? | stub posts one notification | whether notifications move in Phase 2 or wait for Developer ID |
| P5 | EventKit from the bundle: does it prompt as Saturn, and how fast is a two-week, eight-calendar read? | stub child via pyobjc EventKit | whether Calendar/Reminders/Contacts switch off AppleScript |
| P6 | Is `responsibility_spawnattrs_setdisclaim` reachable via ctypes, and does TCC then show Saturn? | ctypes against libsystem | whether Option B1 is real |
| P7 | Do the `Privacy_*` Settings anchors still open the right pane on macOS 27? `AEDeterminePermissionToAutomateTarget` from pyobjc without a dialog? | `open` each URL; call it for Notes | Phase 0 details |

## Decisions to make

1. **Architecture**: A (helper + IPC), B (agent runs as the app — B1 or B2), or C (onboarding
   only, for now). _Leaning: A — it is the only option that meets goal 4, which is the trust
   pitch; C is a fine stop until there are outside users._
2. **How A is built**: A-swift, A-stub or A-frozen. _Leaning: A-stub if P1 holds (one
   implementation, free fallback, grants that survive pip updates); A-swift if it doesn't._
3. **EventKit**: keep AppleScript for Calendar/Reminders/Contacts, or switch once the bundle
   allows it (P5).
4. **Notifications**: keep one LaunchAgent per notification + osascript, or let the helper
   schedule them natively (`UNCalendarNotificationTrigger`) and drop the per-notification
   agents.
5. **Signing**: buy a Developer ID now, or stay ad-hoc until outside users.
6. **Install channel**: Homebrew cask, release zip, or setup downloads it.
7. **Setup's name**: `saturn setup` + `/setup`, or a `/policy` row (the trust front door —
   these are OS-level permissions, not Saturn's gate, which argues for a separate command).

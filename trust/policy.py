"""
The gate policy — ONE object behind every approval decision.

What used to be five separate gate-relaxation mechanisms are now views over this module
(the v1.x "policy-as-configuration" consolidation):

  runtime.auto_approve   the policy's tier threshold — tools AT OR BELOW it run without
                         prompting. The baseline lives in config.yaml like every other knob;
                         `tier()`/`set_tier()` are the read/write path (Shift+Tab cycles it).
  /policy open           a view that sets the threshold to `destructive` (gate fully open)
                         and restores the previous threshold on `off` — not a separate switch.
  --yolo (headless)      the same view, applied at process start.
  /policy risk           edits a TOOL's tier (live in registry.TOOL_RISK; persisted here).
  /policy allow          edits the run_shell prefix allowlist (persisted here).

(The legacy top-level command spellings /risk, /allow, /autoapprove were CUT 2026-07-06 —
_RENAMED pointers cover the muscle memory; the mechanisms themselves are unchanged.)

Durable state is one small, versionable JSON file at `config.path("permissions")`
(database/permissions.json): `risk_overrides` ({tool: tier}, applied over declared tiers at
startup by registry.py) + `shell_allow` ([prefix, ...]). The tier threshold persists via
config.yaml (`/config --save runtime.auto_approve`), not here.

The one gate question is `approves(name, risk, args)` — the approval node asks it for every
tool call, and nothing else decides whether a call skips the human.

Prefix matching is deliberately strict: it is TOKEN-based ("git status" matches "git status
--short" but not "git statusx"), case-insensitive, and refuses to exempt any command containing
shell metacharacters (;, |, &, redirection, substitution, newlines). Without that refusal,
allowing "git status" would also wave through "git status; rm -rf ~" — the gate must fail closed
on anything it can't read at a glance. A background run_shell call (detached, timeout-free) is
never prefix-exempt either: the prefix was granted for a bounded foreground run, not a daemon.

Imports only config + diag (both leaves), so registry.py, the approval node, and the TUI can
import this freely.
"""

from __future__ import annotations

import json
import os
import re

import diag
from config import get_config, persist, RISK_ORDER

# Any of these in a command means it can do more than its first tokens say — chaining, piping,
# redirection, substitution. Such a command is never prefix-exempt; the human reads it at the gate.
_SHELL_META = re.compile(r"[;&|<>`$\n\r]")

# --- the argument-tail screen (transplanted from the gating isolate, 2026-08-15) ------------
#
# A token-prefix grant validated only its HEAD, so `git log --output=<abs>` and
# `git -c core.pager=!sh -c id` rode in on a benign-looking grant. These tables close the NAMED
# laundering paths in the tokens AFTER a granted prefix; the screen re-runs at USE against the
# live command text (never validate-once) and can only ever tighten. It is a denylist and stays
# defense-in-depth: the boundary is the human approving the exact reviewed command.

# Flags that introduce a new exec or write path on otherwise-benign programs.
_CAPABILITY_FLAGS = {
    "-c", "--config", "-e", "--eval", "--exec", "--command",
    "-o", "--output", "--output-file", "--out",
    "--upload-pack", "--receive-pack", "--ext", "--to-command",
    "--pager", "--editor", "--use-compress-program", "-i", "--interactive",
}

# General-purpose interpreters / launchers: a grant on one is honored only as an EXACT command
# ("always allow `npm test`" stays useful; "always allow `npm`" never becomes arbitrary code).
_INTERPRETERS = {
    "sh", "bash", "zsh", "ksh", "dash", "csh", "tcsh", "fish",
    "python", "python2", "python3", "py", "perl", "ruby", "lua", "tclsh", "r", "rscript",
    "node", "deno", "bun", "php", "awk", "gawk", "mawk", "nawk", "sed", "ed",
    "env", "xargs", "find", "make", "cmake", "ninja",
    "npm", "npx", "yarn", "pnpm", "pip", "pip3", "uv", "uvx", "poetry", "pipx",
    "docker", "podman", "kubectl", "ssh", "scp", "rsync", "sudo", "doas", "su",
    "powershell", "pwsh", "cmd", "iex", "wscript", "cscript", "rundll32", "regsvr32",
    "msbuild", "dotnet", "java", "go", "cargo", "gradle", "mvn", "ansible", "terraform",
}

# A glob in the tail: its expansion is not what was screened.
_GLOB = re.compile(r"[*?\[\]]")


# --- the tier threshold (runtime.auto_approve and its views) -----------------------------


def tier() -> str:
    """The effective auto-approve threshold: tools at or below it never prompt."""
    return get_config().auto_approve


def set_tier(new_tier: str, save: bool = False) -> str:
    """Set the threshold (session-scoped, like every cfg.set; `save` persists to config.yaml).
    Unknown tiers fail closed to read_only. Returns the tier actually set."""
    if new_tier not in RISK_ORDER:
        new_tier = "read_only"
    get_config().set("runtime.auto_approve", new_tier)
    if save:
        persist("runtime.auto_approve")
    return new_tier


def auto_approves(risk: str) -> bool:
    """True if a tool of the given risk tier runs without prompting under the current policy."""
    return get_config().auto_approves(risk)


# /policy open is not a sixth mechanism — it's this: threshold = destructive (every tier passes).
# Remember what the threshold was so `off` restores it instead of guessing.
_tier_before_gate_off: "str | None" = None


def gate_off() -> bool:
    """Whether the gate is fully open (threshold at `destructive` — nothing prompts)."""
    return tier() == "destructive"


def set_gate_off(off: bool) -> None:
    """The /policy open · --yolo view: open the gate by raising the threshold to `destructive`;
    close it by restoring the prior threshold (read_only if none was recorded — fail closed)."""
    global _tier_before_gate_off
    if off:
        if not gate_off():
            _tier_before_gate_off = tier()
        set_tier("destructive")
    else:
        set_tier(_tier_before_gate_off or "read_only")
        _tier_before_gate_off = None


# --- the one gate question ----------------------------------------------------------------


def approves(name: str, risk: str, args: "dict | None" = None) -> bool:
    """Whether a tool call runs WITHOUT facing the human. The approval node asks this for every
    pending call; the only two ways through are the tier threshold and (for run_shell only) a
    user-persisted /policy allow prefix on the exact command."""
    if auto_approves(risk):
        return True
    if name == "run_shell":
        command = str((args or {}).get("command", ""))
        return shell_allowed(command) is not None
    return False


# --- durable storage (database/permissions.json) -------------------------------------------


def _path():
    return get_config().path("permissions")


# Set on the first corrupt-policy-file load this process (None = the file loaded cleanly or
# simply doesn't exist yet). The gate itself must never raise — it degrades to safe defaults —
# but a silent reset drops user-RAISED /policy risk overrides below the posture the operator believes
# is in force, so the degradation must be LOUD somewhere: agent.main reads this once at startup
# and warns (the mcp_client.problems() pattern; trust/ never imports tui, so the user-visible
# warning lives at the surface, not here).
_LOAD_PROBLEM: "str | None" = None


def load_problem() -> "str | None":
    """The corrupt-policy-file report, if the durable policy failed to load this session."""
    return _LOAD_PROBLEM


def _load() -> dict:
    """The stored policy file, with safe defaults when missing or unreadable. A MISSING file is
    the normal first run (silent); a GARBLED one (ValueError: bad JSON, wrong shape, undecodable
    bytes) is a posture event — recorded once (`load_problem()` + diag.log) and the bad bytes
    renamed to permissions.json.corrupt, because the next `_save` would otherwise overwrite the
    user's only copy of the prior posture with defaults-plus-one-entry. A TRANSIENT read failure
    (OSError: an AV/backup tool briefly holding the file, a permission hiccup) degrades to
    defaults for the read but leaves the file IN PLACE — the content isn't corrupt, and renaming
    a perfectly valid policy file away on a momentary lock would silently drop the persisted
    posture for every future session."""
    global _LOAD_PROBLEM
    path = _path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON mapping, got {type(data).__name__}")
        # Field SHAPES get the same treatment as a garbled file (transplanted from the gating
        # isolate): a wrong-typed field is never iterated as-is — a string shell_allow would
        # turn each character into an allowlist prefix, a list risk_overrides would raise in
        # the registry — so it fails closed to defaults through the same recorded/renamed path.
        overrides = data.get("risk_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("risk_overrides is not a mapping")
        allow = data.get("shell_allow", [])
        if not isinstance(allow, list) or not all(isinstance(p, str) for p in allow):
            raise ValueError("shell_allow is not a list of prefix strings")
    except FileNotFoundError:
        data = {}
    except OSError as exc:
        data = {}
        if _LOAD_PROBLEM is None:
            _LOAD_PROBLEM = (
                f"gate policy file could not be read ({path}): {exc} — running with defaults; "
                "persisted risk overrides and shell-allow prefixes are NOT in effect (file "
                "left in place; restart to retry)"
            )
            diag.log(f"policy: {_LOAD_PROBLEM}")
    except ValueError as exc:
        data = {}
        if _LOAD_PROBLEM is None:
            saved = ""
            try:  # keep the prior posture recoverable before any mutation rewrites the file
                os.replace(path, path.with_name(path.name + ".corrupt"))
                saved = f" (bad file kept as {path.name}.corrupt)"
            except OSError as move_exc:
                diag.log(f"policy: could not preserve corrupt policy file: {move_exc}")
            _LOAD_PROBLEM = (
                f"gate policy file unreadable ({path}): {exc} — running with defaults; "
                f"persisted risk overrides and shell-allow prefixes are NOT in effect{saved}"
            )
            diag.log(f"policy: {_LOAD_PROBLEM}")
    data.setdefault("risk_overrides", {})
    data.setdefault("shell_allow", [])
    return data


def _save(data: dict) -> None:
    """Crash-safe write: sibling temp file then os.replace (atomic on Windows + POSIX). A kill
    mid-write truncates the temp file, never the live gate policy — a truncated permissions.json
    would silently drop user-RAISED risk overrides on the next load. Local copy of the idiom
    (cf. stores/memory_registry._atomic_write): policy.py is a leaf and must not import stores."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --- risk-tier overrides (/policy risk … --save) ----------------------------------------


def risk_overrides() -> dict:
    """{tool_name: tier} as persisted. Validation against the live registry happens at the
    call site (registry.py) — a stale name for a removed tool is simply ignored there."""
    return dict(_load()["risk_overrides"])


def set_risk_override(tool: str, tier: str) -> None:
    data = _load()
    data["risk_overrides"][tool] = tier
    _save(data)


def clear_risk_override(tool: str) -> bool:
    """Remove a persisted override; True if one was stored."""
    data = _load()
    if tool not in data["risk_overrides"]:
        return False
    del data["risk_overrides"][tool]
    _save(data)
    return True


# --- run_shell prefix allowlist (/policy allow) -----------------------------------------


def shell_allow() -> list[str]:
    return list(_load()["shell_allow"])


def add_shell_allow(prefix: str) -> bool:
    """Store a prefix; False if it (case-insensitively) is already stored. Raises ValueError on
    text that could never be a gate-exempt prefix (`shell_prefix_rejects`: empty, or carrying a
    shell metacharacter) — storing it anyway would create a permanently-inert grant that the
    confirmation copy then claims skips the gate, a posture the matcher (`shell_allowed`)
    contradicts. The screen runs on the RAW input BEFORE whitespace normalization: normalization
    collapses newlines into spaces, which would launder one metacharacter class straight past the
    screen."""
    reason = shell_prefix_rejects(prefix)
    if reason:
        raise ValueError(reason)
    prefix = " ".join(prefix.split())
    data = _load()
    if any(p.lower() == prefix.lower() for p in data["shell_allow"]):
        return False
    data["shell_allow"].append(prefix)
    _save(data)
    return True


def remove_shell_allow(token: str) -> "str | None":
    """Remove a prefix by 1-based index or exact text; returns what was removed, or None."""
    data = _load()
    allow = data["shell_allow"]
    removed = None
    if token.isdigit() and 1 <= int(token) <= len(allow):
        removed = allow.pop(int(token) - 1)
    else:
        for i, p in enumerate(allow):
            if p.lower() == token.strip().lower():
                removed = allow.pop(i)
                break
    if removed is not None:
        _save(data)
    return removed


def shell_prefix_rejects(text: str) -> "str | None":
    """Why `text` could never be a gate-exempt shell prefix (None when it could be): empty, or
    carrying a shell metacharacter — chaining/piping/redirection/substitution means the leading
    tokens don't bound what a command does, so the matcher refuses such text wholesale. THE public
    face of the metacharacter screen: callers (the gate UI's always-allow flow) ask this instead
    of re-reading the private regex."""
    text = str(text)
    if not text.strip():
        return "empty prefix"
    if _SHELL_META.search(text):
        return "contains a shell metacharacter (; & | < > ` $ or a newline)"
    if not text.isascii():
        # A confusable (fullwidth ';' U+FF1B and friends) reads as inert to an ASCII screen and
        # as punctuation to a human reviewing the grant. The automation path stays ASCII;
        # anything else goes to a person.
        return "contains non-ASCII text (confusable with a shell metacharacter)"
    return None


def _escapes_workspace(token: str) -> bool:
    """Whether an argument names a path that could leave the workspace: absolute (POSIX, UNC,
    or drive-lettered), parent-relative, or home-relative. Deliberately syntactic."""
    t = token.strip("'\"")
    if not t:
        return False
    if t.startswith("~") or t.startswith("/") or t.startswith("\\"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", t):  # C:\ or C:/
        return True
    return ".." in re.split(r"[\\/]", t)


def arg_tail_rejects(prefix: str, command: str) -> "str | None":
    """Why the tokens AFTER a granted prefix disqualify `command` (None when they don't): the
    program is a general-purpose interpreter with trailing arguments, or a tail token is a
    capability-introducing flag, a glob, or a path outside the workspace. Pure and
    deterministic; a reason is a sentence the gate UI can print verbatim."""
    p_tokens = str(prefix).split()
    c_tokens = str(command).split()
    tail = c_tokens[len(p_tokens):]

    program = (c_tokens[0] if c_tokens else "").lower()
    program = program.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if program.endswith(".exe"):
        program = program[:-4]
    if program in _INTERPRETERS and tail:
        return (f"`{program}` is a general-purpose interpreter — a grant covers it only as an "
                "exact command, not with trailing arguments")

    for tok in tail:
        flag = tok.lower().split("=", 1)[0]
        if flag in _CAPABILITY_FLAGS:
            return f"argument `{tok}` can introduce a new exec or write path"
        if _GLOB.search(tok):
            return f"argument `{tok}` contains a glob — its expansion is not what was screened"
        if _escapes_workspace(tok):
            return f"argument `{tok}` names a path outside the workspace"
    return None


def shell_prefix_covers(prefix: str, command: str) -> bool:
    """Whether `prefix` would exempt `command` under THE matcher's rules — the per-prefix half of
    `shell_allowed` (metacharacter screen on the command, token-boundary equality, then the
    argument-tail screen), exposed pure so the gate UI can validate a typed grant WITHOUT
    persisting it (the always-allow flow collects grants at decision time and applies them past
    the interrupt). `shell_allowed` delegates here: one matcher, never a second copy of its rule."""
    if shell_prefix_rejects(command):
        return False
    cmd_tokens = [t.lower() for t in str(command).split()]
    p_tokens = [t.lower() for t in str(prefix).split()]
    if not p_tokens or cmd_tokens[: len(p_tokens)] != p_tokens:
        return False
    return arg_tail_rejects(prefix, command) is None


def shell_allowed(command: str) -> "str | None":
    """The allowlisted prefix that exempts `command` from the gate, or None.

    A command is exempt only when (a) the metacharacter screen (`shell_prefix_rejects`) passes —
    no chaining/redirection/substitution anywhere in it — and (b) its leading whitespace-split
    tokens equal some stored prefix's tokens, case-insensitively (`shell_prefix_covers`). Token
    equality (not startswith) so "git status" never matches "git statusx"."""
    if shell_prefix_rejects(command):
        return None
    for prefix in _load()["shell_allow"]:
        if shell_prefix_covers(prefix, command):
            return prefix
    return None


def grant_shell_prefix(prefix: str, command: str, *, dry_run: bool = False) -> "tuple[bool, str]":
    """The gate's scoped always-allow grant: validate `prefix` against `command` through the one
    matcher and (unless `dry_run`) persist it to the /policy allow store. Returns (command now exempt?,
    disclosure message — the gate UI prints it verbatim).

    The UI calls this with dry_run=True at decision time, while the approval interrupt is still
    pending: persisting then would let the node's re-run recompute the batch as ungated and lose
    the human's decision from gate_events (gotcha #7) — so the UI only collects the validated
    grant, and the approval node applies it here past the interrupt. Never raises: every refusal
    is a (False, why) so a typed metacharacter degrades to "it keeps prompting", never a dead
    turn."""
    prefix = " ".join(str(prefix).split())
    if not prefix:
        return False, "run_shell: no prefix granted — it keeps prompting"
    if shell_prefix_rejects(prefix) or not shell_prefix_covers(prefix, command):
        why = (shell_prefix_rejects(prefix) or shell_prefix_rejects(command)
               or arg_tail_rejects(prefix, command)
               or "token boundary, no shell metacharacters")
        return False, (f'run_shell: prefix "{prefix}" would not exempt this command '
                       f"({why}) — no grant, it keeps prompting")
    matched = shell_allowed(command)
    if matched is not None and matched.lower() != prefix.lower():
        # An already-stored prefix covers this command; the new one adds nothing for it — no
        # redundant entry to stack up for the user to audit later.
        return True, f'run_shell: already covered by allowlisted prefix "{matched}"'
    if not dry_run:
        add_shell_allow(prefix)  # screened above — cannot raise
    return True, (f'run_shell: always-allowing commands starting "{prefix}" '
                  "(persisted to the allowlist; undo: /policy allow remove)")

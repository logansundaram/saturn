from commands._framework import command, _print


@command(
    "allow",
    "Allowlist shell command prefixes that skip the approval gate (persisted).",
    usage="/allow [<prefix words…> | remove <n|prefix>]",
    details="""
run_shell is `destructive`, so every command normally faces the approval gate. /allow stores
command PREFIXES that you trust — a run_shell call whose command starts with one runs without
prompting, so the gate stops training you to mash `y` on `git status` while still guarding
everything else.

  /allow                       list the stored prefixes
  /allow git status            allow `git status`, `git status --short`, …
  /allow remove 2              remove a prefix by its list number
  /allow remove git status     …or by its exact text

Matching is strict on purpose:
  - token-boundary: `git status` does NOT match `git statusx`
  - case-insensitive
  - a command containing ; | & < > ` $ or a newline is NEVER exempt, even if its start
    matches — chaining/redirection can smuggle anything behind a trusted prefix, so those
    always face the human.
  - a background run (run_shell with background=true — detached, no timeout) is NEVER exempt
    either; the prefix covers bounded foreground runs only.

Persisted to the policy file database/permissions.json (alongside /risk --save overrides), so it
survives restarts. /allow, /risk and /autoapprove are three views of one gate policy (policy.py).
Allow narrow, read-only prefixes (`git status`, `git log`, `ls`) — not broad ones
(`git`, `python`).
""",
)
def _allow(ctx, args):
    import policy

    if not args:
        prefixes = policy.shell_allow()
        if not prefixes:
            _print("  no allowlisted shell prefixes — add one with /allow <prefix words…>")
            _print("  e.g. /allow git status")
            return
        _print("  run_shell commands starting with these run WITHOUT the approval gate:")
        for i, p in enumerate(prefixes, 1):
            _print(f"    {i}. {p}")
        _print("  remove: /allow remove <n|prefix>")
        return

    if args[0].lower() == "remove":
        if len(args) < 2:
            _print("  usage: /allow remove <n|prefix>")
            return
        removed = policy.remove_shell_allow(" ".join(args[1:]))
        if removed is None:
            _print("  no such prefix — /allow lists them with their numbers.")
        else:
            _print(f"  removed: {removed} (commands like this face the gate again).")
        return

    prefix = " ".join(args)
    if policy.add_shell_allow(prefix):
        _print(f"  allowed: run_shell commands starting with `{prefix}` now skip the gate.")
        _print("  (persisted; undo with /allow remove. Chained/redirected commands still prompt.)")
    else:
        _print(f"  `{prefix}` is already allowlisted.")

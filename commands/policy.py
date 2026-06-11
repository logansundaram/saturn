"""
/policy — the gate policy as one shareable, versionable document.

policy.py consolidated the five gate-relaxation mechanisms into one object; this command makes
that object a FILE you can read, share, and apply: `/policy` shows the live posture, `export`
writes it as a commented YAML profile, `import` applies one (replacing the durable
permissions.json and setting the airgap/redaction knobs). The same profile drives headless runs
via `saturn --policy <file>` — versionable safety postures for people, projects, and CI.
"""

from __future__ import annotations

from pathlib import Path

from commands._framework import command, _print

_PROFILE_HEADER = """\
# Saturn policy profile — the whole gate posture as one shareable document.
# Apply with `/policy import <this file>` or `saturn --policy <this file>`.
"""


def apply_policy_file(path_str: str, save: bool = False) -> str:
    """Load a profile YAML and apply it: policy.apply_profile (threshold + permissions.json +
    airgap/redaction) then sync the LIVE tool registry so the overrides bite immediately.
    Returns a one-line summary; raises on an unreadable/invalid profile (the caller reports)."""
    import yaml

    import policy

    path = Path(path_str).expanduser()
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = policy.apply_profile(profile, save=save)

    # Sync the live registry: back to declared tiers, then the profile's overrides — an import
    # REPLACES the posture (mirrors apply_profile replacing permissions.json), never layers on
    # whatever /risk edits the session had. Stale names (tools this install doesn't have) are
    # reported, not applied — the declared fail-closed tier stays in effect.
    import registry

    for name, tier in registry.DECLARED_RISK.items():
        registry.TOOL_RISK[name] = tier
    stale = []
    applied = 0
    for name, tier in overrides.items():
        if name in registry.tools_by_name:
            registry.TOOL_RISK[name] = tier
            applied += 1
        else:
            stale.append(name)

    # A profile can flip runtime.airgap — drop the model cache so a cloud model built while the
    # boundary was open can't keep serving calls from llms._DERIVED_CACHE (mirrors the
    # /privacy airgap toggle, which does exactly this for exactly this reason).
    import llms

    llms.reset_models()

    summary = (
        f"policy applied from {path.name}: threshold={policy.tier()}, "
        f"{applied} risk override(s), {len(policy.shell_allow())} shell prefix(es)"
    )
    if stale:
        summary += f"  (ignored overrides for unknown tools: {', '.join(stale)})"
    return summary


@command(
    "policy",
    "The gate policy as one object: show the posture, export/import shareable profiles.",
    usage="/policy | /policy export [path] | /policy import <path> [--save]",
    details="""
Every gate-relaxation mechanism (/risk, /allow, /autoapprove, runtime.auto_approve, --yolo) is a
view of one policy object. This command shows that object — and turns it into a file:

  /policy                     the live posture: auto-approve threshold, persisted risk overrides,
                              the shell allowlist, airgap + redaction modes.
  /policy export [path]       write the posture as a YAML profile (default:
                              logging/exports/policy.yaml). Shareable + versionable: commit it,
                              hand it to a teammate, keep a `paranoid.yaml` next to a `ci.yaml`.
  /policy import <path>       apply a profile. REPLACES the durable policy (permissions.json:
                              risk overrides + shell allowlist) and sets the threshold and the
                              airgap/redaction knobs for the session; --save also persists those
                              knobs to config.yaml. Applies to the live registry immediately.

Headless: `saturn -p "query" --policy <file>` applies a profile at process start — pin the exact
safety posture a script or CI job runs under instead of choosing between deny-all and --yolo.
""",
)
def _policy_cmd(ctx, args):
    import policy
    from config import get_config

    if not args:
        cfg = get_config()
        overrides = policy.risk_overrides()
        allow = policy.shell_allow()
        _print("  gate policy (one object — /risk, /allow, /autoapprove are views of it):")
        threshold = policy.tier()
        label = "⚠ GATE OFF (everything auto-approved)" if threshold == "destructive" else threshold
        _print(f"    auto-approve threshold : {label}")
        if overrides:
            _print(f"    risk overrides         : " + ", ".join(
                f"{k}→{v}" for k, v in sorted(overrides.items())))
        else:
            _print("    risk overrides         : (none)")
        if allow:
            _print(f"    shell allowlist        : " + " · ".join(allow))
        else:
            _print("    shell allowlist        : (none)")
        _print(f"    airgap                 : {'on' if cfg.get('runtime.airgap', False) else 'off'}")
        _print(f"    redaction              : {cfg.get('runtime.redaction', 'off') or 'off'}")
        _print("  export it: /policy export [path]   ·   apply one: /policy import <path>")
        return

    sub = args[0].lower()

    if sub == "export":
        import yaml

        if len(args) > 1:
            dest = Path(" ".join(args[1:]).strip('"')).expanduser()
        else:
            dest = get_config().path("exports") / "policy.yaml"
        profile = policy.export_profile()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                _PROFILE_HEADER + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except OSError as exc:
            _print(f"  could not write {dest}: {exc}")
            return
        _print(f"  policy profile exported -> {dest}")
        _print("  apply it anywhere: /policy import <file>  ·  saturn --policy <file>")
        return

    if sub == "import":
        rest = [a for a in args[1:] if a.lower() not in ("--save", "-s")]
        save = len(rest) != len(args[1:])
        if not rest:
            _print("  usage: /policy import <file> [--save]")
            return
        path = " ".join(rest).strip('"')
        try:
            summary = apply_policy_file(path, save=save)
        except FileNotFoundError:
            _print(f"  no such file: {path}")
            return
        except Exception as exc:
            _print(f"  could not apply {path}: {exc}")
            return
        _print(f"  {summary}")
        if not save:
            _print("  (threshold + airgap/redaction set for this session; --save persists them)")
        return

    _print(f"  unknown /policy subcommand: {sub!r} — try: export, import (or /policy --help)")

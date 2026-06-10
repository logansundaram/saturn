import subprocess
import sys
from pathlib import Path

from commands._framework import command, _print

# Saturday ships as a git clone (install.sh / install.ps1), so the repo root IS the install.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, timeout: float = 60):
    """Run a git command at the repo root; (returncode, stdout, stderr), never raises for the
    command failing (only for git itself being absent, handled by the caller)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


@command(
    "update",
    "Update Saturday to the latest version (git pull at the install root).",
    usage="/update [--check]",
    details="""
Pulls the latest Saturday from the git remote it was installed from (the install scripts clone
the repo, so the repo root IS the install).

  /update          fast-forward pull; lists what came in; reinstalls Python dependencies if
                   requirements.txt changed in the pull
  /update --check  fetch only and report how many commits behind you are, without changing
                   anything

Fast-forward only (`git pull --ff-only`): if you have local commits or edits that diverge from
the remote, the pull refuses rather than merging on its own — resolve that in git yourself.
After a successful update, restart (/quit and relaunch) to run the new code; the running
process keeps its already-imported version.
""",
)
def _update(ctx, args):
    try:
        rc, _, _ = _git("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        _print("  git is not installed (or not on PATH) — cannot self-update.")
        return
    except subprocess.TimeoutExpired:
        _print("  git did not respond — cannot self-update.")
        return
    if rc != 0:
        _print(f"  {_REPO_ROOT} is not a git repository — was Saturday installed by hand?")
        _print("  installed via pipx/uv? update with `pipx upgrade saturn-agent` "
               "(or `uv tool upgrade saturn-agent`).")
        _print("  otherwise re-install with the install script, or replace the files yourself.")
        return

    try:
        if any(a in ("--check", "-c") for a in args):
            _print("  checking for updates…")
            rc, _, err = _git("fetch", "--quiet", timeout=120)
            if rc != 0:
                _print(f"  fetch failed: {err or 'unknown error'}")
                return
            rc, behind, err = _git("rev-list", "--count", "HEAD..@{u}")
            if rc != 0:
                _print(f"  no upstream configured for this branch: {err}")
                return
            n = int(behind or 0)
            if n == 0:
                _print("  up to date.")
            else:
                _print(f"  {n} commit(s) behind — run /update to pull them.")
            return

        _, old, _ = _git("rev-parse", "HEAD")

        rc, dirty, _ = _git("status", "--porcelain")
        if rc == 0 and dirty:
            _print("  note: you have local uncommitted changes — the pull will refuse if they conflict.")

        _print("  pulling latest…")
        rc, out, err = _git("pull", "--ff-only", timeout=300)
        if rc != 0:
            _print(f"  pull failed: {err or out or 'unknown error'}")
            _print("  (diverged from the remote? resolve it in git, then retry /update.)")
            return

        _, new, _ = _git("rev-parse", "HEAD")
        if new == old:
            _print("  already up to date.")
            return

        rc, log, _ = _git("log", "--oneline", f"{old}..{new}")
        commits = log.splitlines() if rc == 0 else []
        _print(f"  updated {old[:7]} -> {new[:7]} ({len(commits)} commit(s)):")
        for line in commits[:15]:
            _print(f"    {line}")
        if len(commits) > 15:
            _print(f"    … {len(commits) - 15} more")

        # If the pull changed the dependency list, install it — an updated module importing a
        # package that isn't there yet would otherwise greet the next launch with a stack trace.
        rc, changed, _ = _git("diff", "--name-only", old, new)
        if rc == 0 and "requirements.txt" in changed.splitlines():
            _print("  requirements.txt changed — installing dependencies (this can take a minute)…")
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if proc.returncode == 0:
                _print("  dependencies installed.")
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
                _print(f"  pip install failed: {tail[0] if tail else 'unknown error'}")
                _print(f"  run it yourself: {sys.executable} -m pip install -r requirements.txt")

        _print("  restart Saturday (/quit and relaunch) to run the new version.")
    except subprocess.TimeoutExpired:
        _print("  update timed out — check your network and try again.")

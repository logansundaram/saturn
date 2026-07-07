"""
Lightweight diagnostic log.

Node/tool timing lines and soft, non-fatal warnings (a planner structured-output miss, judge
failures) are diagnostics — useful when debugging, noise during normal use.
They used to `print()` straight to stdout, where they collided with the rich.Live status bar and
the styled trace rail in the TUI. They now go here instead: appended to a file under `logging/`
(gitignored) and silent on the console by default. Set the env var `SATURDAY_DEBUG=1` to also echo
them to stderr during development.

No project imports, so this is safe to import from any node/tool/store without circular-import risk.
The `logging/` directory at the repo root does NOT shadow the stdlib `logging` module here: it has
no `__init__.py`, and a regular package (stdlib) always wins over a namespace-package directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

def _resolve_log_dir() -> Path:
    # Clone mode: logging/ at the repo root (config.yaml sits next to this file). Wheel installs
    # (pipx/uv) must not write into site-packages — use SATURDAY_HOME (~/.saturday), mirroring
    # config.py's lookup (kept in step by hand: this module imports nothing project-side).
    root = Path(__file__).parent
    if (root / "config.yaml").exists():
        return root / "logging"
    return Path(os.environ.get("SATURDAY_HOME") or Path.home() / ".saturday") / "logging"


_LOG_DIR = _resolve_log_dir()
_logger: logging.Logger | None = None


def log_dir() -> Path:
    """THE logging directory (clone: repo logging/; wheel: SATURDAY_HOME/logging). Public so
    other log sinks (mcp_client's mcp.log) share one resolution instead of hand-copying it."""
    return _LOG_DIR


def _get() -> logging.Logger:
    """Lazily build the singleton file logger (and an optional stderr echo under SATURDAY_DEBUG)."""
    global _logger
    if _logger is None:
        lg = logging.getLogger("saturday.diag")
        lg.setLevel(logging.DEBUG)
        lg.propagate = False  # don't bubble into the root logger / stdout
        if not lg.handlers:
            try:
                _LOG_DIR.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(_LOG_DIR / "diag.log", encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
                lg.addHandler(fh)
            except Exception:
                # A log sink must never break the app; degrade to no file handler.
                pass
            if os.getenv("SATURDAY_DEBUG"):
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter("%(message)s"))
                lg.addHandler(sh)
        _logger = lg
    return _logger


def log(msg: object) -> None:
    """Record one diagnostic line. Drop-in replacement for the old `print(...)` timing calls."""
    _get().debug(str(msg))

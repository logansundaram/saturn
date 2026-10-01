"""
Lightweight diagnostic log.

Node/tool timing lines and soft, non-fatal warnings (a malformed model output retried, an
empty thinking pass rerun) are diagnostics — useful when debugging, noise during normal use.
`print()` would collide with the rich.Live status bar and the trace rail in the TUI, so they are
appended to a file under `logging/` (gitignored) and silent on the console by default. Set the
env var `SATURN_DEBUG=1` to also echo them to stderr.

No project imports, so this is safe to import from anywhere — which is why the data-home rule
(`saturn_home`, `data_root`) lives here: config.py and env_keys.py read it from this leaf.
The `logging/` directory at the repo root does NOT shadow the stdlib `logging` module here: it has
no `__init__.py`, and a regular package (stdlib) always wins over a namespace-package directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def saturn_home() -> Path:
    """The user's own Saturn folder: `$SATURN_HOME`, else `~/.saturn` — hand-written files that
    follow the user everywhere (the global SATURN.md, hooks.yaml), and a wheel install's data."""
    return Path(os.environ.get("SATURN_HOME") or Path.home() / ".saturn").expanduser()


def data_root() -> Path:
    """Where user data lives: the repo root in clone mode (config.yaml — or, before the first-run
    seed, the template config.default.yaml — sits next to this file), else `saturn_home()`: a
    wheel (pipx/uv) install must not write into site-packages."""
    root = Path(__file__).parent
    if (root / "config.yaml").exists() or (root / "config.default.yaml").exists():
        return root
    return saturn_home()


_LOG_DIR = data_root() / "logging"
_logger: logging.Logger | None = None


def log_dir() -> Path:
    """THE logging directory (clone: repo logging/; wheel: <wheel data home>/logging). Public so
    other log sinks (mcp_client's mcp.log) share one resolution instead of hand-copying it."""
    return _LOG_DIR


def _get() -> logging.Logger:
    """Lazily build the singleton file logger (and an optional stderr echo under SATURN_DEBUG)."""
    global _logger
    if _logger is None:
        lg = logging.getLogger("saturn.diag")
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
            if os.getenv("SATURN_DEBUG"):
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter("%(message)s"))
                lg.addHandler(sh)
        _logger = lg
    return _logger


def log(msg: object) -> None:
    """Record one diagnostic line."""
    _get().debug(str(msg))

"""The saturn command line: the strict argparse surface + piped-stdin capture.

Strict by design — an unknown flag or an invalid flag combination exits 2 (through
parser.error) instead of silently falling through to the interactive TUI.
"""

import sys

from app import __version__
from core import mentions


def _build_parser():
    """The saturn CLI parser — strict (an unknown flag exits 2 instead of silently launching the
    TUI). The flag reference lives here, in --help, not in main()'s docstring."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="saturn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Saturday.ai — local-first, transparent agent.\n"
            "\n"
            "Run with no arguments for the interactive chat loop (/help lists commands,\n"
            "/quit exits). The flags below are the headless/automation surface."
        ),
        epilog=(
            "headless mode (-p):\n"
            "  Read-only tools run freely; gated (side-effecting/destructive) tool calls are\n"
            "  DENIED by default — there is no human at the approval gate, and safe-by-default\n"
            "  must hold in every mode. Pass --yolo to auto-approve them. Piped stdin attaches\n"
            "  to the turn:\n"
            "    git diff | saturn -p \"review this change\"\n"
        ),
    )
    parser.add_argument("-p", "--prompt", metavar="QUERY", default=None,
                               help="Run a single query headlessly and print the answer to "
                                    "stdout (no TUI, no interactive prompts).")
    parser.add_argument("--yolo", action="store_true",
                               help="Open the approval gate for the whole run — auto-approve "
                                    "side-effecting/destructive tool calls, headless or "
                                    "interactive. The same view of the gate policy as "
                                    "/autoapprove (policy.set_gate_off — threshold: destructive).")
    parser.add_argument("--json", action="store_true",
                               help="With -p: print a structured JSON result (answer, plan, "
                                    "tools, tokens, timing) instead of the bare answer. Errors "
                                    "also emit JSON (status: \"error\") and still exit 1.")
    parser.add_argument("--export", metavar="FILE", default=None,
                               help="With -p: after the turn completes, write the run's complete "
                                    "export record to FILE "
                                    "(the same artifact /trace export writes).")
    parser.add_argument("--replay", metavar="FILE", default=None,
                               help="Replay an exported run record (/trace export) offline — "
                                    "no database needed — then exit.")
    parser.add_argument("--version", action="version", version=f"saturn {__version__}")
    return parser


def _parse_cli(argv=None):
    """Parse + validate the CLI line. Strict by design: argparse exits 2 on an unknown flag, and
    the cross-flag rules below exit 2 through parser.error — a typo'd invocation must never
    silently fall through to the interactive TUI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    # -p "" (present but blank) is an invocation mistake, distinct from -p absent (interactive).
    if args.prompt is not None and not args.prompt.strip():
        parser.error("empty prompt")
    if args.prompt is not None and args.replay:
        parser.error("--replay renders an export offline and cannot be combined with -p/--prompt")
    if args.export and args.prompt is None:
        parser.error("--export only applies to a headless turn — use it with -p/--prompt")
    if args.json and args.prompt is None:
        parser.error("--json only applies to a headless turn — use it with -p/--prompt")
    return args


def _read_piped_stdin() -> str:
    """Piped stdin content for a headless turn, or "" when stdin is a TTY / closed / empty.
    Read as BYTES (sys.stdin.buffer) and decoded as UTF-8 with errors='replace': Windows opens a
    piped text-mode stdin as STRICT cp1252, so `git diff | saturn -p ...` would either mojibake
    the diff or raise UnicodeDecodeError on the first non-cp1252 byte — and a blanket except
    would then silently drop the whole pipe. A genuine OS read failure may still return "", but
    a decode can never empty the input. Clamped to the same per-attachment budget as an @file
    mention (mentions._MAX_FILE_CHARS — the one cap an attachment block honors), with the same
    head-only truncation marker."""
    try:
        stdin = sys.stdin
        if stdin is None or stdin.closed or stdin.isatty():
            return ""
        buffer = getattr(stdin, "buffer", None)
        if buffer is not None:
            # +1 past the budget detects truncation; ×4 because the budget is CHARS and UTF-8
            # spends up to 4 bytes per char — reading only budget+1 BYTES could under-read a
            # multi-byte stream and drop its tail without the truncation marker.
            raw = buffer.read((mentions._MAX_FILE_CHARS + 1) * 4)
            data = raw.decode("utf-8", errors="replace")
        else:
            # A replaced stdin with no byte layer (embedders, tests): already-decoded text,
            # so there is no strict-decode hazard left to guard.
            data = stdin.read(mentions._MAX_FILE_CHARS + 1)
    except (OSError, ValueError, AttributeError):
        return ""
    if not data.strip():
        return ""
    if len(data) > mentions._MAX_FILE_CHARS:
        data = data[: mentions._MAX_FILE_CHARS] + (
            f"\n… [truncated — piped stdin exceeds {mentions._MAX_FILE_CHARS} chars]"
        )
    return data

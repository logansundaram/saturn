"""No new egress (transplanted from the visibility isolate's no-telemetry guard, allowlisted for
Saturn's declared boundary).

The only ways anything leaves the machine are the chokepoints CLAUDE.md names: local Ollama
inference (`core/llms.py`, the raw-mode continuation in `core/continuation.py`), a web search /
page fetch (`tools/web.py`), a configured MCP server (`tools/mcp_client.py`) — every one of them
routed through `trust/egress.py`'s check/record. This test pins that list: a network-client
import anywhere else in the source tree fails, so a new egress path can never land silently. A
legitimate new chokepoint is a deliberate edit HERE, with its egress.check/record wiring.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modules that can open a network connection (or wrap a client that does). `urllib.parse` is
# deliberately NOT here — parsing a URL is not egress (trust/egress.py uses it).
_NET_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+("
    r"requests|httpx|urllib3|urllib\.request|urllib\s*$|http\.client|socket|aiohttp|websockets?|"
    r"ollama|langchain_ollama|langchain_anthropic|langchain_openai|anthropic|openai|"
    r"ddgs|trafilatura|mcp|smtplib|ftplib|telnetlib|paramiko|boto3|botocore|"
    r"google\.(?:cloud|genai|generativeai)"
    r")\b",
    re.MULTILINE,
)

# The declared egress chokepoints (relative POSIX paths) → the clients each may import.
_ALLOWED = {
    "core/llms.py": {"httpx", "langchain_ollama", "ollama"},
    "core/continuation.py": {"httpx"},
    "tools/web.py": {"httpx", "trafilatura", "ddgs"},
    "tools/mcp_client.py": {"mcp"},
    # RAG ingest uses trafilatura's HTML→text EXTRACTION on local files only — never its fetch.
    "stores/rag.py": {"trafilatura"},
}

_SKIP_DIRS = {"tests", "dist", "build", ".venv", "venv", "__pycache__", "logging", "database",
              ".git", "docs"}


def _source_files():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        yield p, rel.as_posix()


def test_no_network_client_import_outside_the_declared_chokepoints():
    offenders = []
    for path, rel in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _NET_IMPORT.finditer(text):
            mod = m.group(1).split(".")[0].strip()
            if mod in _ALLOWED.get(rel, set()):
                continue
            offenders.append(f"{rel}: {m.group(0).strip()}")
    assert not offenders, (
        "network-client import outside the declared egress chokepoints — a new egress path "
        "must be routed through trust/egress and listed in tests/test_no_new_egress.py:\n  "
        + "\n  ".join(offenders)
    )


def test_rag_never_fetches():
    """The one non-chokepoint allowance is extraction-only: rag.py must never call trafilatura's
    fetch or import a client."""
    text = (ROOT / "stores" / "rag.py").read_text(encoding="utf-8")
    assert "fetch_url" not in text and "fetch_response" not in text
    assert not re.search(r"^\s*(?:import|from)\s+(?:httpx|requests)\b", text, re.MULTILINE)


def test_the_allowlist_names_real_files():
    for rel in _ALLOWED:
        assert (ROOT / rel).exists(), rel

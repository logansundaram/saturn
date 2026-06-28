"""
MCP client — remote Model Context Protocol tools inside the trust envelope (roadmap #12).

Servers are declared in config.yaml under `mcp.servers:` (stdio command, or a streamable-HTTP/SSE
url). At startup, registry.py calls `startup()` here: each enabled server is connected, its tools
listed, and every remote tool registered through `toolspec.register_tool_object` as a LangChain
StructuredTool — so the planner catalog, the native tool binding, /tools, /risk, the trace, and
above all the APPROVAL GATE treat a remote tool exactly like a local one. Nothing downstream knows
or cares that the implementation lives in another process.

Trust model (the same hard line as the deferred /learn design):
  - A remote tool NEVER self-declares its risk tier. MCP tool annotations (readOnlyHint etc.) are
    surfaced as advisory info in /mcp, but they never drive the gate — a malicious or sloppy
    server claiming "read-only" is exactly the attack the gate exists to stop.
  - Every MCP tool therefore fails closed to `destructive` (always prompts). The USER may relax
    that: per server via `risk:` in their own config.yaml, or per tool via the existing
    `/risk <tool> <tier> [--save]` — both are user decisions, like /allow.
  - registry.py runs `startup()` BEFORE applying the persisted /risk overrides, so a saved
    override on an MCP tool name survives restarts like any other.

Sync/async bridge: the whole agent loop is synchronous (tool_node calls `tool.invoke(args)`),
while the MCP SDK is async. All sessions live on ONE background daemon thread running an asyncio
event loop; each registered tool's sync function submits `session.call_tool(...)` to that loop via
`asyncio.run_coroutine_threadsafe` and blocks on the result with a timeout (`mcp.call_timeout`).
Tool results flow back as plain strings and are clamped by tool_node like every other observation.

Secrets: any `${VAR}` in a server's url/args/env/headers expands from the environment / the
managed .env (env_keys), so tokens never sit in config.yaml — e.g.
`Authorization: Bearer ${GITHUB_TOKEN}`. A reference to an unset var is a startup problem, not a
silent empty string... it expands to "" so the server still gets a well-formed value, but the gap
is reported (see _expand/_parse_specs).

stdio server stderr goes to `logging/mcp.log` (gitignored, mirrors diag.py's dir resolution) —
NEVER the console, where it would scribble over the rich.Live TUI.

Failure posture: best-effort everywhere. A server that fails to connect is reported (startup
problems surface next to check_models' warnings, and in /mcp and /config setup) and its tools
simply don't exist this session; a tool call that fails returns an "Error: ..." observation to the
model instead of raising; nothing here can take the REPL down. `/mcp reload` is the recovery path
(full reconnect + re-register); a call against a dropped connection also attempts one lazy
reconnect on its own.

Imports nothing project-side except leaf modules (config, diag, textutil, toolspec, env_keys), so
registry.py can import it freely; reload() touches registry/llms/permissions lazily at call time,
when they are fully initialised.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import diag
from trust import egress
from trust import redaction
from config import get_config
from textutil import map_strings, truncate
from tools.toolspec import RISK_TIERS, register_tool_object

# Fallbacks when config.yaml lacks the knobs (mirrors shell.py's local-helper style).
_DEFAULT_CONNECT_TIMEOUT = 20.0   # seconds to start + handshake a server at startup
_DEFAULT_CALL_TIMEOUT = 60.0      # seconds per remote tool call

_TRANSPORTS = ("stdio", "http", "sse")

# Tool-name constraint shared by the providers' tool-calling APIs (Ollama/OpenAI-style):
# [A-Za-z0-9_-], bounded length. Remote names are sanitized into it.
_NAME_OK = re.compile(r"[^A-Za-z0-9_-]")
_MAX_TOOL_NAME = 64

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ── config parsing ────────────────────────────────────────────────────────────


@dataclass
class ServerSpec:
    """One `mcp.servers.<name>` entry, validated and with ${VAR} references expanded."""

    name: str
    transport: str                                   # stdio | http | sse
    command: str = ""                                # stdio: executable
    args: list[str] = field(default_factory=list)    # stdio: argv tail
    env: dict[str, str] = field(default_factory=dict)  # stdio: extra child env (merged over defaults)
    url: str = ""                                    # http/sse: endpoint
    headers: dict[str, str] = field(default_factory=dict)  # http/sse: request headers
    risk: str = "destructive"                        # default tier for this server's tools
    enabled: bool = True

    @property
    def target(self) -> str:
        """One-line 'where does this point' for listings (/mcp, /privacy)."""
        if self.transport == "stdio":
            return " ".join([self.command, *self.args])
        return self.url


def _expand(value: object, missing: set[str]) -> str:
    """Expand ${VAR} references from the live environment / the managed .env (env_keys), so
    secrets stay out of config.yaml. Unset vars expand to "" and are collected in `missing`
    (reported as a startup problem rather than failing the server outright)."""
    import env_keys

    def sub(m: "re.Match[str]") -> str:
        v = env_keys.get(m.group(1))
        if v is None:
            missing.add(m.group(1))
            return ""
        return v

    return _ENV_REF.sub(sub, str(value))


def _parse_specs(problems: list[str]) -> list[ServerSpec]:
    """Read + validate `mcp.servers` from config. Invalid entries are skipped with a problem
    string (never a crash — a config typo must not take the REPL down)."""
    raw = get_config().get("mcp.servers") or {}
    if not isinstance(raw, dict):
        problems.append("mcp.servers in config.yaml must be a mapping of name -> server entry")
        return []

    specs: list[ServerSpec] = []
    for name, entry in raw.items():
        name = str(name)
        if not isinstance(entry, dict):
            problems.append(f"mcp server '{name}': entry must be a mapping (see /mcp --help)")
            continue
        missing: set[str] = set()
        command = str(entry.get("command") or "")
        url = _expand(entry.get("url") or "", missing)
        transport = str(entry.get("transport") or ("stdio" if command else "http")).lower()
        if transport not in _TRANSPORTS:
            problems.append(
                f"mcp server '{name}': unknown transport '{transport}' "
                f"(expected one of {', '.join(_TRANSPORTS)}) — skipped"
            )
            continue
        if transport == "stdio" and not command:
            problems.append(f"mcp server '{name}': transport stdio needs a `command:` — skipped")
            continue
        if transport in ("http", "sse") and not url:
            problems.append(f"mcp server '{name}': transport {transport} needs a `url:` — skipped")
            continue

        # The server-level default tier is a USER declaration (their own config file), so it may
        # relax the gate — but an invalid value fails closed, loudly.
        risk = str(entry.get("risk") or "destructive")
        if risk not in RISK_TIERS:
            problems.append(
                f"mcp server '{name}': unknown risk '{risk}' — its tools fail closed to destructive"
            )
            risk = "destructive"

        spec = ServerSpec(
            name=name,
            transport=transport,
            command=command,
            args=[_expand(a, missing) for a in (entry.get("args") or [])],
            env={str(k): _expand(v, missing) for k, v in (entry.get("env") or {}).items()},
            url=url,
            headers={str(k): _expand(v, missing) for k, v in (entry.get("headers") or {}).items()},
            risk=risk,
            enabled=bool(entry.get("enabled", True)),
        )
        if missing:
            problems.append(
                f"mcp server '{name}': ${{...}} reference(s) to unset env var(s): "
                + ", ".join(sorted(missing))
                + " — set them with `/config key set <NAME> <value>` and run /mcp reload"
            )
        specs.append(spec)
    return specs


def _connect_timeout() -> float:
    v = get_config().get("mcp.connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    try:
        n = float(v)
    except (TypeError, ValueError):
        return _DEFAULT_CONNECT_TIMEOUT
    return n if n > 0 else _DEFAULT_CONNECT_TIMEOUT


def _call_timeout() -> float:
    v = get_config().get("mcp.call_timeout", _DEFAULT_CALL_TIMEOUT)
    try:
        n = float(v)
    except (TypeError, ValueError):
        return _DEFAULT_CALL_TIMEOUT
    return n if n > 0 else _DEFAULT_CALL_TIMEOUT


# ── module state ──────────────────────────────────────────────────────────────

_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None
_SERVERS: dict[str, "_ServerState"] = {}   # name -> live state, in config order
_PROBLEMS: list[str] = []                  # startup/reload problems (surfaced like check_models')
_LOCK = threading.RLock()                  # guards launch/reload/shutdown transitions
_STDERR_LOG = None                         # shared stdio-server stderr sink (lazy)


@dataclass
class _ServerState:
    """Live connection state for one configured server. `ready` is a threading.Event because the
    sync side waits on it; `stop` is an asyncio.Event set via call_soon_threadsafe because the
    server task awaits it on the loop."""

    spec: ServerSpec
    state: str = "starting"        # starting | connected | error | disconnected | disabled
    error: str = ""
    server_info: str = ""          # "<name> <version>" from the initialize handshake
    session: object = None         # mcp.ClientSession while connected
    tools: list = field(default_factory=list)        # mcp.types.Tool listing
    registered: list[str] = field(default_factory=list)  # LangChain tool names we registered
    ready: threading.Event = field(default_factory=threading.Event)
    stop: object = None            # asyncio.Event, created inside the server task (on the loop)
    future: object = None          # concurrent.futures.Future for the running server task


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """The one background event loop every MCP session lives on (daemon thread; created on first
    use, so a config with no servers costs nothing)."""
    global _LOOP, _LOOP_THREAD
    with _LOCK:
        if _LOOP is not None and _LOOP.is_running():
            return _LOOP
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="mcp-client", daemon=True)
        thread.start()
        _LOOP, _LOOP_THREAD = loop, thread
        return loop


def _stderr_log():
    """Shared sink for stdio servers' stderr — a file under logging/ (gitignored), NEVER the
    console where it would collide with the rich.Live TUI. Mirrors diag.py's dir resolution
    (clone: logging/ at the repo root; wheel: SATURDAY_HOME/logging). Best-effort: falls back to
    os.devnull so a log failure can't block a server."""
    global _STDERR_LOG
    if _STDERR_LOG is None:
        try:
            root = Path(__file__).parent
            if (root / "config.yaml").exists():
                log_dir = root / "logging"
            else:
                log_dir = Path(os.environ.get("SATURDAY_HOME") or Path.home() / ".saturday") / "logging"
            log_dir.mkdir(parents=True, exist_ok=True)
            _STDERR_LOG = open(log_dir / "mcp.log", "a", encoding="utf-8", buffering=1)
        except Exception:
            _STDERR_LOG = open(os.devnull, "w", encoding="utf-8")
    return _STDERR_LOG


def _condense_exc(exc: BaseException) -> str:
    """Flatten an exception (anyio surfaces failures as nested ExceptionGroups) into one readable
    line for status displays and error observations."""
    leaves: list[str] = []

    def walk(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                walk(sub)
        elif isinstance(e, asyncio.CancelledError):
            leaves.append("cancelled")
        else:
            msg = str(e).strip()
            leaves.append(f"{type(e).__name__}: {msg}" if msg else type(e).__name__)

    walk(exc)
    seen: list[str] = []
    for leaf in leaves:
        if leaf not in seen:
            seen.append(leaf)
    return truncate("; ".join(seen) or "unknown error", 300)


# ── server lifecycle (runs on the background loop) ────────────────────────────


async def _server_task(st: _ServerState) -> None:
    """Own one server's whole lifetime: open the transport, handshake, list tools, then hold the
    connection open until `stop` is set (shutdown/reload) or the transport dies. The async context
    managers guarantee teardown — for stdio that's what terminates the child process."""
    spec = st.spec
    st.stop = asyncio.Event()
    try:
        from mcp import ClientSession

        if spec.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import get_default_environment, stdio_client

            params = StdioServerParameters(
                command=spec.command,
                args=spec.args,
                # Merge over the SDK's minimal default env (PATH etc.) so `npx`/`uvx` work while
                # the child still doesn't inherit the whole parent environment by accident.
                env={**get_default_environment(), **spec.env},
            )
            ctx = stdio_client(params, errlog=_stderr_log())
        elif spec.transport == "sse":
            from mcp.client.sse import sse_client

            ctx = sse_client(spec.url, headers=spec.headers or None)
        else:  # "http" — streamable HTTP, the current standard remote transport
            from mcp.client.streamable_http import streamablehttp_client

            ctx = streamablehttp_client(spec.url, headers=spec.headers or None)

        async with ctx as streams:
            read, write = streams[0], streams[1]  # streamable HTTP yields a 3rd (session id getter)
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                listed = await session.list_tools()
                st.tools = list(listed.tools or [])
                info = getattr(init, "serverInfo", None)
                if info is not None:
                    st.server_info = f"{getattr(info, 'name', '')} {getattr(info, 'version', '')}".strip()
                st.session = session
                st.state = "connected"
                diag.log(
                    f"mcp '{spec.name}' connected ({spec.transport}) — "
                    f"{len(st.tools)} tool(s), server: {st.server_info or '?'}"
                )
                st.ready.set()
                await st.stop.wait()
                st.state = "disconnected"
    except BaseException as exc:
        st.error = _condense_exc(exc)
        st.state = "disconnected" if st.state == "connected" else "error"
        diag.log(f"mcp '{spec.name}' {st.state}: {st.error}")
    finally:
        st.session = None
        if st.state == "starting":
            st.state = "error"
        st.ready.set()  # never leave a sync waiter hanging


def _launch(st: _ServerState) -> None:
    """Schedule a server's lifecycle task on the loop (non-blocking; pair with a ready.wait)."""
    st.ready = threading.Event()
    st.state = "starting"
    st.error = ""
    st.future = asyncio.run_coroutine_threadsafe(_server_task(st), _ensure_loop())


def _await_ready(states: list[_ServerState], timeout: float) -> None:
    """Wait for the given (already-launched) servers to finish connecting, sharing one deadline —
    they connect in parallel, so the budget is `timeout` total, not per server. A server that
    misses it is cancelled and marked errored."""
    deadline = time.monotonic() + timeout
    for st in states:
        remaining = max(0.0, deadline - time.monotonic())
        if not st.ready.wait(remaining) :
            if st.future is not None:
                st.future.cancel()
            st.state = "error"
            st.error = f"connect timed out after {timeout:g}s"
            diag.log(f"mcp '{st.spec.name}' {st.error}")


# ── tool registration ─────────────────────────────────────────────────────────


def _safe_name(name: str) -> str:
    return _NAME_OK.sub("_", name)[:_MAX_TOOL_NAME]


def _make_func(server: str, mcp_tool: str):
    """The sync callable behind one registered MCP tool — a closure so each StructuredTool binds
    its own (server, tool) pair."""

    def _call(**kwargs):
        return call_tool(server, mcp_tool, kwargs)

    _call.__name__ = _safe_name(f"mcp_{server}_{mcp_tool}")
    return _call


def _register_server_tools(st: _ServerState) -> list[str]:
    """Build + register a StructuredTool for each tool a connected server listed. Names are
    prefixed `mcp_<server>_` so provenance is visible everywhere a name appears (the gate prompt,
    /tools, the trace) and remote names can't shadow local tools. Collisions (with local tools or
    another server) are skipped loudly, never silently overwritten."""
    from langchain_core.tools import StructuredTool
    from tools.toolspec import _RISK  # the live name->tier view; membership == "name is taken"

    registered: list[str] = []
    for t in st.tools:
        mcp_name = getattr(t, "name", "") or ""
        lc_name = _safe_name(f"mcp_{st.spec.name}_{mcp_name}")
        if not mcp_name or not lc_name:
            continue
        if lc_name in _RISK:
            _PROBLEMS.append(
                f"mcp server '{st.spec.name}': tool '{mcp_name}' collides with an existing "
                f"tool name '{lc_name}' — skipped"
            )
            continue
        description = (getattr(t, "description", "") or "").strip() or (
            f"Remote tool '{mcp_name}' from the MCP server '{st.spec.name}'."
        )
        schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        lc_tool = StructuredTool(
            name=lc_name,
            description=description,
            args_schema=schema,  # the server's JSON schema, passed through verbatim
            func=_make_func(st.spec.name, mcp_name),
        )
        # Tier = the user's server-level declaration (validated in _parse_specs; defaults to
        # destructive). register_tool_object itself fails closed again — belt and braces.
        register_tool_object(lc_tool, st.spec.risk)
        registered.append(lc_name)
    return registered


# ── the sync call bridge (what a registered tool actually runs) ───────────────

# Exception types that mean "the connection is gone" (vs. a per-call error the server reported):
# seeing one flips the server to error so the NEXT call attempts a lazy reconnect.
_DEAD_CONNECTION_MARKERS = (
    "ClosedResourceError",
    "BrokenResourceError",
    "ConnectionError",
    "ConnectionClosed",
    "EndOfStream",
)


def _redact_args(args):
    """Deep-copy a tool-call args tree with every secret-like span replaced
    (`redaction.redact`) — the redact-mode twin of the warn-mode count at the MCP boundary.
    Walks `textutil.map_strings`, the rewrite twin of the `iter_strings` walk that warn mode's
    `redaction.scan_args` counts with — one walker, so the two modes can never disagree about
    what counts as argument content. Only string leaves change; structure and non-string
    values pass through untouched."""
    total = 0

    def _swap(s):
        nonlocal total
        new, findings = redaction.redact(s)
        total += len(findings)
        return new

    return map_strings(args, _swap), total


def call_tool(server: str, tool: str, args: dict) -> str:
    """Execute one remote tool call synchronously (the bridge tool_node ends up in). Always
    returns a string observation — errors are reported to the model, never raised, matching how
    tool_node treats local tool failures."""
    st = _SERVERS.get(server)
    if st is None:
        return f"Error: MCP server '{server}' is not configured."

    # Network boundary: a remote (http/sse) server call leaves the machine — gate it on air-gap and
    # record it to the egress ledger. A stdio server is a local child process (its own egress, if
    # any, is shown in /privacy), so it isn't gated here.
    if st.spec.transport in ("http", "sse"):
        host = egress.host_of(st.spec.url)  # the shared ledger host derivation (trust/egress.py)
        gblocked = egress.check("mcp", host, f"{server}.{tool}")
        if gblocked:
            return gblocked
        # Redaction parity with the cloud-LLM boundary (llms._CloudBoundaryModel): tool args
        # cross the wire too. `warn` counts secret-like values into the egress event; `redact`
        # replaces them in the args actually sent. The gate may have shown the human the call,
        # but a tier relaxed via /risk sends without a prompt — the boundary itself can't be blind.
        redactions = 0
        if redaction.active():
            if redaction.mode() == "redact":
                args, redactions = _redact_args(args or {})
            else:
                redactions = len(redaction.scan_args(args or {}))
        try:
            n_bytes = len(json.dumps(args or {}, default=str))
        except Exception:
            n_bytes = 0
        egress.record("mcp", host, f"{server}.{tool}", provider=server,
                      n_bytes=n_bytes, redactions=redactions)

    # Lazy reconnect: a server that crashed or dropped (state error/disconnected) gets ONE fresh
    # connection attempt per call. /mcp reload remains the full recovery (re-lists + re-registers).
    if st.state != "connected" or st.session is None:
        with _LOCK:
            if st.state != "connected":
                diag.log(f"mcp '{server}': reconnecting for call to '{tool}'")
                _launch(st)
                _await_ready([st], _connect_timeout())
        if st.state != "connected" or st.session is None:
            return (
                f"Error: MCP server '{server}' is not connected"
                f"{f' ({st.error})' if st.error else ''} — the user can run /mcp reload."
            )

    timeout = _call_timeout()
    start = time.perf_counter()
    try:
        fut = asyncio.run_coroutine_threadsafe(
            st.session.call_tool(tool, args or {}, read_timeout_seconds=timedelta(seconds=timeout)),
            _ensure_loop(),
        )
        try:
            result = fut.result(timeout + 5)  # outer belt over the protocol-level read timeout
        except concurrent.futures.TimeoutError:
            fut.cancel()
            return f"Error: MCP tool '{tool}' on server '{server}' timed out after {timeout:g}s."
    except Exception as exc:
        msg = _condense_exc(exc)
        if any(marker in msg for marker in _DEAD_CONNECTION_MARKERS):
            st.state = "error"
            st.error = msg
        return f"Error calling MCP tool '{tool}' on server '{server}': {msg}"
    finally:
        diag.log(f"mcp {server}.{tool} : {time.perf_counter() - start:.4f}s")

    return _result_text(result)


def _result_text(result) -> str:
    """Flatten a CallToolResult into the plain-string observation the loop expects. Text content
    passes through; binary content is summarized, not dumped (the gotcha-#5 rule — and base64
    would be clamped into garbage anyway); structured-only results render as JSON."""
    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(getattr(item, "text", "") or "")
        elif kind in ("image", "audio"):
            mime = getattr(item, "mimeType", "?")
            size = len(getattr(item, "data", "") or "")
            parts.append(f"[{kind} content ({mime}, ~{size * 3 // 4} bytes) — not rendered]")
        elif kind == "resource":
            res = getattr(item, "resource", None)
            text = getattr(res, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(f"[binary resource: {getattr(res, 'uri', '?')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {getattr(item, 'uri', '?')}]")

    structured = getattr(result, "structuredContent", None)
    if not any(p.strip() for p in parts) and structured is not None:
        try:
            parts = [json.dumps(structured, ensure_ascii=False, default=str)]
        except Exception:
            parts = [str(structured)]

    text = "\n".join(p for p in parts if p).strip() or "(empty result)"
    if getattr(result, "isError", False):
        return f"Error from MCP tool: {text}"
    return text


# ── public lifecycle API ──────────────────────────────────────────────────────


def startup() -> None:
    """Connect every enabled `mcp.servers` entry (in parallel, bounded by `mcp.connect_timeout`)
    and register each connected server's tools. Called by registry.py at import time, BEFORE the
    persisted /risk overrides apply — so a saved override on an MCP tool name works like any
    other. No servers configured -> nothing happens (no thread, no cost)."""
    with _LOCK:
        specs = _parse_specs(_PROBLEMS)
        if not specs:
            return
        for spec in specs:
            st = _ServerState(spec=spec)
            _SERVERS[spec.name] = st
            if not spec.enabled:
                st.state = "disabled"
                continue
            _launch(st)
        active = [st for st in _SERVERS.values() if st.state != "disabled"]
        _await_ready(active, _connect_timeout())
        for st in active:
            if st.state == "connected":
                st.registered = _register_server_tools(st)
                diag.log(
                    f"mcp '{st.spec.name}': registered {len(st.registered)} tool(s) "
                    f"at tier '{st.spec.risk}'"
                )
            else:
                _PROBLEMS.append(
                    f"mcp server '{st.spec.name}' failed to connect: {st.error or 'unknown error'}"
                    " — its tools are unavailable; fix `mcp.servers` in config.yaml and run /mcp reload"
                )


def reload() -> list[str]:
    """Full refresh for /mcp reload: drop every registered MCP tool from the live registry views,
    tear down all connections, re-read `mcp:` from config, reconnect, re-register, re-apply the
    persisted /risk overrides, and rebind the tool model. Returns the new problem list.

    Imports registry/llms/policy lazily — they are fully initialised by the time a slash
    command can run, and importing them at module level would be circular (registry imports us)."""
    from trust import policy
    from tools import registry
    from core.llms import reset_models

    with _LOCK:
        old = {n for st in _SERVERS.values() for n in st.registered}
        # registry.tool IS toolspec._TOOLS (same list object) — mutate in place so every holder
        # of the list (llms' bind, messages' catalog) sees the change.
        registry.tool[:] = [t for t in registry.tool if t.name not in old]
        for name in old:
            registry.tools_by_name.pop(name, None)
            registry.TOOL_RISK.pop(name, None)
            registry.DECLARED_RISK.pop(name, None)
            registry.RETRIEVAL_TOOLS.discard(name)

        _stop_servers()
        _SERVERS.clear()
        _PROBLEMS.clear()

        startup()

        # Fold the new tools into the views registry built at import time, then re-apply the
        # user's persisted per-tool tier overrides over the fresh declarations.
        new = {n for st in _SERVERS.values() for n in st.registered}
        for t in registry.tool:
            if t.name in new:
                registry.tools_by_name[t.name] = t
                registry.DECLARED_RISK[t.name] = registry.TOOL_RISK[t.name]
        for name, tier in policy.risk_overrides().items():
            if name in new and tier in RISK_TIERS:
                registry.TOOL_RISK[name] = tier

        reset_models()  # rebind the tool model so the agent sees the new tool set
        return list(_PROBLEMS)


def _stop_servers(timeout: float = 5.0) -> None:
    """Signal every live server task to exit and wait briefly — exiting the async contexts is
    what closes transports and terminates stdio child processes."""
    loop = _LOOP
    if loop is None or not loop.is_running():
        return
    futures = []
    for st in _SERVERS.values():
        if st.stop is not None:
            loop.call_soon_threadsafe(st.stop.set)
        if st.future is not None and not st.future.done():
            futures.append(st.future)
    if futures:
        concurrent.futures.wait(futures, timeout=timeout)


def shutdown() -> None:
    """Process-exit cleanup (atexit): close every connection and stop the loop thread. Best-effort
    — the thread is a daemon and stdio children also die with their pipes, this just makes the
    common path orderly."""
    with _LOCK:
        try:
            _stop_servers()
        finally:
            loop = _LOOP
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)


atexit.register(shutdown)


# ── status / readouts (for /mcp, /privacy, /config setup, startup warnings) ───


@dataclass(frozen=True)
class ToolStatus:
    name: str          # the registered LangChain name (mcp_<server>_<tool>)
    mcp_name: str      # the server's own name for it
    description: str   # first line
    hints: str         # advisory annotations, e.g. "read-only?" — NEVER drives the gate


@dataclass(frozen=True)
class ServerStatus:
    name: str
    transport: str
    target: str        # command line / url
    state: str
    error: str
    server_info: str
    risk: str
    tools: tuple


def _hints(t) -> str:
    """Render a tool's MCP annotations as short advisory text. Suffixed '?' on purpose: these are
    the SERVER's claims about itself; they inform the human, never the approval gate."""
    a = getattr(t, "annotations", None)
    if a is None:
        return ""
    out = []
    if getattr(a, "readOnlyHint", None):
        out.append("read-only?")
    if getattr(a, "destructiveHint", None):
        out.append("destructive?")
    if getattr(a, "idempotentHint", None):
        out.append("idempotent?")
    if getattr(a, "openWorldHint", None):
        out.append("open-world?")
    return " ".join(out)


def status() -> list[ServerStatus]:
    """Snapshot of every configured server for the readout commands."""
    out: list[ServerStatus] = []
    for st in _SERVERS.values():
        tools = []
        if st.state == "connected":
            for t in st.tools:
                mcp_name = getattr(t, "name", "") or ""
                lc_name = _safe_name(f"mcp_{st.spec.name}_{mcp_name}")
                if lc_name not in st.registered:
                    continue  # skipped at registration (collision)
                desc = ((getattr(t, "description", "") or "").strip().splitlines() or [""])[0]
                tools.append(ToolStatus(lc_name, mcp_name, desc, _hints(t)))
        out.append(
            ServerStatus(
                name=st.spec.name,
                transport=st.spec.transport,
                target=st.spec.target,
                state=st.state,
                error=st.error,
                server_info=st.server_info,
                risk=st.spec.risk,
                tools=tuple(tools),
            )
        )
    return out


def configured() -> bool:
    """Whether config.yaml declares any MCP servers (drives whether readouts show a section)."""
    return bool(get_config().get("mcp.servers"))


def problems() -> list[str]:
    """Startup/reload problem strings — agent.main warns these next to check_models()'s."""
    return list(_PROBLEMS)

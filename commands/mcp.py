from commands._framework import command, _print


@command(
    "mcp",
    "MCP servers: connection status, the remote tools they add, reconnect.",
    usage="/mcp | /mcp reload",
    details="""
Saturn is an MCP client: servers declared under `mcp.servers:` in config.yaml are connected at
startup and every remote tool they expose registers behind the SAME risk-tier approval gate as
the local tools (named `mcp_<server>_<tool>`; they show in /tools and the planner sees them).

Trust model — a remote tool never picks its own tier. Every MCP tool fails closed to
`destructive` (always prompts) unless YOU relax it: per server with `risk:` in config.yaml, or
per tool with /risk <tool> <tier> [--save]. The server's own annotations (read-only etc.) are
shown here as advisory hints only — they never drive the gate.

  /mcp           server connection status + the remote tools each one added
  /mcp reload    tear down every connection, re-read `mcp:` from config.yaml, reconnect and
                 re-register the tools (the recovery path after a config edit or a crashed
                 server; session-only /config edits to `mcp.*` apply too). Persisted
                 /risk --save overrides re-apply; session-only /risk overrides reset to the
                 declared tier, like every session-only setting.

Adding a server (config.yaml; secrets via ${VAR} from .env — /config key):

  mcp:
    servers:
      github:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
      internal-docs:
        url: https://mcp.example.com/mcp
        risk: read_only

Examples:
  /mcp
  /mcp reload
""",
)
def _mcp(ctx, args):
    import mcp_client
    from registry import risk_of
    from tui import ui

    if args and args[0].lower() in ("reload", "reconnect", "refresh"):
        _print("  reconnecting MCP servers…")
        mcp_client.reload()

    statuses = mcp_client.status()
    if not statuses:
        if mcp_client.configured():
            # Configured but nothing connected this session (e.g. servers added to config.yaml
            # after startup) — a reload picks them up.
            _print("  MCP servers are configured but not loaded — run /mcp reload.")
        else:
            _print("  no MCP servers configured.")
            _print("  declare them under `mcp.servers:` in config.yaml (see /mcp --help for an")
            _print("  example), then run /mcp reload. Remote tools always face the approval gate")
            _print("  unless you lower their risk tier yourself.")
        return

    connected = [s for s in statuses if s.state == "connected"]
    n_tools = sum(len(s.tools) for s in statuses)
    ui.section(
        "mcp",
        f"{len(connected)}/{len(statuses)} server(s) connected  ·  {n_tools} remote tool(s)"
        "  ·  unconfigured risk fails closed to destructive",
    )

    state_style = {
        "connected": ui.risk_style("read_only"),       # green — healthy
        "disabled": "dim",
        "starting": "dim",
        "disconnected": ui.risk_style("side_effecting"),
        "error": ui.risk_style("destructive"),
    }
    rows = []
    for s in statuses:
        label = s.name + (f"  ({s.server_info})" if s.server_info else "")
        rows.append(
            (
                label,
                (s.state, state_style.get(s.state, "")),
                (f"{s.transport}: {s.target}", "dim"),
            )
        )
    ui.table(rows)
    for s in statuses:
        if s.error and s.state != "connected":
            _print(f"    {s.name}: {s.error}")

    if n_tools:
        _print("  remote tools (hints are the server's own claims — advisory, never the gate)")
        tool_rows = []
        for s in connected:
            for t in s.tools:
                risk = risk_of(t.name)
                desc = (t.hints + "  " if t.hints else "") + t.description
                tool_rows.append((t.name, (risk, ui.risk_style(risk)), (desc, "dim")))
        ui.table(tool_rows)

    for p in mcp_client.problems():
        ui.warn(p)

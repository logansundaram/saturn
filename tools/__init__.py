"""Everything tools: the @register_tool primitive (toolspec), the active registry + risk views
(registry — importing it registers all local tools and connects MCP), the MCP client
(mcp_client), and the tool implementations grouped by domain (calculator, clock, web, files,
knowledge, memory, shell). Import `tools.registry` for the live tool list; import a tool module
directly only in tests.
"""

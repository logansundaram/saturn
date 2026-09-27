"""The graph nodes, one per file: ground -> agent -> approval -> tools -> agent … (the v2 loop,
2026-09-27). Routing helpers live with their node (route_after_agent in agent; approval routes
through Command(goto=...))."""

"""The graph nodes, one per file: ground -> plan -> plan_gate -> agent -> approval -> tools ->
update_plan -> replan -> synthesize. Routing helpers live with their node (route_after_agent in
agent, route_after_gate in plan_gate). Graph assembly stays in the root agent.py — never here.
"""

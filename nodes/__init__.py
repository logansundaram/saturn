"""The graph nodes, one per file: ground -> plan -> plan_gate -> execute -> approval -> tools ->
update_plan -> rectify -> (replan | plan_gate | synthesize). Routing helpers live with their node
(route_after_execute in execute, route_after_rectify in rectify, route_after_gate in plan_gate).
Graph assembly stays in app/graph.py — never here.
"""

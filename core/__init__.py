"""The engine room: agent state + plan helpers (state), the role->model factory (llms), every
system prompt (messages), the hardened structured-output layer (structured), the plan-as-data-bus
context builders (plan_context), tool-argument recovery for small models (tool_args), history
compaction (compaction), @file mention expansion (mentions), and the plan-review seam (plan_ops:
pause controller + plan editor).
"""

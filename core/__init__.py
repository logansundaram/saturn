"""The engine room: agent state (state), the role->model factory (llms), every system prompt
(messages), the structured-output layer + the per-task invoke options (structured), the
grounding halves (context), the answer's source numbering (sources), tool-argument recovery for
small models (tool_args), history compaction (compaction), @file mention expansion (mentions),
and the pause/steer latch the agent node consults (pause).
"""

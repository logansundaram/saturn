#!/usr/bin/env bash
# Launch Saturday.ai from anywhere — no cd, so relative arguments resolve against the
# CALLER's cwd. Prefers the repo's own venv interpreter when one exists.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$DIR/.venv/bin/python" ]; then
  exec "$DIR/.venv/bin/python" "$DIR/agent.py" "$@"
else
  exec python3 "$DIR/agent.py" "$@"
fi

"""
Plan recipes — a vetted plan captured as a reusable template (/plan save · /plan run).

A recipe is the SHAPE of a turn that worked: the plan's steps (label + intended_tool, statuses
stripped) plus the query that produced them, stored as one JSON file under `paths.recipes`
(default database/recipes/ — user data, gitignored, survives /update). Running one seeds the next
turn's planner with the saved steps instead of drafting fresh (node_registry.plan.seed_next_plan)
— execution itself is completely normal: lockstep, the approval gate, the trace, all unchanged.
The plan stops being a per-turn artifact and becomes a workflow primitive.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from textutil import safe_stem

_RECIPE_VERSION = 1


def recipes_dir() -> Path:
    """The recipe store directory (`paths.recipes`, default database/recipes), created on use.
    Tolerates a config.yaml predating the key — installed-mode users upgrade in place."""
    from config import get_config

    cfg = get_config()
    rel = cfg.get("paths.recipes", "database/recipes")
    p = Path(rel)
    if not p.is_absolute():
        p = cfg.path("database").parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _recipe_file(name: str) -> Path:
    return recipes_dir() / f"{safe_stem(name, 'recipe')}.json"


def normalize_steps(plan: list[dict]) -> list[dict]:
    """Strip a live plan down to its reusable shape: label + intended_tool only (statuses and ids
    belong to the run, not the recipe). Skipped steps are kept — the user vetted the plan as a
    whole; a step skipped once isn't a step removed."""
    return [
        {"label": str(s.get("label") or ""), "intended_tool": s.get("intended_tool")}
        for s in (plan or [])
        if str(s.get("label") or "").strip()
    ]


def save_recipe(name: str, query: str, plan: list[dict]) -> Path:
    """Persist a recipe. Raises ValueError on an empty plan (nothing to reuse)."""
    steps = normalize_steps(plan)
    if not steps:
        raise ValueError("the current plan has no steps to save")
    path = _recipe_file(name)
    payload = {
        "saturn_recipe": _RECIPE_VERSION,
        "name": path.stem,
        "query": str(query or ""),
        "steps": steps,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_recipe(name: str) -> "dict | None":
    """The stored recipe payload, or None when missing/unreadable."""
    path = _recipe_file(name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("steps"):
        return None
    return payload


def list_recipes() -> list[dict]:
    """Every stored recipe's payload, sorted by name. Unreadable files are skipped."""
    out = []
    d = recipes_dir()
    for f in sorted(d.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("steps"):
            payload.setdefault("name", f.stem)
            out.append(payload)
    return out


def delete_recipe(name: str) -> bool:
    """Remove a stored recipe; True if a file was deleted."""
    path = _recipe_file(name)
    if path.exists():
        path.unlink()
        return True
    return False

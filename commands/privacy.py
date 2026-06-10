from commands._framework import command, _print

# The roles the loop binds (see llms.get_model) — rendered in loop order.
_ROLES = ("planner", "tool_caller", "synthesizer", "utility", "judge")


@command(
    "privacy",
    "What can leave this machine right now — inference, web egress, keys, data locations.",
    details="""
Answers one question: what can leave this machine right now? The privacy claim is meant to be
checked, not believed — this is the in-app readout of the things to check.

  inference   which model serves each role, and whether it runs locally (Ollama) or is bound
              to a cloud provider. Cloud-bound roles send your prompts and context off-machine;
              that is an explicit opt-in, and this view is where it stays visible.
  web egress  what the web tools send out when the agent uses them: the search query goes to
              the search backend (Tavily with a key, keyless DuckDuckGo otherwise);
              web_extract fetches pages directly from this machine. The exact query is always
              visible in the live trace and /trace.
  api keys    which provider keys are set (set keys enable the egress above; values masked).
  your data   where everything Saturn stores actually lives on disk — all local paths.

Telemetry: none. There is nothing to configure off because nothing phones home.

Example:
  /privacy
""",
)
def _privacy(ctx, args):
    import env_keys
    from config import get_config
    from tui import ui

    cfg = get_config()

    # --- inference ---------------------------------------------------------
    bindings = []
    for role in _ROLES:
        try:
            spec = cfg.model_for_role(role)
            bindings.append((role, spec.provider, spec.model))
        except KeyError:
            continue
    bindings.append(("embedder", "ollama", cfg.embedder_model))
    cloud = sorted({p for _, p, _ in bindings if p != "ollama"})

    if cloud:
        verdict = f"cloud-bound roles send prompts+context to: {', '.join(cloud)}"
    else:
        verdict = "all inference is local — prompts, documents, and memory stay on this machine"
    ui.section("privacy", verdict)

    _print("  inference")
    ui.table(
        [
            (
                role,
                model,
                ("local", ui.risk_style("read_only")) if provider == "ollama"
                else (f"cloud — {provider}", ui.risk_style("side_effecting")),
            )
            for role, provider, model in bindings
        ]
    )

    # --- web egress ---------------------------------------------------------
    provider = str(cfg.get("web.provider", "auto"))
    tavily = env_keys.is_set("TAVILY_API_KEY")
    backend = "Tavily" if (provider == "tavily" or (provider == "auto" and tavily)) else "DuckDuckGo (keyless)"
    _print("  web egress (only when the agent uses a web tool — every call shows in /trace)")
    ui.table(
        [
            ("web_search / deep_research", f"the search query goes to {backend}"),
            ("web_extract", "fetches the page directly from this machine; extraction is local"),
        ]
    )

    # --- api keys ------------------------------------------------------------
    _print("  api keys (set keys enable the egress above — /config key to manage)")
    ui.table(
        [
            (
                k.label,
                k.name,
                ("set", ui.risk_style("side_effecting")) if env_keys.is_set(k.name)
                else ("not set", "dim"),
            )
            for k in env_keys.KNOWN_KEYS
        ]
    )

    # --- data locations -------------------------------------------------------
    _print("  your data (all local)")
    rows = []
    for label, name in (
        ("workspace", "workspace"),
        ("documents (RAG)", "documents"),
        ("memory", "memory"),
        ("traces + checkpoints", "db_sqlite"),
        ("sessions", "sessions"),
        ("exports", "exports"),
    ):
        try:
            rows.append((label, (str(cfg.path(name)), "dim")))
        except KeyError:
            continue
    ui.table(rows)

    _print("  telemetry: none — no analytics, no crash reporting, no phone-home.")
    _print("  verify it: the source is MIT-licensed, and a network monitor will show only the")
    _print("  calls listed above.")

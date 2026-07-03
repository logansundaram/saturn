from commands._framework import command, _print
from commands._utils import parse_toggle_status, split_save_flag

# One byte formatter for every trust surface (textutil.human_bytes) — the per-answer receipt and
# these readouts must render the same byte count identically, or the "receipt echoes the ledger"
# story quietly stops being true.
from textutil import human_bytes as _human_bytes

_REDACT_MODES = ("off", "warn", "redact")


@command(
    "privacy",
    "The privacy surface: what CAN leave this machine, what DID, and the controls that seal it.",
    usage="/privacy [egress [clear|n] | airgap [on|off] [--save] | "
          "redact [<mode>|preview] [--save]]",
    details="""
One front door for the whole network boundary. Bare /privacy answers "what can leave this
machine right now?"; the subcommands are its verifiable companions — what actually left, the
seal, and the secret-stripper:

  /privacy                 the posture readout: which model serves each role (local vs cloud),
                           what the web tools send out, which API keys are set, where your data
                           lives on disk. The privacy claim is meant to be checked, not believed.

  /privacy egress          the ledger: what ACTUALLY left this session — every web search, page
                           fetch, http_request, remote MCP call, and cloud-model invocation, with
                           channel/host/bytes, plus every attempt BLOCKED by air-gap. Pair it
                           with a network monitor and the two agree.
    /privacy egress 20       just the last 20 events
    /privacy egress clear    reset the in-memory ledger for this session

  /privacy airgap          seal the boundary. With no argument, prints the enforcement posture
                           (what is open vs sealed right now). When ON: web tools refuse, remote
                           MCP calls refuse, and a cloud-bound role refuses to run.
    /privacy airgap on|off   set; add --save to persist to config.yaml (`--save` with no value
                             persists the CURRENT setting without changing it)

  /privacy redact          strip secrets (API keys, tokens, private keys, JWTs, emails) from
                           prompts before they reach a cloud model. Modes: off | warn | redact.
    /privacy redact preview  scan the CURRENT context and report what WOULD be stripped — sends
                             nothing
    /privacy redact <mode> [--save]   (`--save` with no mode persists the current one unchanged)

Telemetry: none. There is nothing to configure off because nothing phones home.

Related: /trace export (a portable, replayable run record).
""",
)
def _privacy(ctx, args):
    if args:
        sub = args[0].lower()
        if sub in ("egress", "ledger"):
            return _egress(ctx, args[1:])
        if sub in ("airgap", "air-gap", "seal"):
            return _airgap(ctx, args[1:])
        if sub in ("redact", "redaction"):
            return _redact(ctx, args[1:])
        _print(f"  unknown subcommand: {sub} — usage: /privacy [egress|airgap|redact]")
        return
    _overview(ctx)


# ── bare /privacy — the posture readout ──────────────────────────────────────────────────────


def _overview(ctx):
    import env_keys
    from config import get_config
    from tui import ui

    cfg = get_config()

    # --- inference ---------------------------------------------------------
    # egress._inference is THE locality classifier (loopback-aware: a remote OLLAMA_HOST
    # classifies "remote", never "local") and offmachine_destinations THE where-list assembly —
    # reused here, not re-rolled.
    from trust.egress import _inference, offmachine_destinations

    inf = _inference()
    if inf["all_local"]:
        verdict = "all inference is local — prompts, documents, and memory stay on this machine"
    else:
        verdict = ("off-machine roles send prompts+context to: "
                   f"{', '.join(offmachine_destinations(inf))}")
    ui.section("privacy", verdict)

    _print("  inference")
    ui.table([(b["role"], b["model"], _locality_cell(b, inf, ui)) for b in inf["bindings"]])

    # --- web egress ---------------------------------------------------------
    provider = str(cfg.get("web.provider", "auto"))
    tavily = env_keys.is_set("TAVILY_API_KEY")
    backend = "Tavily" if (provider == "tavily" or (provider == "auto" and tavily)) else "DuckDuckGo (keyless)"
    _print("  web egress (only when the agent uses a web tool — every call shows in /trace)")
    ui.table(
        [
            ("web_search", f"the search query goes to {backend}"),
            ("web_extract", "fetches the page directly from this machine; extraction is local"),
            ("http_request", "sends exactly the request you approve at the gate — any URL"),
        ]
    )

    # --- mcp servers ----------------------------------------------------------
    from tools import mcp_client

    statuses = mcp_client.status()
    if statuses or mcp_client.configured():
        _print("  mcp servers (remote tools — a call sends its arguments to the server)")
        if statuses:
            ui.table(
                [
                    (
                        s.name,
                        (
                            "local process — egress is whatever this server itself does"
                            if s.transport == "stdio"
                            else f"remote — tool args go to {s.target}"
                        ),
                        (s.state, ui.risk_style("read_only") if s.state == "connected" else "dim"),
                    )
                    for s in statuses
                ]
            )
        else:
            _print("    configured in config.yaml but not loaded — run /mcp reload")
        _print("    every MCP tool faces the approval gate unless you lowered its tier (/mcp, /risk).")

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
    for label, name, note in (
        ("workspace", "workspace", ""),
        ("documents (RAG)", "documents", ""),
        ("memory", "memory", ""),
        ("traces + checkpoints", "db_sqlite", ""),
        ("sessions", "sessions", ""),
        ("exports", "exports", ""),
        ("gate policy", "permissions", "risk overrides + shell allowlist (/policy)"),
    ):
        try:
            rows.append((label, (str(cfg.path(name)), "dim"), (note, "dim")))
        except KeyError:
            continue
    ui.table(rows)

    # The EFFECTIVE mode (quarantine.mode() normalizes case + falls back to "gate" on an invalid
    # value) — the posture readout must state the mode in force, not echo a raw config string.
    from trust import quarantine

    qmode = quarantine.mode()
    _print(f"  injection quarantine: {qmode} — untrusted tool output (web/http/MCP/corpus) is")
    _print("  screened for instruction-shaped content before the model sees it.")
    _print("  telemetry: none — no analytics, no crash reporting, no phone-home.")
    _print("  verify it: the source is MIT-licensed, and a network monitor will show only the")
    _print("  calls listed above.")

    # This view is what CAN leave. The verifiable companions live behind the same front door.
    _print("")
    _print("  prove it:  /privacy egress (what actually left this session) · /privacy airgap")
    _print("  (seal + verify the boundary) · /privacy redact (strip secrets before cloud sends)")


# ── /privacy egress — the session ledger ─────────────────────────────────────────────────────


def _egress(ctx, args):
    from trust import egress
    from tui import ui

    if args and args[0].lower() in ("clear", "reset"):
        egress.clear()
        _print("  egress ledger cleared for this session.")
        return

    limit = None
    for a in args:
        if a.lstrip("+-").isdigit():
            limit = max(1, int(a))

    evs = egress.events()
    s = egress.summary()

    airgap = ""
    try:
        from config import get_config
        if bool(get_config().get("runtime.airgap", False)):
            airgap = "  ·  air-gap ON"
    except Exception:
        pass

    # A clear-emptied ledger must never read as "nothing left this machine" — the counts below
    # are since the clear. Same unknown-over-local-only contract the receipt and Glass Box apply.
    cleared = bool(s.get("cleared"))
    if not evs:
        if cleared:
            ui.section("egress", "no events since the ledger was cleared" + airgap)
            _print("  the in-memory ledger was cleared this session — earlier egress is not")
            _print("  shown here.")
        else:
            ui.section("egress", "nothing has left this machine this session" + airgap)
            _print("  the boundary has stayed closed — no web, http, MCP, or cloud-model egress.")
        return

    hosts = s["hosts"]
    headline = (
        f"{s['sent']} egress event(s), {_human_bytes(s['bytes'])} sent to "
        f"{len(hosts)} host(s)"
        + (f", {s['blocked']} blocked" if s["blocked"] else "")
        + (f", {s['redactions']} secret(s) redacted" if s["redactions"] else "")
        + airgap
    )
    ui.section("egress", headline)
    if cleared:
        _print("  (ledger cleared this session — counts are since the clear)")

    shown = evs[-limit:] if limit else evs
    if limit and len(evs) > limit:
        _print(f"  last {limit} of {len(evs)} event(s) — newest last:")

    rows = []
    for e in shown:
        when = (e.ts or "")[11:19]
        status = (
            ("BLOCKED", ui.risk_style("destructive")) if e.status == egress.BLOCKED
            else (_human_bytes(e.n_bytes), "dim")
        )
        detail = e.detail
        if e.redactions:
            detail += f"  ⟨{e.redactions} redacted⟩"
        rows.append((when, e.channel, e.host, detail, status))
    ui.table(rows, styles=["dim", "accent", None, None, None])

    if s["by_channel"]:
        mix = " · ".join(f"{v} {k}" for k, v in sorted(s["by_channel"].items()))
        _print(f"  by channel: {mix}")


# ── /privacy airgap — seal the boundary ──────────────────────────────────────────────────────


def _airgap(ctx, args):
    from trust import egress
    from config import get_config, persist
    from tui import ui

    cfg = get_config()
    toggle_args, save = split_save_flag(args)
    new = parse_toggle_status(toggle_args)

    # No on/off -> status; `--save` alone persists the CURRENT value (the shared convention —
    # it mutates nothing live, so the seal can't silently flip).
    if new is None:
        if save:
            cur = "on" if bool(cfg.get("runtime.airgap", False)) else "off"
            try:
                persist("runtime.airgap")
                _print(f"  airgap is {cur} — saved runtime.airgap to config.yaml "
                       "(no change made; /privacy airgap on|off changes it).")
            except Exception as exc:
                _print(f"  (could not persist to config.yaml: {exc})")
            return
        _show_posture(ctx, cfg, ui, egress)
        return
    if new == "invalid":
        _print(f"  usage: /privacy airgap on|off [--save]   (currently "
               f"{'on' if cfg.get('runtime.airgap', False) else 'off'})")
        return

    cfg.set("runtime.airgap", new)
    # Drop the model cache so a cloud model built while air-gap was OFF can't keep serving calls —
    # the next get_model rebuild re-checks the gate and refuses. Web tools / MCP check the gate live
    # on every call, so they need nothing here.
    try:
        from core.llms import reset_models
        reset_models()
    except Exception:
        pass

    if save:
        try:
            persist("runtime.airgap")
        except Exception as exc:
            _print(f"  (could not persist to config.yaml: {exc})")

    if new:
        offmachine = _offmachine_roles(cfg)
        _print("  ┏━ ⛓  AIR-GAP ON")
        _print("  ┃  the network boundary is SEALED. web tools, remote")
        _print("  ┃  MCP calls, and off-machine roles are blocked and")
        _print("  ┃  logged. /privacy airgap off to re-open ·")
        _print("  ┃  /privacy egress to inspect.")
        _print("  ┗━")
        if offmachine:
            roles = ", ".join(f"{r} ({p}:{m})" for r, p, m in offmachine)
            _print(f"  ⚠  off-machine role(s) will now FAIL: {roles}")
            _print("     switch to a local tier first:  /models tier workstation")
    else:
        _print("  air-gap off — network access restored.")
    if save:
        _print("  saved runtime.airgap to config.yaml (survives restart).")


def _locality_cell(b: dict, inf: dict, ui):
    """The styled locality cell for one inference binding (the /privacy inference table renders
    through this)."""
    from trust.egress import remote_ollama_label

    if b["locality"] == "local":
        return ("local", ui.risk_style("read_only"))
    if b["locality"] == "remote":
        return (f"remote — {remote_ollama_label(inf)}", ui.risk_style("side_effecting"))
    return (f"cloud — {b['provider']}", ui.risk_style("side_effecting"))


def _offmachine_roles(cfg):
    """(role, where, model) for every role whose inference LEAVES this machine — cloud-bound
    roles and Ollama roles behind a remote OLLAMA_HOST (egress._inference, the one locality
    classifier; the endpoint label via remote_ollama_label, the one spelling)."""
    from trust.egress import _inference, remote_ollama_label

    inf = _inference()
    out = []
    for b in inf["bindings"]:
        if b["locality"] == "cloud":
            out.append((b["role"], b["provider"], b["model"]))
        elif b["locality"] == "remote":
            out.append((b["role"], remote_ollama_label(inf), b["model"]))
    return out


def _show_posture(ctx, cfg, ui, egress):
    from trust.egress import _inference

    on = bool(cfg.get("runtime.airgap", False))
    inf = _inference()
    offmachine = not inf["all_local"]
    if on:
        verdict = "SEALED — web, remote MCP, and off-machine roles are blocked"
    elif offmachine:
        verdict = "open — and off-machine role(s) are sending prompts off this machine right now"
    else:
        verdict = "open — but every role is local, so nothing leaves unless a web tool is used"
    ui.section("air-gap", verdict)

    sealed = lambda: ("sealed", ui.risk_style("read_only")) if on else ("open", ui.risk_style("destructive"))

    from trust.egress import remote_ollama_label

    rows = []
    for b in inf["bindings"]:
        if b["locality"] == "local":
            rows.append((b["role"], b["model"], ("local", ui.risk_style("read_only"))))
        else:
            where = (f"remote — {remote_ollama_label(inf)}"
                     if b["locality"] == "remote" else f"cloud — {b['provider']}")
            label = f"BLOCKED — {where}" if on else where
            rows.append((b["role"], b["model"], (label, ui.risk_style("destructive"))))
    _print("  inference (off-machine roles refuse to run under air-gap)")
    ui.table(rows)

    _print("  egress paths")
    ui.table(
        [
            ("web tools", "web_search / web_extract / http_request", sealed()),
            ("remote MCP", "http/sse server calls (stdio = local process)", sealed()),
            ("off-machine models", "prompts + context to a cloud provider or remote Ollama",
             ("sealed", ui.risk_style("read_only")) if (on or not offmachine)
             else ("open", ui.risk_style("destructive"))),
        ]
    )

    s = egress.summary()
    _print(f"  this session: {s['sent']} egress event(s), {s['blocked']} blocked "
           f"— full ledger in /privacy egress")
    if not on:
        _print("  seal it with  /privacy airgap on   (then re-run /privacy airgap to verify).")


# ── /privacy redact — the cloud-boundary secret stripper ─────────────────────────────────────


def _redact(ctx, args):
    from trust import redaction
    from config import get_config, persist
    from tui import ui

    cfg = get_config()

    if args and args[0].lower() in ("preview", "scan", "check", "test"):
        return _redact_preview(ctx, redaction, ui)

    mode_args, save = split_save_flag(args)

    # No mode -> status; `--save` alone persists the CURRENT mode (the shared convention — it
    # mutates nothing live).
    if not mode_args:
        if save:
            try:
                persist("runtime.redaction")
                _print(f"  redaction mode: {redaction.mode()} — saved runtime.redaction to "
                       "config.yaml (no change made; /privacy redact <mode> changes it).")
            except Exception as exc:
                _print(f"  (could not persist to config.yaml: {exc})")
            return
        _print(f"  redaction mode: {redaction.mode()}   (off | warn | redact)")
        _print("  /privacy redact <mode> to change · /privacy redact preview to see what would")
        _print("  be stripped now.")
        return

    new = mode_args[0].lower()
    if new not in _REDACT_MODES:
        _print(f"  unknown mode {new!r} — use one of: {', '.join(_REDACT_MODES)}")
        return
    cfg.set("runtime.redaction", new)

    if save:
        try:
            persist("runtime.redaction")
        except Exception as exc:
            _print(f"  (could not persist to config.yaml: {exc})")

    m = redaction.mode()
    explain = {
        "off": "no scanning — outgoing text is sent as-is.",
        "warn": "secrets are detected + counted (see /privacy egress) but sent unmodified.",
        "redact": "secrets are replaced with [REDACTED:<kind>] before every cloud send.",
    }[m]
    _print(f"  redaction mode: {m} — {explain}")
    if save:
        _print("  saved runtime.redaction to config.yaml (survives restart).")
    if m != "off":
        if not _offmachine_roles(cfg):
            _print("  note: every role is local right now, so there is no off-machine boundary")
            _print("        to guard (redaction applies to cloud and remote-Ollama sends).")


def _redact_preview(ctx, redaction, ui):
    """Scan the live turn context for secrets and report what WOULD be stripped — sends nothing."""
    s = ctx.state or {}
    sources = [
        ("query", s.get("current_query", "")),
        ("grounding context", s.get("context", "")),
        ("attachments", s.get("attachments", "")),
    ]
    for i, m in enumerate(s.get("messages", []) or []):
        content = getattr(m, "content", None)
        if isinstance(content, str) and content.strip():
            sources.append((f"message[{i}] {type(m).__name__}", content))

    rows = []
    total = 0
    for label, text in sources:
        for f in redaction.scan(text or ""):
            total += 1
            rows.append((label, f.kind, f.preview))

    if not total:
        ui.section("redaction preview", "no secrets detected in the current context")
        _print("  nothing in your query, grounding, attachments, or scratchpad matches a secret")
        _print("  pattern — a cloud send right now would carry no detectable credentials.")
        return

    ui.section("redaction preview", f"{total} secret-like value(s) in the current context")
    ui.table(rows, styles=["dim", "accent", None])
    mode = redaction.mode()
    if mode == "redact":
        _print("  these WOULD be replaced with [REDACTED:<kind>] on the next cloud send.")
    elif mode == "warn":
        _print("  mode is `warn`: these would be flagged + counted but SENT — use")
        _print("  /privacy redact redact to strip.")
    else:
        _print("  mode is `off`: these would be sent UNMODIFIED — use /privacy redact redact")
        _print("  to strip them.")

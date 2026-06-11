from commands._framework import command, _print
from commands._utils import _parse_toggle, _ROLES

# One byte formatter for every trust surface (textutil.human_bytes) — the per-answer receipt and
# these readouts must render the same byte count identically, or the "receipt echoes the ledger"
# story quietly stops being true.
from textutil import human_bytes as _human_bytes

_REDACT_MODES = ("off", "warn", "redact")


@command(
    "privacy",
    "The privacy surface: what CAN leave this machine, what DID, and the controls that seal it.",
    usage="/privacy [egress [n|clear] | airgap [on|off] [--save] | redact [<mode>|preview] [--save]]",
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
    /privacy egress log [n]  the DURABLE log: what left across ALL sessions (survives restarts,
                             append-only, hash-chained) — the in-memory ledger's permanent twin
    /privacy egress verify   walk the durable log's hash chain and prove no entry was edited,
                             reordered, or deleted since it was written

  /privacy airgap          seal the boundary. With no argument, prints the enforcement posture
                           (what is open vs sealed right now). When ON: web tools refuse, remote
                           MCP calls refuse, and a cloud-bound role refuses to run.
    /privacy airgap on|off   toggle; add --save to persist to config.yaml

  /privacy redact          strip secrets (API keys, tokens, private keys, JWTs, emails) from
                           prompts before they reach a cloud model. Modes: off | warn | redact.
    /privacy redact preview  scan the CURRENT context and report what WOULD be stripped — sends
                             nothing
    /privacy redact <mode> [--save]

  /privacy report          the trust report: ONE signed document gathering every posture above —
                           inference bindings, gate policy, air-gap/redaction, and what actually
                           left (this session + the durable log) — into a portable attestation.
    /privacy report -o <path>   write it as a signed JSON artifact (verify with /trace verify)

Telemetry: none. There is nothing to configure off because nothing phones home.

Related: /dryrun (plan + decide everything, execute nothing) · /trace export (durable audit
record with an integrity digest).
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
        if sub in ("report", "attest", "attestation"):
            return _report(ctx, args[1:])
        _print(f"  unknown subcommand: {sub} — usage: /privacy [egress|airgap|redact|report]")
        return
    _overview(ctx)


# ── bare /privacy — the posture readout ──────────────────────────────────────────────────────


def _overview(ctx):
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
            ("web_search", f"the search query goes to {backend}"),
            ("web_extract", "fetches the page directly from this machine; extraction is local"),
            ("http_request", "sends exactly the request you approve at the gate — any URL"),
        ]
    )

    # --- mcp servers ----------------------------------------------------------
    import mcp_client

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

    # This view is what CAN leave. The verifiable companions live behind the same front door.
    _print("")
    _print("  prove it:  /privacy egress (what actually left this session) · /privacy airgap")
    _print("  (seal + verify the boundary) · /privacy redact (strip secrets before cloud sends)")


# ── /privacy egress — the session ledger ─────────────────────────────────────────────────────


def _egress(ctx, args):
    import egress
    from tui import ui

    if args and args[0].lower() in ("clear", "reset"):
        egress.clear()
        _print("  egress ledger cleared for this session.")
        return

    if args and args[0].lower() in ("log", "history", "durable"):
        return _egress_log(ctx, args[1:])
    if args and args[0].lower() in ("verify", "check", "audit"):
        return _egress_verify(ctx)

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

    if not evs:
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

    _print("  durable record across sessions: /privacy egress log  ·  verify it: /privacy egress verify")


def _egress_log(ctx, args):
    """`/privacy egress log [n]` — the DURABLE, cross-session egress record (paths.egress_log),
    the in-memory ledger's twin that survives restarts and is tamper-evident."""
    import egress
    from tui import ui

    limit = None
    for a in args:
        if a.lstrip("+-").isdigit():
            limit = max(1, int(a))

    s = egress.log_summary()
    if s["lines"] == 0:
        ui.section("egress log", "the durable egress log is empty")
        _print("  no egress has been recorded to disk yet (or runtime.egress_log is off).")
        return

    span = ""
    if s["first"] and s["last"]:
        span = f"  ·  {s['first'][:10]} → {s['last'][:10]}"
    ui.section(
        "egress log",
        f"{s['sent']} event(s), {_human_bytes(s['bytes'])} sent across {len(s['sessions'])} "
        f"session(s){span}",
    )

    rows = []
    for r in egress.read_log(limit=limit or 30):
        when = (r.get("ts") or "")[:19].replace("T", " ")
        status = (
            ("BLOCKED", ui.risk_style("destructive")) if r.get("status") == egress.BLOCKED
            else (_human_bytes(r.get("n_bytes")), "dim")
        )
        rows.append((when, r.get("channel", "?"), r.get("host", "?"),
                     str(r.get("detail", ""))[:48], status))
    ui.table(rows, styles=["dim", "accent", None, None, None])
    _print(f"  showing the last {len(rows)} of {s['lines']} line(s)  ·  /privacy egress verify "
           "to check the chain is intact")


def _egress_verify(ctx):
    """`/privacy egress verify` — walk the durable log's hash chain and report whether any line was
    edited, reordered, or removed."""
    import egress
    from tui import ui

    v = egress.verify_log()
    if not v.get("exists"):
        ui.section("egress log", "no durable log to verify yet")
        _print("  nothing has been written to paths.egress_log (or runtime.egress_log is off).")
        return
    if v["ok"]:
        ui.section("egress log", f"✓ intact — {v['lines']} line(s), "
                                 f"{len(v.get('sessions', []))} session(s)")
        _print("  every entry's hash recomputes and links to its predecessor: no line in the")
        _print("  middle of the log was edited, reordered, or deleted since it was written.")
        _print("  (a truncated tail can't be proven against here — that's the signed /trace")
        _print("   export's job for the run record.)")
    else:
        ui.section("egress log", f"⨯ BROKEN at line {v.get('broken_at')} of {v['lines']}")
        _print("  the hash chain does not verify — a line was modified, reordered, or removed")
        _print(f"  at or before entry #{v.get('broken_at')}. Treat the log as compromised.")


# ── /privacy airgap — seal the boundary ──────────────────────────────────────────────────────


def _airgap(ctx, args):
    import egress
    from config import get_config, persist
    from tui import ui

    cfg = get_config()
    save = any(a.lower() in ("--save", "-s", "save") for a in args)
    toggle_args = [a for a in args if a.lower() not in ("--save", "-s", "save")]

    # No on/off -> just show the posture.
    if not toggle_args and not save:
        _show_posture(ctx, cfg, ui, egress)
        return

    new = _parse_toggle(toggle_args, bool(cfg.get("runtime.airgap", False)))
    if new is None:
        _print(f"  usage: /privacy airgap on|off [--save]   (currently "
               f"{'on' if cfg.get('runtime.airgap', False) else 'off'})")
        return

    cfg.set("runtime.airgap", new)
    # Drop the model cache so a cloud model built while air-gap was OFF can't keep serving calls —
    # the next get_model rebuild re-checks the gate and refuses. Web tools / MCP check the gate live
    # on every call, so they need nothing here.
    try:
        from llms import reset_models
        reset_models()
    except Exception:
        pass

    if save:
        try:
            persist("runtime.airgap")
        except Exception as exc:
            _print(f"  (could not persist to config.yaml: {exc})")

    if new:
        cloud = _cloud_roles(cfg)
        _print("  ┏━ ⛓  AIR-GAP ON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        _print("  ┃  the network boundary is SEALED. web tools, remote")
        _print("  ┃  MCP calls, and cloud-bound roles are blocked and")
        _print("  ┃  logged. /privacy airgap off to re-open ·")
        _print("  ┃  /privacy egress to inspect.")
        _print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if cloud:
            roles = ", ".join(f"{r} ({p}:{m})" for r, p, m in cloud)
            _print(f"  ⚠  cloud-bound role(s) will now FAIL: {roles}")
            _print("     switch to a local tier first:  /models tier workstation")
    else:
        _print("  air-gap off — network access restored.")
    if save:
        _print("  saved runtime.airgap to config.yaml (survives restart).")


def _cloud_roles(cfg):
    """(role, provider, model) for every role bound to a non-Ollama (cloud) model."""
    out = []
    for role in _ROLES:
        try:
            spec = cfg.model_for_role(role)
        except KeyError:
            continue
        if spec.provider != "ollama":
            out.append((role, spec.provider, spec.model))
    return out


def _show_posture(ctx, cfg, ui, egress):
    on = bool(cfg.get("runtime.airgap", False))
    cloud = _cloud_roles(cfg)
    if on:
        verdict = "SEALED — web, remote MCP, and cloud-bound roles are blocked"
    elif cloud:
        verdict = "open — and cloud-bound role(s) are sending prompts off-machine right now"
    else:
        verdict = "open — but every role is local, so nothing leaves unless a web tool is used"
    ui.section("air-gap", verdict)

    sealed = lambda: ("sealed", ui.risk_style("read_only")) if on else ("open", ui.risk_style("destructive"))

    rows = []
    for role in _ROLES:
        if not _safe_spec(cfg, role):
            continue
        spec = cfg.model_for_role(role)
        if spec.provider == "ollama":
            rows.append((role, spec.model, ("local", ui.risk_style("read_only"))))
        else:
            label = "BLOCKED — cloud" if on else f"cloud — {spec.provider}"
            rows.append((role, spec.model, (label, ui.risk_style("destructive"))))
    _print("  inference (cloud-bound roles refuse to run under air-gap)")
    ui.table(rows)

    _print("  egress paths")
    ui.table(
        [
            ("web tools", "web_search / web_extract / http_request", sealed()),
            ("remote MCP", "http/sse server calls (stdio = local process)", sealed()),
            ("cloud models", "prompts + context to a cloud provider",
             ("sealed", ui.risk_style("read_only")) if (on or not cloud)
             else ("open", ui.risk_style("destructive"))),
        ]
    )

    s = egress.summary()
    _print(f"  this session: {s['sent']} egress event(s), {s['blocked']} blocked "
           f"— full ledger in /privacy egress")
    if not on:
        _print("  seal it with  /privacy airgap on   (then re-run /privacy airgap to verify).")


def _safe_spec(cfg, role) -> bool:
    try:
        cfg.model_for_role(role)
        return True
    except KeyError:
        return False


# ── /privacy redact — the cloud-boundary secret stripper ─────────────────────────────────────


def _redact(ctx, args):
    import redaction
    from config import get_config, persist
    from tui import ui

    cfg = get_config()

    if args and args[0].lower() in ("preview", "scan", "check", "test"):
        return _redact_preview(ctx, redaction, ui)

    save = any(a.lower() in ("--save", "-s", "save") for a in args)
    mode_args = [a for a in args if a.lower() not in ("--save", "-s", "save")]

    if not mode_args and not save:
        _print(f"  redaction mode: {redaction.mode()}   (off | warn | redact)")
        _print("  /privacy redact <mode> to change · /privacy redact preview to see what would")
        _print("  be stripped now.")
        return

    if mode_args:
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
        cloud = _has_cloud_role(cfg)
        if not cloud:
            _print("  note: every role is local right now, so there is no cloud boundary to guard")
            _print("        (redaction applies only when a role is bound to a cloud model).")


def _has_cloud_role(cfg) -> bool:
    for role in _ROLES:
        try:
            if cfg.model_for_role(role).provider != "ollama":
                return True
        except KeyError:
            continue
    return False


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


# ── /privacy report — the signed trust attestation ───────────────────────────────────────────


def _report(ctx, args):
    """`/privacy report [-o path]` — gather every trust surface into one document and (when a path
    is given) write it as a signed, verifiable JSON artifact."""
    import json
    from pathlib import Path

    import trust_report
    from tui import ui

    out_path = None
    it = iter(args)
    for a in it:
        if a.lower() in ("-o", "--out", "--output"):
            out_path = next(it, None)

    report = trust_report.build_report()
    inf = report["inference"]
    verdict = ("all inference local — nothing leaves unless a web tool runs"
               if inf["all_local"]
               else f"cloud-bound roles send to: {', '.join(inf['cloud_providers'])}")
    ui.section("trust report", verdict)

    # Inference
    _print("  inference")
    ui.table([
        (b["role"], b["model"],
         ("local", ui.risk_style("read_only")) if b["locality"] == "local"
         else (f"cloud — {b['provider']}", ui.risk_style("side_effecting")))
        for b in inf["bindings"]
    ])

    # Policy / boundary posture
    pol = report["policy"]
    bnd = report["boundary"]
    gate = "OPEN (gate off — nothing prompts)" if pol["gate_off"] else f"prompt above {pol['auto_approve']}"
    _print("  posture")
    ui.table([
        ("approval gate", gate,
         ("⚠", ui.risk_style("destructive")) if pol["gate_off"] else ("ok", ui.risk_style("read_only"))),
        ("air-gap", "SEALED" if bnd["airgap"] else "open",
         ("sealed", ui.risk_style("read_only")) if bnd["airgap"] else ("open", "dim")),
        ("redaction", (bnd["redaction"], "dim")),
        ("injection quarantine", (bnd["quarantine"], "dim")),
        ("risk overrides", (f"{len(pol['risk_overrides'])} tool(s)", "dim")),
        ("shell allowlist", (f"{len(pol['shell_allow'])} prefix(es)", "dim")),
    ])

    # Egress — session + durable
    se = report["egress_session"]
    du = report["egress_durable"]
    _print("  egress")
    chain = ("intact", ui.risk_style("read_only")) if du["chain_ok"] else \
            (f"BROKEN @ {du['chain_broken_at']}", ui.risk_style("destructive"))
    ui.table([
        ("this session",
         (f"{se['sent']} sent ({_human_bytes(se['bytes'])}), {se['blocked']} blocked", "dim")),
        ("durable log",
         (f"{du['sent']} sent over {du['sessions']} session(s), {_human_bytes(du['bytes'])}",
          "dim")),
        ("log integrity", "hash chain", chain),
    ])

    # Signing
    sg = report["signing"]
    if sg["available"]:
        _print(f"  signed by ed25519 key {sg['key_id']}  (publish: /trace key)")
    else:
        _print("  unsigned — install `cryptography` to sign this report")

    if out_path:
        signed = trust_report.sign_report(report)
        dest = Path(out_path).expanduser()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(signed, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            _print(f"  could not write {dest}: {e}")
            return
        _print("")
        _print(f"  signed trust report written -> {dest}")
        _print(f"    sha256 {signed['integrity']['digest']}")
        if signed.get("signature"):
            _print(f"    signed   ed25519 by key {signed['signature'].get('key_id', '?')}")
    else:
        _print("")
        _print("  write a signed, portable copy with:  /privacy report -o trust.json")

"""policy.py — the one gate-policy object: the /allow shell allowlist matcher (a security
boundary: a match skips the approval gate entirely), the persisted /risk overrides, and the
auto-approve threshold with its /autoapprove (gate-off) view."""

import policy


# ── shell_allowed: the strict matcher the approval gate calls ─────────────────────────────────


def test_shell_allowed_prefix_and_token_boundary(isolated_paths):
    assert policy.add_shell_allow("git status")
    # The prefix itself and longer commands under it match…
    assert policy.shell_allowed("git status") == "git status"
    assert policy.shell_allowed("git status --short") == "git status"
    # …but token equality, not startswith: "git statusx" must NOT match.
    assert policy.shell_allowed("git statusx") is None
    # And a shorter command never matches a longer prefix.
    assert policy.shell_allowed("git") is None


def test_shell_allowed_case_insensitive(isolated_paths):
    policy.add_shell_allow("git status")
    assert policy.shell_allowed("GIT Status -sb") == "git status"


def test_shell_allowed_refuses_metacharacters(isolated_paths):
    """A command containing chaining/redirection/substitution can do more than its first tokens
    say — it must always face the human, even when it starts with an allowlisted prefix."""
    policy.add_shell_allow("git status")
    for cmd in (
        "git status; rm -rf ~",
        "git status && rm -rf ~",
        "git status | tee out.txt",
        "git status > out.txt",
        "git status < in.txt",
        "git status `whoami`",
        "git status $(id)",
        "git status\nrm -rf ~",
        "git status & del *",
    ):
        assert policy.shell_allowed(cmd) is None, cmd


def test_shell_allowed_empty_and_whitespace(isolated_paths):
    policy.add_shell_allow("git status")
    assert policy.shell_allowed("") is None
    assert policy.shell_allowed("   ") is None


def test_add_shell_allow_dedupes_case_insensitively(isolated_paths):
    assert policy.add_shell_allow("git status")
    assert not policy.add_shell_allow("GIT STATUS")
    assert policy.shell_allow() == ["git status"]


def test_remove_shell_allow_by_index_and_text(isolated_paths):
    policy.add_shell_allow("git status")
    policy.add_shell_allow("ls -la")
    # 1-based index, matching the /allow listing.
    assert policy.remove_shell_allow("1") == "git status"
    # Exact text (case-insensitive).
    assert policy.remove_shell_allow("LS -LA") == "ls -la"
    assert policy.remove_shell_allow("nope") is None
    assert policy.shell_allow() == []


# ── risk overrides (/risk … --save) ───────────────────────────────────────────────────────────


def test_risk_override_roundtrip(isolated_paths):
    assert policy.risk_overrides() == {}
    policy.set_risk_override("run_shell", "side_effecting")
    assert policy.risk_overrides() == {"run_shell": "side_effecting"}
    assert policy.clear_risk_override("run_shell")
    assert not policy.clear_risk_override("run_shell")  # already gone
    assert policy.risk_overrides() == {}


def test_corrupt_policy_file_fails_safe(isolated_paths):
    """Garbage on disk must degrade to empty defaults, never raise into the gate."""
    path = isolated_paths / "database" / "permissions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all", encoding="utf-8")
    assert policy.shell_allow() == []
    assert policy.risk_overrides() == {}
    assert policy.shell_allowed("git status") is None


# ── the tier threshold + its /autoapprove (gate-off) view ─────────────────────────────────────


def _restore_tier(prev):
    """Put the session threshold back so the config singleton doesn't leak across tests."""
    policy.set_tier(prev)
    policy._tier_before_gate_off = None


def test_approves_tier_threshold_and_allowlist(isolated_paths):
    prev = policy.tier()
    try:
        policy.set_tier("read_only")
        # Tier threshold: at or below passes, above prompts.
        assert policy.approves("calculate", "read_only")
        assert not policy.approves("write_file", "side_effecting")
        assert not policy.approves("run_shell", "destructive", {"command": "rm -rf ~"})
        # The ONLY other way through is a persisted /allow prefix on a run_shell command…
        policy.add_shell_allow("git status")
        assert policy.approves("run_shell", "destructive", {"command": "git status --short"})
        # …which never exempts a chained command, and never exempts another tool.
        assert not policy.approves("run_shell", "destructive", {"command": "git status; rm -rf ~"})
        assert not policy.approves("write_file", "destructive", {"command": "git status"})
    finally:
        _restore_tier(prev)


def test_allow_prefix_never_covers_background_jobs(isolated_paths):
    """A prefix granted for bounded foreground runs must not authorize the same command as a
    detached, timeout-free background job — the semantics the user approved have changed."""
    prev = policy.tier()
    try:
        policy.set_tier("read_only")
        policy.add_shell_allow("npm")
        assert policy.approves("run_shell", "destructive", {"command": "npm run dev"})
        assert not policy.approves(
            "run_shell", "destructive", {"command": "npm run dev", "background": True}
        )
        # An explicit falsy background is still the foreground case.
        assert policy.approves(
            "run_shell", "destructive", {"command": "npm run dev", "background": False}
        )
    finally:
        _restore_tier(prev)


def test_set_gate_off_round_trip_restores_threshold(isolated_paths):
    """/autoapprove is a view of the threshold: on -> destructive (everything approves),
    off -> the PREVIOUS threshold, not a guess."""
    prev = policy.tier()
    try:
        policy.set_tier("side_effecting")
        policy.set_gate_off(True)
        assert policy.gate_off()
        assert policy.tier() == "destructive"
        assert policy.approves("run_shell", "destructive", {"command": "rm -rf ~"})
        policy.set_gate_off(False)
        assert not policy.gate_off()
        assert policy.tier() == "side_effecting"
        # Turning it off twice (or with nothing recorded) fails closed to read_only.
        policy.set_gate_off(False)
        assert policy.tier() == "read_only"
    finally:
        _restore_tier(prev)


def test_set_tier_unknown_fails_closed(isolated_paths):
    prev = policy.tier()
    try:
        assert policy.set_tier("nonsense") == "read_only"
        assert policy.tier() == "read_only"
    finally:
        _restore_tier(prev)

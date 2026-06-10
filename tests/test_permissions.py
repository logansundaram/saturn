"""permissions.py — the /allow shell allowlist matcher (a security boundary: a match skips the
approval gate entirely) and the persisted /risk overrides."""

import permissions


# ── shell_allowed: the strict matcher the approval gate calls ─────────────────────────────────


def test_shell_allowed_prefix_and_token_boundary(isolated_paths):
    assert permissions.add_shell_allow("git status")
    # The prefix itself and longer commands under it match…
    assert permissions.shell_allowed("git status") == "git status"
    assert permissions.shell_allowed("git status --short") == "git status"
    # …but token equality, not startswith: "git statusx" must NOT match.
    assert permissions.shell_allowed("git statusx") is None
    # And a shorter command never matches a longer prefix.
    assert permissions.shell_allowed("git") is None


def test_shell_allowed_case_insensitive(isolated_paths):
    permissions.add_shell_allow("git status")
    assert permissions.shell_allowed("GIT Status -sb") == "git status"


def test_shell_allowed_refuses_metacharacters(isolated_paths):
    """A command containing chaining/redirection/substitution can do more than its first tokens
    say — it must always face the human, even when it starts with an allowlisted prefix."""
    permissions.add_shell_allow("git status")
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
        assert permissions.shell_allowed(cmd) is None, cmd


def test_shell_allowed_empty_and_whitespace(isolated_paths):
    permissions.add_shell_allow("git status")
    assert permissions.shell_allowed("") is None
    assert permissions.shell_allowed("   ") is None


def test_add_shell_allow_dedupes_case_insensitively(isolated_paths):
    assert permissions.add_shell_allow("git status")
    assert not permissions.add_shell_allow("GIT STATUS")
    assert permissions.shell_allow() == ["git status"]


def test_remove_shell_allow_by_index_and_text(isolated_paths):
    permissions.add_shell_allow("git status")
    permissions.add_shell_allow("ls -la")
    # 1-based index, matching the /allow listing.
    assert permissions.remove_shell_allow("1") == "git status"
    # Exact text (case-insensitive).
    assert permissions.remove_shell_allow("LS -LA") == "ls -la"
    assert permissions.remove_shell_allow("nope") is None
    assert permissions.shell_allow() == []


# ── risk overrides (/risk … --save) ───────────────────────────────────────────────────────────


def test_risk_override_roundtrip(isolated_paths):
    assert permissions.risk_overrides() == {}
    permissions.set_risk_override("run_shell", "side_effecting")
    assert permissions.risk_overrides() == {"run_shell": "side_effecting"}
    assert permissions.clear_risk_override("run_shell")
    assert not permissions.clear_risk_override("run_shell")  # already gone
    assert permissions.risk_overrides() == {}


def test_corrupt_permissions_file_fails_safe(isolated_paths):
    """Garbage on disk must degrade to empty defaults, never raise into the gate."""
    path = isolated_paths / "database" / "permissions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all", encoding="utf-8")
    assert permissions.shell_allow() == []
    assert permissions.risk_overrides() == {}
    assert permissions.shell_allowed("git status") is None

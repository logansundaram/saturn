"""policy.py — the one gate-policy object: the /allow shell allowlist matcher (a security
boundary: a match skips the approval gate entirely), the persisted /risk overrides, and the
auto-approve threshold with its /autoapprove (gate-off) view."""

import pytest

from trust import policy


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


# ── shell_prefix_rejects: the public face of the metacharacter screen ────────────────────────


def test_shell_prefix_rejects_metacharacters():
    """The gate UI (always-allow prefix proposal) asks this predicate instead of re-reading the
    private regex — every metacharacter the matcher refuses must come back with a reason."""
    for bad in ("git;", "a | b", "x && y", "a > b", "a < b", "`id`", "$(id)", "a\nb", "a\rb"):
        assert policy.shell_prefix_rejects(bad), bad


def test_shell_prefix_rejects_empty():
    assert policy.shell_prefix_rejects("")
    assert policy.shell_prefix_rejects("   ")


def test_shell_prefix_accepts_plain_prefixes():
    assert policy.shell_prefix_rejects("git status") is None
    assert policy.shell_prefix_rejects("ls -la") is None


def test_shell_allowed_routes_through_the_one_screen(isolated_paths, monkeypatch):
    """shell_allowed and shell_prefix_rejects must be ONE screen — a command the predicate
    rejects can never be exempt, whatever the allowlist stores."""
    policy.add_shell_allow("git status")
    rejected = []

    real = policy.shell_prefix_rejects

    def spy(text):
        out = real(text)
        if out:
            rejected.append(text)
        return out

    monkeypatch.setattr(policy, "shell_prefix_rejects", spy)
    assert policy.shell_allowed("git status; rm -rf ~") is None
    assert rejected == ["git status; rm -rf ~"]


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


def test_garbled_policy_file_preserved_as_corrupt(isolated_paths, monkeypatch):
    """Genuinely corrupt CONTENT (bad JSON / wrong shape) is renamed aside — recoverable, and
    safe from the next _save overwriting the user's only copy of the prior posture — and the
    degradation is recorded for the startup warning (load_problem)."""
    monkeypatch.setattr(policy, "_LOAD_PROBLEM", None)  # one-shot per process — reset for the test
    path = isolated_paths / "database" / "permissions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all", encoding="utf-8")
    assert policy.shell_allow() == []
    assert not path.exists()  # the bad bytes were moved aside…
    assert path.with_name(path.name + ".corrupt").exists()  # …and kept recoverable
    assert policy.load_problem()  # agent.main warns from this at startup


def test_transient_read_failure_leaves_policy_file_in_place(isolated_paths, monkeypatch):
    """A momentary OSError (an AV/backup tool briefly holding the file — routine on Windows)
    must degrade THE READ to defaults, never displace the perfectly valid file: renaming it to
    .corrupt would silently drop the persisted /risk overrides and /allow prefixes for every
    future session."""
    import json
    from pathlib import Path

    monkeypatch.setattr(policy, "_LOAD_PROBLEM", None)
    path = isolated_paths / "database" / "permissions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"risk_overrides": {}, "shell_allow": ["git status"]}),
                    encoding="utf-8")

    real_read = Path.read_text
    state = {"failed": False}

    def flaky_read(self, *a, **k):
        if self.name == "permissions.json" and not state["failed"]:
            state["failed"] = True
            raise PermissionError("file briefly locked by another process")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert policy.shell_allow() == []  # this read degrades to defaults…
    assert policy.load_problem()  # …loudly (the startup warning reads this)
    assert path.exists()  # but the file was left exactly where it was
    assert not path.with_name(path.name + ".corrupt").exists()
    # The lock cleared: the very next read finds the persisted posture intact.
    assert policy.shell_allow() == ["git status"]


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


# --- the argument-tail screen (transplanted from the gating isolate) -------------------------
#
# A token-prefix grant validated only its HEAD, so `git log --output=<abs>` and
# `git -c core.pager=!sh -c id` rode in on a benign-looking grant. The screen re-runs at USE
# against the live command text and only ever tightens: interpreters are honored as exact
# commands only, capability-introducing flags / globs / workspace-escaping paths in the tail
# disqualify the command, and non-ASCII text (a confusable `；`) never passes the automation
# path. Deterministic — no model in the loop.


def test_prefix_grant_does_not_launder_write_path(isolated_paths):
    policy.add_shell_allow("git log")
    assert policy.approves("run_shell", "destructive",
                           {"command": r"git log --output=C:\Users\x\laundered.txt"}) is False
    assert policy.approves("run_shell", "destructive",
                           {"command": "git log --output=/tmp/laundered.txt"}) is False
    assert policy.approves("run_shell", "destructive",
                           {"command": "git log --oneline -n 5"}) is True


def test_prefix_grant_does_not_launder_exec(isolated_paths):
    policy.add_shell_allow("git")
    assert policy.approves("run_shell", "destructive",
                           {"command": 'git -c core.pager=!sh -c "id"'}) is False
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is True


def test_arg_tail_screen_is_defense_in_depth_not_a_boundary(isolated_paths):
    """The honest bound: a denylist closes the NAMED laundering paths and nothing more. An
    unlisted capability flag still rides through — asserted so the screen is never mistaken
    for the boundary (the human gate on the exact reviewed command is the boundary)."""
    policy.add_shell_allow("tar")
    assert policy.approves("run_shell", "destructive",
                           {"command": "tar --use-compress-program=sh -cf x ."}) is False
    assert policy.approves("run_shell", "destructive",
                           {"command": "tar --totals=/bin/sh -cf x ."}) is True


_EVASION_CORPUS = [
    ("cat notes.txt", "cat notes.txt ../policy.py", "arg-tail path read"),
    ("git", 'git -c core.pager=!sh -c "id"', "config-exec laundering"),
    ("cp", "cp notes.txt /dev/stdout", "versatile copy"),
    ("awk", 'awk \'BEGIN{system("id")}\'', "versatile interpreter -> exec"),
    ("cat", "cat *", "glob expansion"),
    ("git status", "git status ； rm a.txt", "unicode lookalike ';' (inert in sh/pwsh)"),
    ("python", "python evil.py", "unlisted-by-a-denylist interpreter runs a script"),
    ("npm", "npm run pwn", "package script is arbitrary code"),
    ("find", "find . -exec sh -c id ;", "find -exec"),
    ("sed", "sed -i s/a/b/ ../policy.py", "in-place edit outside the workspace"),
    ("diff", "diff notes.txt /etc/passwd", "diff as a file reader"),
    ("cp", "cp ../permissions.json /tmp/x", "cp exfiltrating the policy store"),
    ("tee", "tee ../escape.txt", "tee as a writer outside the workspace"),
    ("powershell", "powershell -c Get-Process", "the host shell itself"),
    ("Get-Content", r"Get-Content ~\secrets.txt", "home-relative path"),
]


@pytest.mark.parametrize("grant,command,label", _EVASION_CORPUS)
def test_screen_evasion_corpus_routes_to_ask(isolated_paths, grant, command, label):
    policy.add_shell_allow(grant)
    assert policy.approves("run_shell", "destructive", {"command": command}) is False, label


@pytest.mark.parametrize("grant,command", [
    ("git status", "git status --short"),
    ("git log", "git log --oneline -n 5"),
    ("pytest", "pytest"),
    ("npm test", "npm test"),            # interpreter, exact match — still granted
    ("python -m pytest", "python -m pytest"),
    ("ls", "ls -la subdir"),
    ("Get-ChildItem", "Get-ChildItem subdir -Recurse"),
])
def test_screen_does_not_over_block_benign_commands(isolated_paths, grant, command):
    """The other half of the screen's job: a grant has to stay USEFUL, or the fix is approval
    fatigue with extra steps."""
    policy.add_shell_allow(grant)
    assert policy.approves("run_shell", "destructive", {"command": command}) is True


def test_shell_prefix_rejects_non_ascii():
    assert policy.shell_prefix_rejects("git status ；rm a") is not None
    assert policy.shell_prefix_rejects("git status") is None


def test_arg_tail_rejects_is_pure_and_names_the_reason():
    assert policy.arg_tail_rejects("git log", "git log --oneline") is None
    why = policy.arg_tail_rejects("git log", "git log --output=/x")
    assert why and "--output" in why
    why = policy.arg_tail_rejects("cat", "cat *.txt")
    assert why and "glob" in why
    why = policy.arg_tail_rejects("cat", "cat ../x")
    assert why and "outside the workspace" in why
    why = policy.arg_tail_rejects("python", "python evil.py")
    assert why and "interpreter" in why


def test_grant_shell_prefix_refuses_a_prefix_the_tail_screen_would_never_honor(isolated_paths):
    """The gate's always-allow flow (dry-run) must not collect a grant that could not exempt the
    reviewed command — and the disclosure says why."""
    ok, why = policy.grant_shell_prefix("git log", "git log --output=/x", dry_run=True)
    assert ok is False and "--output" in why
    # The FULL command as the prefix (the gate's default proposal) still works — the tail is
    # empty, so the reviewed command is exactly what is exempted.
    ok, why = policy.grant_shell_prefix("git log --output=out.txt", "git log --output=out.txt",
                                        dry_run=True)
    assert ok is True

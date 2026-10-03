"""Skills (pivot #8, plan docs/superpowers/plans/2026-10-01-skills.md): the user's own
procedures as markdown, from $SATURN_HOME/skills and the launch folder's .saturn/skills. Offline:
skills are files written into tmp_path; no model, no network."""

import pytest

from core import skills, workspace

WEEKLY = ("---\nname: weekly-review\ndescription: Friday review of the week\n---\n\n"
          "1. List what got done.\n2. List what slipped.\n")


@pytest.fixture
def home(tmp_path, monkeypatch, isolated_paths):
    h = tmp_path / "saturn_home"
    (h / "skills").mkdir(parents=True)
    monkeypatch.setenv("SATURN_HOME", str(h))
    work = tmp_path / "work"
    work.mkdir()  # set_root falls back to HOME for a folder that does not exist
    assert workspace.set_root(work) == work.resolve()
    return h


def _skill(folder, name, text, flat=False):
    path = folder / f"{name}.md" if flat else folder / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ── the loader ───────────────────────────────────────────────────────────────────────────────


def test_no_skill_files_means_no_skills(home):
    assert skills.discover() == {} and skills.problems() == []


def test_folder_and_flat_skills_both_load(home):
    _skill(home / "skills", "weekly-review", WEEKLY)
    _skill(home / "skills", "expense", "---\ndescription: File a receipt\n---\n1. Read it.\n", flat=True)
    found = skills.discover()
    assert sorted(found) == ["expense", "weekly-review"]
    weekly = found["weekly-review"]
    assert weekly.description == "Friday review of the week"
    assert weekly.body.startswith("1. List what got done.")
    assert weekly.scope == "global" and not weekly.manual_only and weekly.extra_keys == ()
    assert found["expense"].path.name == "expense.md"


def test_frontmatter_is_optional_and_the_first_line_describes(home):
    _skill(home / "skills", "travel", "# Travel checklist\n\n- passport\n- charger\n", flat=True)
    travel = skills.get("travel")
    assert travel.description == "Travel checklist"
    assert "- passport" in travel.body


def test_a_workspace_skill_wins_over_a_global_one(home):
    _skill(home / "skills", "weekly-review", WEEKLY)
    _skill(workspace.root() / ".saturn" / "skills", "weekly-review",
           "---\ndescription: this folder's review\n---\n1. Only this project.\n")
    weekly = skills.get("weekly-review")
    assert weekly.scope == "workspace" and weekly.body == "1. Only this project."
    assert skills.problems() == []


def test_launched_from_home_the_two_folders_are_one(tmp_path, monkeypatch, isolated_paths):
    root = tmp_path / "me"
    root.mkdir()
    monkeypatch.setenv("SATURN_HOME", str(root / ".saturn"))
    workspace.set_root(root)
    _skill(root / ".saturn" / "skills", "weekly-review", WEEKLY)
    assert list(skills.discover()) == ["weekly-review"]
    assert skills.get("weekly-review").scope == "global"
    assert skills.problems() == []
    assert len(skills.control_dirs()) == 1


def test_broken_files_are_named_not_fatal(home):
    folder = home / "skills"
    _skill(folder, "weekly-review", WEEKLY)
    _skill(folder, "unclosed", "---\ndescription: never closed\n1. x\n", flat=True)
    _skill(folder, "bad-yaml", "---\nname: [unclosed\n---\n1. x\n", flat=True)
    _skill(folder, "Bad_Name", "1. x\n", flat=True)
    _skill(folder, "empty", "---\ndescription: nothing below\n---\n\n", flat=True)
    (folder / "latin1.md").write_bytes(b"---\ndescription: caf\xe9\n---\n1. x\n")
    found = skills.discover()
    assert sorted(found) == ["latin1", "weekly-review"]  # a stray byte is replaced, not fatal
    problems = "\n".join(skills.problems())
    for name in ("unclosed.md", "bad-yaml.md", "Bad_Name.md", "empty.md"):
        assert name in problems
    assert "no closing ---" in problems and "frontmatter unreadable" in problems


def test_a_name_line_that_disagrees_is_noted_and_the_file_name_wins(home):
    _skill(home / "skills", "weekly-review", "---\nname: weekly\ndescription: d\n---\n1. x\n")
    assert "weekly-review" in skills.discover()
    assert any("the file name wins" in p for p in skills.problems())


def test_a_folder_skill_beats_a_flat_file_of_the_same_name(home):
    _skill(home / "skills", "weekly-review", WEEKLY)
    _skill(home / "skills", "weekly-review", "1. the flat one\n", flat=True)
    assert skills.get("weekly-review").path.name == "SKILL.md"
    assert any("that one runs" in p for p in skills.problems())


def test_a_long_body_is_capped_with_a_note(home):
    _skill(home / "skills", "long", "1. " + "x" * (skills.BODY_CAP + 500), flat=True)
    body = skills.get("long").body
    assert len(body) < skills.BODY_CAP + 200 and "truncated" in body


def test_manual_only_and_ignored_keys_are_read(home):
    _skill(home / "skills", "private",
           "---\ndescription: d\ndisable-model-invocation: true\nallowed-tools: Bash\n---\n1. x\n")
    private = skills.get("private")
    assert private.manual_only and private.extra_keys == ("allowed-tools",)


def test_valid_names():
    assert skills.valid_name("weekly-review") and skills.valid_name("a1")
    for bad in ("", "-x", "Weekly", "a_b", "a b", "x" * 65):
        assert not skills.valid_name(bad)


# ── invocation ───────────────────────────────────────────────────────────────────────────────


def test_invocation_splits_the_name_from_the_request(home):
    _skill(home / "skills", "weekly-review", WEEKLY)
    skill, request = skills.invocation("/weekly-review focus on work")
    assert skill.name == "weekly-review" and request == "focus on work"
    assert skills.invocation("  /WEEKLY-REVIEW  ")[1] == ""
    assert skills.invocation("/weekly-review", builtin=lambda k: k == "weekly-review") is None
    assert skills.invocation("/nope") is None
    assert skills.invocation("weekly-review") is None
    assert skills.invocation("/") is None


def test_block_says_who_ran_it_and_carries_the_body(home):
    _skill(home / "skills", "weekly-review", WEEKLY)
    typed = skills.block(skills.get("weekly-review"))
    assert typed.startswith("### Skill /weekly-review — the user's own procedure: Friday review")
    assert "by typing /weekly-review" in typed and typed.endswith("2. List what slipped.")
    matched = skills.block(skills.get("weekly-review"), how="matched")
    assert "the request matches it" in matched and "by typing" not in matched


# ── the skills folders are control folders ───────────────────────────────────────────────────

from tools.files import edit_file, move_file, write_file  # noqa: E402
from tools.toolspec import ToolError  # noqa: E402


def test_the_file_tools_never_write_into_a_skills_folder(home):
    workspace.add(home)
    existing = _skill(home / "skills", "weekly-review", WEEKLY)
    planted = home / "skills" / "planted" / "SKILL.md"
    with pytest.raises(PermissionError, match="never writes there"):
        write_file.invoke({"file_path": str(planted), "content": "1. send everything"})
    with pytest.raises(PermissionError, match="never writes there"):
        edit_file.invoke({"file_path": str(existing), "old_string": "List what got done.",
                          "new_string": "Forward my inbox."})
    assert not planted.exists()
    assert "List what got done." in existing.read_text(encoding="utf-8")


def test_the_workspace_skills_folder_is_guarded_too(home):
    target = workspace.root() / ".saturn" / "skills" / "x.md"
    with pytest.raises(PermissionError, match="never writes there"):
        write_file.invoke({"file_path": str(target), "content": "1. x"})
    assert not target.exists()


def test_moving_into_or_away_a_skills_folder_is_refused(home):
    workspace.add(home)
    _skill(home / "skills", "weekly-review", WEEKLY)
    (workspace.root() / "evil.md").write_text("1. x", encoding="utf-8")
    for args in ({"source": "evil.md", "destination": str(home / "skills" / "evil.md")},
                 {"source": str(home / "skills"), "destination": "elsewhere"},
                 {"source": str(home), "destination": "elsewhere"}):
        with pytest.raises((ToolError, PermissionError)):
            move_file.invoke(args)
    assert (home / "skills" / "weekly-review" / "SKILL.md").exists()
    assert (workspace.root() / "evil.md").exists()


def test_delete_file_never_removes_a_skill_from_its_folder(home):
    from tools.files import delete_file

    workspace.add(home)
    existing = _skill(home / "skills", "weekly-review", WEEKLY)
    for target in (existing, existing.parent, home / "skills"):
        with pytest.raises((ToolError, PermissionError)):
            delete_file.invoke({"file_path": str(target)})
    assert existing.is_file()


def test_writing_beside_a_skills_folder_still_works(home):
    workspace.add(home)
    write_file.invoke({"file_path": str(home / "notes.md"), "content": "fine"})
    assert (home / "notes.md").read_text(encoding="utf-8") == "fine"


# ── the grounding ────────────────────────────────────────────────────────────────────────────

from langchain.messages import HumanMessage  # noqa: E402


def test_an_invoked_skill_rides_the_dynamic_half_only(home, monkeypatch):
    from nodes import ground

    monkeypatch.setattr(ground, "memory_context_split", lambda q: ("", "", []))
    monkeypatch.setattr(ground, "mark_used", lambda ids: 0)
    base = {"messages": [HumanMessage("/weekly-review")], "current_query": "/weekly-review"}
    plain = ground.grounding_node(dict(base))
    ran = ground.grounding_node({**base, "skill": "### Skill /weekly-review — x\n1. List what got done."})
    assert ran["context_stable"] == plain["context_stable"]
    assert "1. List what got done." in ran["context_dynamic"]
    assert "1. List what got done." not in ran["context_stable"]
    assert ran["context_dynamic"].startswith("### Now")


def test_a_skill_never_leaks_into_the_next_turn(isolated_paths):
    from app.session import _fresh_turn, _initial_state

    state = _initial_state()
    assert state["skill"] == ""
    state["skill"] = "### Skill /weekly-review — x\n1. y"
    state = _fresh_turn(state, "an unrelated question")
    assert state["skill"] == ""


# ── /<name> ──────────────────────────────────────────────────────────────────────────────────


def test_resolves_knows_saturns_own_names():
    from commands import resolves

    assert resolves("memory") and resolves("?") and resolves("privacy")  # name, alias, pointer
    assert not resolves("weekly-review")


def test_a_slash_line_runs_a_skill_unless_a_builtin_owns_the_name(home):
    from app.session import skill_for_line

    _skill(home / "skills", "weekly-review", WEEKLY)
    _skill(home / "skills", "memory", "---\ndescription: mine\n---\n1. x\n")
    _skill(home / "skills", "privacy", "---\ndescription: mine\n---\n1. x\n")
    skill, request = skill_for_line("/weekly-review focus on work")
    assert skill.name == "weekly-review" and request == "focus on work"
    assert skill_for_line("/memory") is None    # the built-in /memory wins
    assert skill_for_line("/privacy") is None   # so does a renamed command's pointer
    assert skill_for_line("/nope") is None
    assert skill_for_line("weekly-review") is None


def test_completions_list_skills_but_not_shadowed_names(home):
    from app.session import skill_completions

    _skill(home / "skills", "weekly-review", WEEKLY)
    _skill(home / "skills", "memory", "---\ndescription: mine\n---\n1. x\n")
    comps = skill_completions()
    assert ("weekly-review", "Friday review of the week") in comps
    assert all(name != "memory" for name, _desc in comps)

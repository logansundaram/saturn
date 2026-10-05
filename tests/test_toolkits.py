"""
Toolkits — tools in groups the user can turn off. Spec:
docs/superpowers/specs/2026-10-05-toolkits-design.md.
"""

import pytest

CORE = {"plan", "ask_user", "remember", "recall", "calculate", "current_time"}
SWITCHABLE = ["files", "web", "shell", "knowledge", "notes", "calendar", "mail", "contacts",
              "reminders", "messages", "shortcuts", "desktop", "notifications", "skills"]


# ── the table and the tagging (tools/toolspec.py) ────────────────────────────────────────────


def test_the_table_is_core_first_then_the_switchable_toolkits():
    from tools import toolspec

    keys = [k for k in toolspec.TOOLKITS if not k.startswith("mcp:")]
    assert keys == ["core"] + SWITCHABLE
    assert toolspec.TOOLKITS["core"].core
    assert not any(toolspec.TOOLKITS[k].core for k in SWITCHABLE)


def test_every_registered_tool_belongs_to_a_toolkit_in_the_table():
    from tools import registry, toolspec

    for t in registry.all_tools:
        assert toolspec.toolkit_of(t.name) in toolspec.TOOLKITS, t.name


def test_the_core_is_exactly_the_six():
    from tools import registry, toolspec

    assert {t.name for t in registry.all_tools if toolspec.toolkit_of(t.name) == "core"} == CORE


def test_a_tool_takes_its_modules_name_unless_it_says_otherwise():
    from tools import toolspec

    assert toolspec.toolkit_of("read_mail") == "mail"                     # tools/mail.py
    assert toolspec.toolkit_of("search_knowledge_base") == "knowledge"    # beside remember/recall
    assert toolspec.toolkit_of("schedule_notification") == "notifications"  # tools/notify.py
    assert toolspec.toolkit_of("no_such_tool") is None


def test_every_switchable_toolkit_has_a_tool():
    from tools import registry, toolspec

    used = {toolspec.toolkit_of(t.name) for t in registry.all_tools}
    assert set(SWITCHABLE) <= used


def test_registering_under_an_unknown_toolkit_raises():
    from tools import toolspec

    with pytest.raises(ValueError, match="unknown toolkit"):
        @toolspec.register_tool("read_only", toolkit="no_such_kit")
        def never_registered():
            """never registered"""
    assert toolspec.toolkit_of("never_registered") is None


# ── the bound set and the switch (tools/registry.py) ─────────────────────────────────────────

MESSAGES = {"find_group_chats", "send_message", "read_messages"}


@pytest.fixture
def rebinds(monkeypatch):
    """Count the model rebinds a toggle asks for (core.llms.reset_models), without doing one."""
    from core import llms

    calls: list = []
    monkeypatch.setattr(llms, "reset_models", lambda: calls.append(1))
    return calls


def test_every_toolkit_is_on_by_default():
    from tools import registry

    assert registry.off_toolkits() == []
    assert [t.name for t in registry.tool] == [t.name for t in registry.all_tools]


def test_off_unbinds_a_toolkits_tools_in_place(rebinds):
    from tools import registry

    bound, by_name = registry.tool, registry.tools_by_name
    registry.set_toolkits(["messages"], False)

    assert registry.tool is bound and registry.tools_by_name is by_name   # every holder sees it
    assert MESSAGES.isdisjoint(t.name for t in bound)
    assert MESSAGES.isdisjoint(by_name)
    assert MESSAGES <= set(registry.all_by_name)
    assert {t.name for t in bound} == set(by_name)


def test_on_restores_registration_order(rebinds):
    from tools import registry

    registry.set_toolkits(["files", "messages"], False)
    registry.set_toolkits(["messages", "files"], True)

    assert [t.name for t in registry.tool] == [t.name for t in registry.all_tools]


def test_a_toggle_rebinds_the_model_once_and_only_when_something_changed(rebinds):
    from tools import registry

    assert registry.set_toolkits(["messages", "mail"], False) == ["messages", "mail"]
    assert len(rebinds) == 1
    assert registry.set_toolkits(["messages"], False) == []      # already off
    assert len(rebinds) == 1


def test_off_toolkits_and_is_off_follow_the_switch(rebinds):
    from tools import registry

    registry.set_toolkits(["web", "calendar"], False)

    assert registry.off_toolkits() == ["web", "calendar"]          # the table's order
    assert registry.is_off("web_search") and registry.is_off("list_calendar_events")
    assert not registry.is_off("read_file")
    assert not registry.is_off("no_such_tool")                    # unknown is not "off"


def test_the_risk_tables_and_the_scanners_sets_do_not_move(rebinds):
    from tools import registry
    from trust import quarantine

    risk, declared = dict(registry.TOOL_RISK), dict(registry.DECLARED_RISK)
    registry.set_toolkits(["messages", "web", "shell"], False)

    assert registry.TOOL_RISK == risk and registry.DECLARED_RISK == declared
    assert registry.risk_of("send_message") == "destructive"
    assert quarantine.is_untrusted("web_search")
    assert registry.is_action("send_message")


@pytest.mark.parametrize("keys, why", [
    (["core"], "always on"),
    (["no_such_kit"], "no toolkit"),
    (["messages", "no_such_kit"], "no toolkit"),
])
def test_a_toolkit_that_cannot_be_switched_is_refused_and_nothing_changes(rebinds, keys, why):
    from tools import registry

    with pytest.raises(ValueError, match=why):
        registry.set_toolkits(keys, False)
    assert registry.off_toolkits() == [] and rebinds == []


def test_a_toolkit_managed_elsewhere_is_refused(rebinds):
    from tools import registry, toolspec

    toolspec.add_toolkit("mcp:demo", toolspec.Toolkit("MCP demo", "x", managed_by="/mcp"))
    try:
        with pytest.raises(ValueError, match="/mcp"):
            registry.set_toolkits(["mcp:demo"], False)
    finally:
        toolspec.TOOLKITS.pop("mcp:demo")


def test_the_config_block_decides_what_is_bound_at_startup(monkeypatch):
    from config import get_config
    from tools import registry

    monkeypatch.setitem(get_config()._data, "toolkits", {"shortcuts": False, "mail": True})
    registry.apply_toolkits()

    assert registry.off_toolkits() == ["shortcuts"]
    assert "run_shortcut" not in registry.tools_by_name and "read_mail" in registry.tools_by_name


@pytest.mark.parametrize("block, bound, said", [
    ({"core": False}, "calculate", "core"),                      # ignored: core stays on
    ({"mesages": False}, "send_message", "mesages"),             # an unknown key turns nothing off
    ({"mail": "nope"}, "read_mail", "mail"),                     # not true/false: treated as on
])
def test_a_bad_config_block_is_reported_and_fails_toward_on(monkeypatch, block, bound, said):
    from config import get_config
    from tools import registry

    monkeypatch.setitem(get_config()._data, "toolkits", block)
    registry.apply_toolkits()

    assert bound in registry.tools_by_name
    problems = registry.toolkit_problems()
    assert len(problems) == 1 and said in problems[0]


def test_a_block_that_is_not_a_mapping_is_reported(monkeypatch):
    from config import get_config
    from tools import registry

    monkeypatch.setitem(get_config()._data, "toolkits", ["messages"])
    registry.apply_toolkits()

    assert registry.off_toolkits() == []
    assert len(registry.toolkit_problems()) == 1


def test_a_saved_risk_override_on_a_tool_that_is_off_still_applies(monkeypatch, rebinds):
    from tools import registry
    from trust import policy

    registry.set_toolkits(["notes"], False)
    monkeypatch.setattr(policy, "risk_overrides", lambda: {"create_note": "read_only"})
    monkeypatch.setitem(registry.TOOL_RISK, "create_note", "side_effecting")
    registry.apply_risk_overrides()

    assert registry.TOOL_RISK["create_note"] == "read_only"


# ── the prompt (core/messages.py) and the grounding (nodes/ground.py) ────────────────────────


def _sys(*off):
    from core.messages import agent_sys_text

    return agent_sys_text(frozenset(off))


def test_with_every_toolkit_on_the_prompt_is_the_literal():
    from core import messages

    assert _sys() == messages._AGENT_SYS
    assert messages.agent_sys_msg().content == messages._AGENT_SYS


def test_every_sentence_a_toolkit_owns_occurs_exactly_once_in_the_literal():
    from core import messages

    for kit, sentence, _instead in messages._TOOLKIT_SENTENCES:
        assert messages._AGENT_SYS.count(sentence) == 1, (kit, sentence)
    assert messages._AGENT_SYS.count(messages._reader_sentence(frozenset())) == 1
    assert messages._AGENT_SYS.count("\n\nRules:") == 1


@pytest.mark.parametrize("key", SWITCHABLE)
def test_the_prompt_never_names_a_tool_whose_toolkit_is_off(key):
    from tools import registry

    text = _sys(key)
    for t in registry.toolkit_tools(key):
        assert t.name not in text, t.name


@pytest.mark.parametrize("off", [(k,) for k in SWITCHABLE] + [tuple(SWITCHABLE)])
def test_a_cut_leaves_no_ragged_edge(off):
    for line in _sys(*off).splitlines():
        assert line == line.rstrip(), repr(line)
        assert line.strip() != "-" and "  " not in line and not line.startswith("- ."), repr(line)


def test_web_off_keeps_the_rest_of_its_bullet():
    text = _sys("web")

    assert "- Today's date, weekday and the time are in the Now line" in text
    assert "Arithmetic comes from calculate" in text


def test_files_off_drops_the_file_sentences_and_the_edit_bullet():
    text = _sys("files")

    assert "/add-dir" not in text and "Trash" not in text
    assert "The knowledge base is searched with search_knowledge_base." in text


def test_contacts_off_still_forbids_a_guessed_address():
    text = _sys("contacts")

    assert "Never guess a person's address or number — ask the user for it." in text


def test_the_reader_sentence_lists_only_what_is_on():
    assert ("- The user's own notes, documents, reminders, contacts and messages come from "
            "the matching reader tools. ") in _sys("calendar", "mail")
    assert "- The user's own mail comes from the matching reader tools. " in _sys(
        "notes", "files", "calendar", "reminders", "contacts", "messages")
    assert "reader tools" not in _sys(
        "notes", "files", "mail", "calendar", "reminders", "contacts", "messages")


def test_the_off_line_names_the_toolkits_and_the_switch():
    text = _sys("messages", "calendar")       # given in any order, listed in the table's

    line = [ln for ln in text.splitlines() if "turned off" in ln]
    assert len(line) == 1
    assert "calendar, messages" in line[0] and "/tools on" in line[0]
    assert text.index(line[0]) < text.index("\n\nRules:")
    assert text.split("\n\nRules:")[1] == _sys().split("\n\nRules:")[1]


def test_nothing_off_means_no_off_line():
    assert "turned off" not in _sys()


def test_the_prompt_follows_the_switch(rebinds):
    from core import messages
    from tools import registry

    registry.set_toolkits(["shell", "web"], False)

    assert messages.agent_sys_msg().content == _sys("web", "shell")
    assert messages.agent_sys_msg().content == messages.agent_sys_msg().content


def test_the_knowledge_base_manifest_follows_its_toolkit(isolated_paths, rebinds):
    from nodes import ground
    from tools import registry

    assert "### Knowledge base" in ground.stable_grounding("")
    registry.set_toolkits(["knowledge"], False)
    assert "Knowledge base" not in ground.stable_grounding("")

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

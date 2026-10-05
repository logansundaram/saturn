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


# ── the backstop: a call to a tool whose toolkit is off (nodes/agent.py, nodes/tools.py) ─────

from langchain.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def _state(msgs):
    return {"messages": msgs, "current_query": str(msgs[0].content), "context": "", "plan": [],
            "iteration": 0, "tools_called": [], "tool_results": [], "documents_retrieved": [],
            "tool_events": [], "gate_events": []}


def test_hygiene_answers_a_call_to_a_tool_that_is_off(monkeypatch, rebinds):
    from nodes import agent
    from tools import registry

    registry.set_toolkits(["calendar"], False)
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("list_calendar_events", {})]))
    out = agent.agent_node(_state([HumanMessage(content="what's on tomorrow?")]))

    answer = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(answer) == 1
    assert answer[0].content == agent.TOOLKIT_OFF_TEXT.format(
        name="list_calendar_events", key="calendar")
    assert "/tools on calendar" in answer[0].content
    assert answer[0].additional_kwargs["saturn_status"] == "error"
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"   # no gate


def test_the_off_answer_comes_before_every_other_check(rebinds):
    from nodes import agent
    from tools import registry

    registry.set_toolkits(["messages"], False)
    off = agent.TOOLKIT_OFF_TEXT.format(name="send_message", key="messages")
    invented = _call("send_message", {"text": "hi", "to": "+15550001"})

    assert agent._hygiene(invented, [], provenance=("", ""))[1].content == off
    assert agent._hygiene(invented, [], malformed=True)[1].content == off


def test_the_wrong_arguments_redirect_never_points_at_a_tool_that_is_off(rebinds):
    from nodes import agent
    from tools import registry

    slip = _call("calculate", {"file_path": "notes.txt"})
    assert "read_file" in agent._hygiene(slip, [])[1].content

    registry.set_toolkits(["files"], False)
    answer = agent._hygiene(slip, [])[1].content
    assert "read_file" not in answer and "calculate(" in answer


def test_the_incidents_note_words_an_off_toolkit_for_the_user(rebinds):
    from nodes import agent
    from tools import registry

    registry.set_toolkits(["calendar"], False)
    off = agent.TOOLKIT_OFF_TEXT.format(name="list_calendar_events", key="calendar")
    turn = [HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[_call("list_calendar_events", {})]),
            ToolMessage(content=off, tool_call_id="c1", name="list_calendar_events",
                        additional_kwargs={"saturn_status": "error"})]

    assert agent.incidents(turn) == [
        "list_calendar_events() — not run: the calendar toolkit is turned off "
        "(/tools on calendar)"]


def test_the_tools_node_cannot_run_a_tool_that_is_off(isolated_paths, rebinds):
    from nodes.tools import tool_node
    from tools import registry

    def run():
        out = tool_node(_state([HumanMessage(content="q"),
                                AIMessage(content="", tool_calls=[_call("list_directory", {})])]))
        return out["messages"][0]

    (isolated_paths / "database" / "workspace").mkdir(parents=True)
    assert "unknown tool" not in run().content                 # on: it runs

    registry.set_toolkits(["files"], False)
    refused = run()
    assert "unknown tool" in refused.content
    assert refused.additional_kwargs["saturn_status"] == "error"


# ── saving (config.py, config.default.yaml) ──────────────────────────────────────────────────

import yaml  # noqa: E402

OLD_CONFIG = "active_tier: 4b   # keep me\nruntime:\n  think: auto\n# toolkits: not this\nmcp:\n  toolkits: nor this\n"


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """A config.yaml from before toolkits existed, as the live file."""
    import config

    path = tmp_path / "config.yaml"
    path.write_text(OLD_CONFIG, encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return path


def test_the_block_lists_every_switchable_toolkit_as_on():
    from tools import registry

    block = yaml.safe_load(registry.toolkit_block())
    assert list(block) == ["toolkits"]
    assert list(block["toolkits"].items()) == [(k, True) for k in SWITCHABLE]


def test_the_template_carries_the_block_verbatim():
    from pathlib import Path

    from tools import registry

    template = (Path(__file__).resolve().parents[1] / "config.default.yaml").read_text("utf-8")
    assert registry.toolkit_block() in template
    assert yaml.safe_load(template)["toolkits"] == {k: True for k in SWITCHABLE}


def test_an_older_config_gains_the_block_by_append(config_file):
    import config
    from tools import registry

    assert config.append_block("toolkits", registry.toolkit_block()) is True

    text = config_file.read_text("utf-8")
    assert text.startswith(OLD_CONFIG)                         # no existing line is touched
    data = yaml.safe_load(text)
    assert data["toolkits"] == {k: True for k in SWITCHABLE}
    assert data["mcp"] == {"toolkits": "nor this"} and data["runtime"] == {"think": "auto"}


def test_the_block_is_appended_once(config_file):
    import config
    from tools import registry

    config.append_block("toolkits", registry.toolkit_block())
    once = config_file.read_text("utf-8")

    assert config.append_block("toolkits", registry.toolkit_block()) is False
    assert config_file.read_text("utf-8") == once


def test_an_append_survives_a_file_without_a_final_newline(config_file):
    import config
    from tools import registry

    config_file.write_text(OLD_CONFIG.rstrip("\n"), encoding="utf-8")
    config.append_block("toolkits", registry.toolkit_block())

    assert yaml.safe_load(config_file.read_text("utf-8"))["mcp"] == {"toolkits": "nor this"}


def test_a_toggle_persists_through_the_single_line_edit(config_file, monkeypatch):
    import config
    from tools import registry

    config.append_block("toolkits", registry.toolkit_block())
    before = config_file.read_text("utf-8").splitlines()
    monkeypatch.setitem(config.get_config()._data, "toolkits", {"messages": False})
    config.persist("toolkits.messages")

    after = config_file.read_text("utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1
    assert changed[0][0].startswith("  messages: true") and changed[0][1].startswith("  messages: false")
    assert changed[0][0].split("#")[1] == changed[0][1].split("#")[1]     # the comment stays


# ── the command (commands/runtime.py) ────────────────────────────────────────────────────────

from types import SimpleNamespace  # noqa: E402


@pytest.fixture
def tools_cmd(capsys, rebinds, monkeypatch):
    """Run a /tools line and return what it printed. Counts the prefix primes it starts."""
    import commands
    from core import prime

    primes: list = []
    monkeypatch.setattr(prime, "start_priming", lambda *a, **k: primes.append(1))

    def run(line):
        capsys.readouterr()
        commands.dispatch(line, SimpleNamespace(state={}, should_quit=False))
        return capsys.readouterr().out

    run.primes = primes
    return run


def _cells(out):
    """The table rows of a readout, each as its cells (the rail glyph dropped)."""
    return [ln.split()[1:] for ln in out.splitlines() if ln.split()[:1] == ["│"]]


def _row(out, key):
    """The readout's row for one toolkit or tool."""
    rows = [r for r in _cells(out) if r[0] == key]
    assert len(rows) == 1, (key, out)
    return rows[0]


def test_bare_tools_is_the_toolkit_readout_and_never_a_flip(tools_cmd, config_file):
    from tools import registry

    registry.set_toolkits(["messages"], False)
    out = tools_cmd("/tools")

    assert _row(out, "core")[1:4] == ["6", "always", "on"]
    assert _row(out, "mail")[1:3] == ["6", "on"]
    assert _row(out, "messages")[1:3] == ["3", "off"]
    assert "45 tools" in out and "42 bound" in out and "1 toolkit off" in out
    assert registry.off_toolkits() == ["messages"] and tools_cmd.primes == []
    assert config_file.read_text("utf-8") == OLD_CONFIG


def test_tools_with_a_toolkit_lists_its_tools_and_tiers(tools_cmd):
    out = tools_cmd("/tools mail")

    for name in ("list_mail", "search_mail", "read_mail", "draft_mail", "reply_mail", "update_mail"):
        assert name in out
    assert "side_effecting" in out and "read_only" in out
    assert "read_file" not in out


def test_off_unbinds_saves_and_primes(tools_cmd, config_file):
    from tools import registry

    out = tools_cmd("/tools off messages mail")

    assert registry.off_toolkits() == ["mail", "messages"]
    saved = yaml.safe_load(config_file.read_text("utf-8"))["toolkits"]
    assert saved == {k: k not in ("mail", "messages") for k in SWITCHABLE}
    assert config_file.read_text("utf-8").startswith(OLD_CONFIG)
    assert "messages off" in out and "mail off" in out and "36 of 45" in out
    assert "next request" in out
    assert tools_cmd.primes == [1]


def test_on_binds_again_and_saves(tools_cmd, config_file):
    from tools import registry

    tools_cmd("/tools off messages")
    out = tools_cmd("/tools on messages")

    assert registry.off_toolkits() == []
    assert yaml.safe_load(config_file.read_text("utf-8"))["toolkits"]["messages"] is True
    assert "messages on" in out and "45 of 45" in out


def test_session_only_leaves_the_file_alone(tools_cmd, config_file):
    from tools import registry

    out = tools_cmd("/tools off shell --session")

    assert registry.off_toolkits() == ["shell"]
    assert config_file.read_text("utf-8") == OLD_CONFIG
    assert "session only" in out


def test_one_unknown_name_refuses_the_whole_line(tools_cmd, config_file):
    from tools import registry

    out = tools_cmd("/tools off mail mesages")

    assert registry.off_toolkits() == [] and tools_cmd.primes == []
    assert config_file.read_text("utf-8") == OLD_CONFIG
    assert "mesages" in out and "messages" in out          # the closest match is offered


@pytest.mark.parametrize("line, said", [
    ("/tools off core", "always on"),
    ("/tools off", "/tools off <toolkit>"),
    ("/tools nosuch", "no toolkit"),
])
def test_what_cannot_be_done_is_said_and_changes_nothing(tools_cmd, config_file, line, said):
    from tools import registry

    assert said in tools_cmd(line)
    assert registry.off_toolkits() == [] and tools_cmd.primes == []
    assert config_file.read_text("utf-8") == OLD_CONFIG


def test_a_toolkit_already_in_that_state_is_said_to_be(tools_cmd, config_file):
    assert "mail is already on" in tools_cmd("/tools on mail")
    assert tools_cmd.primes == [] and config_file.read_text("utf-8") == OLD_CONFIG


def test_names_are_not_case_sensitive(tools_cmd, config_file):
    from tools import registry

    tools_cmd("/tools OFF Mail")
    assert registry.off_toolkits() == ["mail"]


def test_contacts_off_beside_messages_gets_a_note(tools_cmd, config_file):
    assert "look a name up" in tools_cmd("/tools off contacts")
    assert "look a name up" not in tools_cmd("/tools off messages")


def test_all_is_the_flat_list_with_each_tools_toolkit(tools_cmd):
    from tools import registry

    registry.set_toolkits(["shell"], False)
    out = tools_cmd("/tools --all")

    rows = _cells(out)
    assert sorted(r[0] for r in rows) == sorted(t.name for t in registry.all_tools)
    assert _row(out, "read_mail")[1:3] == ["mail", "read_only"]
    assert _row(out, "run_shell")[1:4] == ["shell", "off", "destructive"]
    kits = [r[1] for r in rows]                       # grouped: the table's order, core first
    assert kits == sorted(kits, key=(["core"] + SWITCHABLE).index)


def test_help_explains_and_does_not_run(tools_cmd, config_file):
    from tools import registry

    out = tools_cmd("/tools off mail --help")

    assert "/tools off" in out and "--session" in out
    assert registry.off_toolkits() == []

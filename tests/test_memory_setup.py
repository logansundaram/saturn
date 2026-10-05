"""The first-run interview (core/memory_setup.py) and the registry predicate it relies on.
Offline: `ask` is scripted, the store is isolated, the default name is stubbed."""

from config import get_config
from core import memory_setup as ms
from stores import memory_registry as mr


def _scripted(answers):
    """An `ask` that replays `answers`, then answers Enter ("") forever."""
    it = iter(answers)

    def ask(_prompt, **_kw):
        return next(it, "")

    return ask


def _quiet(_line):
    pass


def test_is_always_loaded_names_the_every_turn_layers():
    assert mr.is_always_loaded("user") and mr.is_always_loaded("commitments")
    assert mr.is_always_loaded("preferences")  # an alias folds to user first
    assert not mr.is_always_loaded("entities")
    assert not mr.is_always_loaded("memo")  # only its recent digest rides; the layer is by-match
    assert not mr.is_always_loaded("health")  # a hand-made heading is a by-match section


def _q(key):
    return next(q for q in ms.QUESTIONS if q.key == key)


def test_the_five_questions_in_order():
    assert [q.key for q in ms.QUESTIONS] == ["name", "work", "people", "help", "never"]
    assert _q("people").category == "setup-people"


def test_pieces_one_fact_question_keeps_the_answer_whole():
    assert ms.pieces(_q("name"), "  Logan  ") == ["Call me Logan"]
    assert ms.pieces(_q("work"), "product, at a small startup.") == [
        "What I do: product, at a small startup"]
    assert ms.pieces(_q("help"), "") == []


def test_pieces_people_split_on_semicolons():
    assert ms.pieces(_q("people"), "Petra, my manager; Sam, my partner;") == [
        "Petra, my manager", "Sam, my partner"]


def test_pieces_people_split_on_commas_only_between_names():
    assert ms.pieces(_q("people"), "Petra (my manager), Sam (partner), Mom in Lisbon") == [
        "Petra (my manager)", "Sam (partner)", "Mom in Lisbon"]
    # "my manager" is not a name: the comma is part of one description
    assert ms.pieces(_q("people"), "Petra, my manager") == ["Petra, my manager"]


def test_pieces_rules_read_as_rules():
    assert ms.pieces(_q("never"), "schedule anything before 10am; don't email Petra, ever") == [
        "Never schedule anything before 10am", "Don't email Petra, ever"]
    assert ms.pieces(_q("never"), "never book flights") == ["Never book flights"]


def test_rule_layer_is_loaded_every_turn():
    assert mr.is_always_loaded(ms.rule_layer())
    assert ms.layer_for(_q("never")) == ms.rule_layer()
    assert ms.layer_for(_q("people")) == "entities"


def test_system_first_name_is_the_first_word_of_the_full_name(monkeypatch):
    import pwd

    fake = type("P", (), {"pw_gecos": "Jean-Luc Picard,,,"})()
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: fake)
    assert ms.system_first_name() == "Jean-Luc"


def test_system_first_name_refuses_junk(monkeypatch):
    import pwd

    for gecos in ("", "   ", "1234", "_daemon"):
        fake = type("P", (), {"pw_gecos": gecos})()
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid, f=fake: f)
        assert ms.system_first_name() is None, gecos


_FIVE = [
    "Logan",
    "product manager at a small startup",
    "Petra, my manager; Sam, my partner",
    "keeping on top of mail",
    "never schedule anything before 10am; don't email Petra without showing me",
]


def test_five_answers_land_in_their_layers(isolated_paths):
    lines = []
    out = ms.run_interview(ask=_scripted(_FIVE), emit=lines.append)
    rules = ms.rule_layer()
    got = {(e["layer"], e["category"], e["text"], e["by"]) for e in mr.entries()}
    assert got == {
        ("user", "setup-name", "Call me Logan", "user"),
        ("user", "setup-work", "What I do: product manager at a small startup", "user"),
        ("entities", "setup-people", "Petra, my manager", "user"),
        ("entities", "setup-people", "Sam, my partner", "user"),
        ("user", "setup-help", "What I want help with: keeping on top of mail", "user"),
        (rules, "setup-never", "Never schedule anything before 10am", "user"),
        (rules, "setup-never", "Don't email Petra without showing me", "user"),
    }
    assert len(out["saved"]) == 7 and out["left"] is False
    assert str(get_config().path("memory")) in "\n".join(lines)  # where the facts live


def test_each_fact_says_which_question_it_answered(isolated_paths):
    ms.run_interview(ask=_scripted(_FIVE), emit=_quiet)
    assert {e["src"] for e in mr.entries()} == {
        "setup:name", "setup:work", "setup:people", "setup:help", "setup:never"}
    # the stamp survives the file round trip, so /memory why can read it
    assert mr.entry(ms.current(_q("name"))[0]["id"])["src"] == "setup:name"


def test_enter_skips_and_the_default_name_is_offered(isolated_paths):
    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return ""

    ms.run_interview(ask=ask, emit=_quiet, default_name="Logan")
    assert "[Enter = Logan]" in asked[0]
    assert len(asked) == 5
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]  # every other question skipped


def test_keys_choose_which_questions_are_asked(isolated_paths):
    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return "x"

    ms.run_interview(ask=ask, emit=_quiet, keys=("name", "never"))
    assert [a.split(" »")[0] for a in asked] == [
        "[1/2] What should I call you?", "[2/2] Is there anything I should never do?"]
    assert {e["category"] for e in mr.entries()} == {"setup-name", "setup-never"}


def test_a_new_answer_names_the_similar_fact_already_stored(isolated_paths):
    mr.add_memory("Petra Novak is my manager at Acme", layer="entities")
    lines = []
    ms.run_interview(ask=_scripted(["", "", "Petra Novak, my manager"]), emit=lines.append)
    assert any("similar: #1" in line and "/memory remove 1" in line for line in lines)


def test_dash_skips_even_with_a_default(isolated_paths):
    ms.run_interview(ask=_scripted(["-"]), emit=_quiet, default_name="Logan")
    assert mr.entries() == []


def test_q_stops_and_keeps_what_was_answered(isolated_paths):
    out = ms.run_interview(ask=_scripted(["Logan", "q", "never asked"]), emit=_quiet)
    assert out["left"] is True
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]


def test_ctrl_c_at_the_first_question_saves_nothing(isolated_paths):
    out = ms.run_interview(ask=_scripted([ms.INTERRUPT]), emit=_quiet, default_name="Logan")
    assert out == {"saved": [], "left": True}
    assert mr.entries() == []


def test_a_too_long_answer_is_asked_again(isolated_paths):
    lines = []
    ms.run_interview(ask=_scripted(["x" * 400, "Logan"]), emit=lines.append)
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]
    assert any("characters" in line for line in lines)


def test_an_answer_holding_a_secret_is_asked_again(isolated_paths):
    lines = []
    out = ms.run_interview(
        ask=_scripted(["Logan", "my password is hunter2-swordfish", "product manager"]),
        emit=lines.append)
    assert [e["text"] for e in mr.entries()] == ["Call me Logan", "What I do: product manager"]
    assert out["left"] is False  # the interview went on to the other questions
    assert any("a password" in line for line in lines)


def test_a_rule_loads_for_an_unrelated_request(isolated_paths, monkeypatch):
    monkeypatch.setattr(mr, "_inference_is_local", lambda: True)
    ms.run_interview(ask=_scripted(["", "", "", "", "never schedule anything before 10am"]),
                     emit=_quiet)
    always, _matched, _ids = mr.memory_context_split("book a dentist appointment for Tuesday")
    assert "Never schedule anything before 10am" in always  # the every-turn half, not by match


def test_rerun_with_enter_everywhere_changes_nothing(isolated_paths):
    ms.run_interview(ask=_scripted(_FIVE), emit=_quiet)
    path = get_config().path("memory")
    before = path.read_bytes()
    out = ms.run_interview(ask=_scripted([]), emit=_quiet)
    assert out["saved"] == []
    assert path.read_bytes() == before


def test_rerun_supersedes_a_one_fact_answer(isolated_paths):
    ms.run_interview(ask=_scripted(["Logan"]), emit=_quiet)
    lines = []
    out = ms.run_interview(ask=_scripted(["Lo"]), emit=lines.append)
    assert [e["text"] for e in mr.entries()] == ["Call me Lo"]
    assert "replaces #1" in out["saved"][0]
    assert any("now: Call me Logan" in line for line in lines)  # the old answer was shown


def test_rerun_adds_people_and_keeps_an_unchanged_name(isolated_paths):
    ms.run_interview(ask=_scripted(["Logan", "", "Petra, my manager"]), emit=_quiet)
    name_id = ms.current(_q("name"))[0]["id"]
    ms.run_interview(ask=_scripted(["Logan", "", "Jonah, an old friend"]), emit=_quiet)
    assert [e["id"] for e in ms.current(_q("name"))] == [name_id]  # same text: no write, same id
    assert [e["text"] for e in ms.current(_q("people"))] == [
        "Petra, my manager", "Jonah, an old friend"]


def _recorder():
    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return ""

    return asked, ask


def test_fresh_install_is_offered_three_questions_once(isolated_paths, monkeypatch):
    monkeypatch.setattr(ms, "system_first_name", lambda: None)
    asked, ask = _recorder()  # Enter at the offer starts it
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=True) == "interview"
    assert "Enter starts" in asked[0] and "n skips" in asked[0]  # consent first, then questions
    assert [a.split("]")[0] for a in asked[1:]] == ["[1/3", "[2/3", "[3/3"]
    assert [q.prompt in a for q, a in zip(map(_q, ms.FIRST_RUN), asked[1:])] == [True] * 3
    assert ms.FIRST_RUN == ("name", "work", "never")  # people and help wait for /memory setup
    assert ms.marker_path().exists()
    asked.clear()
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=True) == ""
    assert asked == []


def test_a_declined_offer_asks_nothing_and_is_never_made_again(isolated_paths, monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(ms, "system_first_name", lambda: "Logan")
    for reply in ("n", "no", "q", "\x1b", ms.INTERRUPT, "what?"):
        if ms.marker_path().exists():
            ms.marker_path().unlink()
        asked, lines = [], []

        def ask(prompt, _reply=reply, **_kw):
            asked.append(prompt)
            return _reply

        out = ms.offer_at_launch(ask=ask, emit=lines.append, note=_quiet, interactive=True)
        assert out == "declined", repr(reply)
        assert len(asked) == 1, repr(reply)  # the offer line only — no question followed
        assert mr.entries() == []  # the offered default name was not taken
        assert ms.marker_path().exists()
        assert any("/memory setup" in line for line in lines)  # where to find it later


def test_existing_memory_gets_the_hint_not_the_questions(isolated_paths):
    mr.add_memory("I like tea")
    asked, ask = _recorder()
    notes = []
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=notes.append, interactive=True) == "hint"
    assert asked == []
    assert notes and "/memory setup" in notes[0]
    assert ms.marker_path().exists()  # the hint is said once
    notes.clear()
    ms.offer_at_launch(ask=ask, emit=_quiet, note=notes.append, interactive=True)
    assert notes == []


def test_no_terminal_no_questions_and_no_marker(isolated_paths):
    asked, ask = _recorder()
    assert ms.offer_at_launch(ask=ask, emit=_quiet, note=_quiet, interactive=False) == ""
    assert asked == []
    assert not ms.marker_path().exists()  # the first real terminal session still gets it


def test_ctrl_c_at_launch_still_marks_done(isolated_paths, monkeypatch):
    # past the offer (Enter), Ctrl-C at the very first question
    monkeypatch.setattr(ms, "system_first_name", lambda: "Logan")
    ms.offer_at_launch(ask=_scripted(["", ms.INTERRUPT]), emit=_quiet, note=_quiet,
                       interactive=True)
    assert mr.entries() == []
    assert ms.marker_path().exists()


def test_a_crash_inside_the_interview_still_marks_done(isolated_paths, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ms, "run_interview", boom)
    try:
        ms.offer_at_launch(ask=_scripted([]), emit=_quiet, note=_quiet, interactive=True)
    except RuntimeError:
        pass
    assert ms.marker_path().exists()  # a crash must not become a question at every launch


def test_repl_offers_the_interview_after_models():
    import inspect

    from app import repl

    src = inspect.getsource(repl.run_repl)
    assert "memory_setup.offer_at_launch(" in src
    assert src.index('commands.dispatch("/models", cmd_ctx)') < src.index(
        "memory_setup.offer_at_launch(")


_TTY = type("T", (), {"isatty": staticmethod(lambda: True)})()
_NOT_TTY = type("T", (), {"isatty": staticmethod(lambda: False)})()


def test_memory_setup_needs_a_terminal(isolated_paths, ctx, monkeypatch, capsys):
    import commands

    monkeypatch.setattr("sys.stdin", _NOT_TTY)
    commands.dispatch("/memory setup", ctx)
    assert "interactive terminal" in capsys.readouterr().out
    assert mr.entries() == []


def test_memory_setup_runs_the_interview(isolated_paths, ctx, monkeypatch, capsys):
    import commands
    from tui import ui

    monkeypatch.setattr("sys.stdin", _TTY)
    monkeypatch.setattr(ms, "system_first_name", lambda: None)
    monkeypatch.setattr(ui, "ask", _scripted(["Logan"]))
    commands.dispatch("/memory setup", ctx)
    assert [e["text"] for e in mr.entries()] == ["Call me Logan"]
    assert ms.marker_path().exists()  # a later launch does not offer it again
    assert "Remembered #1" in capsys.readouterr().out


def test_memory_setup_asks_all_five(isolated_paths, ctx, monkeypatch):
    import commands
    from tui import ui

    asked = []

    def ask(prompt, **_kw):
        asked.append(prompt)
        return ""

    monkeypatch.setattr("sys.stdin", _TTY)
    monkeypatch.setattr(ms, "system_first_name", lambda: None)
    monkeypatch.setattr(ui, "ask", ask)
    commands.dispatch("/memory setup", ctx)
    assert len(asked) == 5 and asked[0].startswith("[1/5]")  # no offer line: you asked for it


def test_memory_help_lists_setup(ctx, capsys):
    import commands

    commands.dispatch("/memory --help", ctx)
    assert "/memory setup" in capsys.readouterr().out

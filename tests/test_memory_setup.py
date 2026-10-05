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

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

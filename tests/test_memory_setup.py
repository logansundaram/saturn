"""The first-run interview (core/memory_setup.py) and the registry predicate it relies on.
Offline: `ask` is scripted, the store is isolated, the default name is stubbed."""

from config import get_config
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

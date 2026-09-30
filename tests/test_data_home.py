"""Where a wheel install keeps its data: one home, $SATURN_HOME or ~/.saturn — the same folder
as the global SATURN.md and hooks.yaml. (The $SATURDAY_HOME / ~/.saturday fallbacks were cut
2026-09-30.)
config.py, diag.py and env_keys.py each carry the rule (leaf modules, no project imports);
these tests pin the three together."""

import pytest

import config
import diag
import env_keys

_RULES = (config.wheel_data_home, diag._wheel_data_home, env_keys._wheel_data_home)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SATURN_HOME", raising=False)
    monkeypatch.delenv("SATURDAY_HOME", raising=False)
    return tmp_path


def _all(expected):
    assert [rule() for rule in _RULES] == [expected] * len(_RULES)


def test_a_new_install_lives_in_dot_saturn(home):
    _all(home / ".saturn")
    assert config.saturn_home() == home / ".saturn"  # one folder with SATURN.md and hooks.yaml


def test_an_old_saturday_install_is_no_longer_read(home):
    (home / ".saturday").mkdir()
    (home / ".saturday" / "config.yaml").write_text("active_tier: 4b\n")
    _all(home / ".saturn")


def test_a_saturday_folder_without_a_config_is_not_an_install(home):
    (home / ".saturday").mkdir()  # e.g. an empty leftover
    _all(home / ".saturn")


def test_saturn_home_wins_over_the_legacy_folder(home, monkeypatch):
    (home / ".saturday").mkdir()
    (home / ".saturday" / "config.yaml").write_text("x: 1\n")
    monkeypatch.setenv("SATURN_HOME", str(home / "mine"))
    _all(home / "mine")


def test_the_old_saturday_home_variable_is_ignored(home, monkeypatch):
    monkeypatch.setenv("SATURDAY_HOME", str(home / "old"))
    _all(home / ".saturn")
    monkeypatch.setenv("SATURN_HOME", str(home / "mine"))
    _all(home / "mine")

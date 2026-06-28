"""config.py — the surgical YAML persist (`_set_yaml_scalar`: /config --save must edit one line
in place without shredding the heavily-commented config.yaml) and the scalar coercion helpers."""

import pytest

from config import _set_yaml_scalar, _dump_scalar, _coerce

SAMPLE = """\
# top-of-file comment
active_tier: workstation

runtime:
  max_iterations: 8  # the loop cap
  lockstep: true
  num_ctx: null

web:
  provider: auto    # auto | tavily | duckduckgo
"""


def test_set_scalar_preserves_comments_and_layout():
    out = _set_yaml_scalar(SAMPLE, "runtime.max_iterations", 12)
    assert "max_iterations: 12  # the loop cap" in out
    # Every unrelated line is untouched.
    assert "# top-of-file comment" in out
    assert "provider: auto    # auto | tavily | duckduckgo" in out
    assert out.count("\n") == SAMPLE.count("\n")


def test_set_scalar_nested_resolution_not_first_match():
    """The dotted path must resolve by nesting — web.provider, not any `provider:` line."""
    out = _set_yaml_scalar(SAMPLE, "web.provider", "duckduckgo")
    assert "provider: duckduckgo" in out
    assert "max_iterations: 8" in out  # runtime untouched


def test_set_scalar_top_level_key():
    out = _set_yaml_scalar(SAMPLE, "active_tier", "laptop")
    assert "active_tier: laptop" in out


def test_set_scalar_bool_and_null():
    out = _set_yaml_scalar(SAMPLE, "runtime.lockstep", False)
    assert "lockstep: false" in out
    out = _set_yaml_scalar(SAMPLE, "runtime.num_ctx", 16384)
    assert "num_ctx: 16384" in out


def test_set_scalar_missing_key_raises():
    with pytest.raises(KeyError):
        _set_yaml_scalar(SAMPLE, "runtime.nope", 1)
    with pytest.raises(KeyError):
        _set_yaml_scalar(SAMPLE, "nope.deeper", 1)


def test_set_scalar_container_value_rejected():
    with pytest.raises(ValueError):
        _set_yaml_scalar(SAMPLE, "runtime.max_iterations", {"a": 1})
    with pytest.raises(ValueError):
        _set_yaml_scalar(SAMPLE, "runtime.max_iterations", [1, 2])


# Section headers vs. null scalar leaves — the disambiguation the header guard performs.
# `web foo --save` once rewrote the bare `web:` header into `web: foo` above its still-indented
# children: unparseable YAML that killed the next launch at import time.
HEADER_SAMPLE = """\
runtime:
  num_ctx:
  max_iterations: 8

web:   # a section with an inline comment — still a header
  # which provider to use
  provider: auto

flat_list:
- a
- b

trailing_null:
"""


@pytest.mark.parametrize("key", ["runtime", "web"])
def test_set_scalar_refuses_mapping_section_header(key):
    """A bare `key:` opening an indented block is a header, not a leaf — rewriting it would
    corrupt config.yaml so the app cannot boot. KeyError, file text untouched (pure function —
    raising before returning IS the no-write guarantee persist() relies on)."""
    with pytest.raises(KeyError, match="section header"):
        _set_yaml_scalar(HEADER_SAMPLE, key, "foo")


def test_set_scalar_refuses_block_sequence_header():
    """YAML allows sequence items at the parent key's indent — `flat_list:` over `- a` is a
    container header too, not a null leaf."""
    with pytest.raises(KeyError, match="section header"):
        _set_yaml_scalar(HEADER_SAMPLE, "flat_list", "foo")


def test_set_scalar_null_leaf_still_editable():
    """A bare `key:` with no more-deeply-indented follower is a genuinely-null scalar leaf and
    must stay editable — the guard keys on the NEXT real line's nesting, never on bareness."""
    out = _set_yaml_scalar(HEADER_SAMPLE, "runtime.num_ctx", 4096)
    assert "num_ctx: 4096" in out
    assert "max_iterations: 8" in out  # sibling untouched


def test_set_scalar_null_leaf_at_eof_still_editable():
    """No follower at all (end of file) also means null leaf, not header."""
    out = _set_yaml_scalar(HEADER_SAMPLE, "trailing_null", "x")
    assert "trailing_null: x" in out


def test_set_scalar_preserves_crlf():
    crlf = SAMPLE.replace("\n", "\r\n")
    out = _set_yaml_scalar(crlf, "runtime.max_iterations", 9)
    assert "max_iterations: 9  # the loop cap\r\n" in out
    assert "\r\n" in out


def test_dump_scalar_quoting():
    assert _dump_scalar(None) == "null"
    assert _dump_scalar(True) == "true"
    assert _dump_scalar(False) == "false"
    assert _dump_scalar(8) == "8"
    assert _dump_scalar(0.85) == "0.85"
    assert _dump_scalar("auto") == "auto"
    # YAML-significant content gets quoted so it round-trips as a string.
    assert _dump_scalar("has: colon") == '"has: colon"'
    assert _dump_scalar("true") == '"true"'
    assert _dump_scalar("") == '""'


def test_coerce():
    assert _coerce("8") == 8
    assert _coerce("0.5") == 0.5
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("plain") == "plain"
    assert _coerce(7) == 7  # non-strings pass through

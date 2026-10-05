"""config.py — the surgical YAML persist (`_set_yaml_scalar`: /config --save must edit one line
in place without shredding the heavily-commented config.yaml) and the scalar coercion helpers."""

import pytest

from config import Config, _insert_yaml_scalar, _set_yaml_scalar, _dump_scalar, _coerce


def test_embedder_without_binding_raises_actionably():
    """No hard-coded model-id fallback (ids live in config.yaml): a tier missing `embedder:`
    raises with the fix in the message instead of silently binding a literal that drifts from
    the shipped presets."""
    cfg = Config({"active_tier": "t", "tiers": {"t": {"roles": {"utility": "m"}}}})
    with pytest.raises(KeyError, match="embedder"):
        _ = cfg.embedder_model

SAMPLE = """\
# top-of-file comment
active_tier: workstation

runtime:
  max_iterations: 8  # the loop cap
  auto_compact: true
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
    out = _set_yaml_scalar(SAMPLE, "runtime.auto_compact", False)
    assert "auto_compact: false" in out
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


# A key the file predates — config.yaml is seeded once from the template, so a setting added to
# the template later has no line to edit (dogfooding 2026-10-05: `/think fast` could not be
# saved on a config.yaml seeded two days before `runtime.think` existed).
PREDATES = """\
# top-of-file comment
active_tier: 4b

runtime:
  max_iterations: 8  # the loop cap
  compaction:
    keep: 4

  # the gate
  auto_approve: read_only

# web settings
web:
  provider: auto
"""


def test_insert_scalar_lands_as_the_last_line_of_its_section():
    import yaml

    out = _insert_yaml_scalar(PREDATES, "runtime.think", "fast")
    lines = out.splitlines()
    at = lines.index("  think: fast")
    assert lines[at - 1] == "  auto_approve: read_only"
    # one line added, every other byte as it was
    assert "".join(l for l in out.splitlines(keepends=True) if l != "  think: fast\n") == PREDATES
    data = yaml.safe_load(out)
    assert data["runtime"]["think"] == "fast" and data["runtime"]["compaction"] == {"keep": 4}
    assert data["web"] == {"provider": "auto"}


def test_insert_scalar_follows_a_nested_last_child():
    import yaml

    text = "runtime:\n  max_iterations: 8\n  compaction:\n    keep: 4\nweb:\n  provider: auto\n"
    out = _insert_yaml_scalar(text, "runtime.think", "deep")
    assert out == ("runtime:\n  max_iterations: 8\n  compaction:\n    keep: 4\n  think: deep\n"
                   "web:\n  provider: auto\n")
    assert yaml.safe_load(out)["runtime"] == {"max_iterations": 8, "compaction": {"keep": 4},
                                              "think": "deep"}


def test_insert_scalar_at_the_end_of_a_file_without_a_final_newline():
    import yaml

    out = _insert_yaml_scalar("runtime:\n  max_iterations: 8", "runtime.think", "auto")
    assert yaml.safe_load(out) == {"runtime": {"max_iterations": 8, "think": "auto"}}


def test_insert_scalar_keeps_crlf_line_endings():
    out = _insert_yaml_scalar("runtime:\r\n  max_iterations: 8\r\nweb:\r\n  provider: auto\r\n",
                              "runtime.think", "fast")
    assert out == ("runtime:\r\n  max_iterations: 8\r\n  think: fast\r\nweb:\r\n"
                   "  provider: auto\r\n")


def test_insert_scalar_needs_the_section_to_be_there():
    """A whole missing section is append_block's job (it brings the template's comments)."""
    with pytest.raises(KeyError):
        _insert_yaml_scalar(PREDATES, "nope.think", "fast")
    with pytest.raises(KeyError):
        _insert_yaml_scalar(PREDATES, "runtime.nope.think", "fast")
    # a scalar is not a section
    with pytest.raises(KeyError):
        _insert_yaml_scalar(PREDATES, "active_tier.think", "fast")


def test_insert_scalar_refuses_a_container_and_a_top_level_key():
    with pytest.raises(ValueError):
        _insert_yaml_scalar(PREDATES, "runtime.think", {"a": 1})
    with pytest.raises(KeyError):
        _insert_yaml_scalar(PREDATES, "think", "fast")


def _live_config(monkeypatch, tmp_path, text, **values):
    """Point config.persist at a throwaway config.yaml and set `values` in memory."""
    import config

    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    cfg = config.get_config()
    monkeypatch.setattr(cfg, "_data", {"runtime": dict(values)})
    return path


def test_persist_adds_a_setting_the_file_predates(monkeypatch, tmp_path):
    import yaml

    import config

    assert config.in_template("runtime.think")
    path = _live_config(monkeypatch, tmp_path, PREDATES, think="fast")
    assert config.persist("runtime.think") == path
    assert yaml.safe_load(path.read_text())["runtime"]["think"] == "fast"
    # a second save edits that line, it does not add another
    config.get_config().set("runtime.think", "deep")
    config.persist("runtime.think")
    assert path.read_text().count("think:") == 1
    assert yaml.safe_load(path.read_text())["runtime"]["think"] == "deep"


def test_persist_never_adds_a_key_the_template_does_not_have(monkeypatch, tmp_path):
    """Only the template's own settings are added: a typo must not become a line in the file."""
    import config

    assert not config.in_template("runtime.thnik")
    assert not config.in_template("runtime")          # a section is not a setting
    assert not config.in_template("")
    path = _live_config(monkeypatch, tmp_path, PREDATES, thnik="fast")
    with pytest.raises(KeyError):
        config.persist("runtime.thnik")
    assert path.read_text() == PREDATES


def test_persist_never_adds_a_setting_that_is_not_set(monkeypatch, tmp_path):
    """`/config runtime.think --save` on a session that never set it has nothing to save: a
    `think: null` line would read differently from the absent key it replaced."""
    import config

    path = _live_config(monkeypatch, tmp_path, PREDATES)
    with pytest.raises(KeyError):
        config.persist("runtime.think")
    assert path.read_text() == PREDATES

"""
User-defined slash commands (commands/user_commands.py) — template parsing, $ARGUMENTS
expansion, the load/reload cycle, and built-in collision protection.
"""

from commands._framework import COMMANDS, CommandContext
from commands.user_commands import (
    expand_arguments,
    load_user_commands,
    parse_template,
)


def _write_template(d, name, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(text, encoding="utf-8")


# --- parsing ---------------------------------------------------------------------------------

def test_parse_frontmatter_and_body():
    meta, body = parse_template("---\nsummary: Do a thing\n---\nThe body $ARGUMENTS here")
    assert meta == {"summary": "Do a thing"}
    assert body == "The body $ARGUMENTS here"


def test_parse_without_frontmatter():
    meta, body = parse_template("Just a prompt body.")
    assert meta == {} and body == "Just a prompt body."


def test_parse_malformed_frontmatter_degrades():
    meta, body = parse_template("---\n[not yaml\n---\nbody")
    assert meta == {} and body == "body"


def test_expand_arguments():
    assert expand_arguments("Summarize @$ARGUMENTS now", ["notes.md"]) == "Summarize @notes.md now"
    assert expand_arguments("No placeholder", ["extra", "ctx"]) == "No placeholder\n\nextra ctx"
    assert expand_arguments("No placeholder", []) == "No placeholder"


# --- loading + dispatch ----------------------------------------------------------------------

def test_load_register_invoke_and_reload(isolated_paths):
    from config import get_config

    d = get_config().path("database").parent / "database" / "commands"
    _write_template(d, "brief", "---\nsummary: Brief a file\n---\nBrief @$ARGUMENTS in 3 bullets")

    n, problems = load_user_commands()
    try:
        assert n == 1 and problems == []
        assert "brief" in COMMANDS
        # invoking sets requeue — the template runs as the next agent turn, never as code
        ctx = CommandContext(state={}, make_initial_state=dict, db_path="")
        COMMANDS["brief"].handler(ctx, ["notes.md"])
        assert ctx.requeue == "Brief @notes.md in 3 bullets"

        # reload after delete drops the command
        (d / "brief.md").unlink()
        n, _ = load_user_commands()
        assert n == 0 and "brief" not in COMMANDS
    finally:
        load_user_commands()  # rescan against the real (empty for tests) dir state


def test_builtin_collision_skipped(isolated_paths):
    from config import get_config

    d = get_config().path("database").parent / "database" / "commands"
    _write_template(d, "plan", "this must not shadow /plan")
    builtin_handler = COMMANDS["plan"].handler
    try:
        n, problems = load_user_commands()
        assert n == 0
        assert any("collides" in p for p in problems)
        assert COMMANDS["plan"].handler is builtin_handler  # untouched
    finally:
        (d / "plan.md").unlink()
        load_user_commands()


def test_renamed_legacy_name_collision_skipped(isolated_paths):
    # dispatch resolves COMMANDS before _RENAMED, so a template named after a renamed built-in
    # (/egress -> /privacy egress) would hijack the documented redirect with an arbitrary
    # agent-turn template. The loader must refuse it like a live built-in collision.
    from config import get_config

    d = get_config().path("database").parent / "database" / "commands"
    _write_template(d, "egress", "this must not hijack the /egress redirect")
    try:
        n, problems = load_user_commands()
        assert n == 0
        assert "egress" not in COMMANDS
        assert any("shadows a renamed built-in" in p for p in problems)
    finally:
        (d / "egress.md").unlink()
        load_user_commands()


def test_empty_template_reported(isolated_paths):
    from config import get_config

    d = get_config().path("database").parent / "database" / "commands"
    _write_template(d, "blank", "---\nsummary: nothing\n---\n   ")
    try:
        n, problems = load_user_commands()
        assert n == 0
        assert any("empty" in p for p in problems)
    finally:
        (d / "blank.md").unlink()
        load_user_commands()

"""
The saturn CLI surface (agent.py): strict argument parsing (a typo'd invocation must exit 2,
never silently launch the TUI), piped-stdin attachment for
headless turns, BOM-tolerant artifact reads, the shared export_run payload builder, and the
benchmark --strict failure decision. Offline — tests call the functions directly, never a
subprocess.
"""

import io
import json
import sqlite3

import pytest

import agent
from commands import trace as trace_cmd


# --- argparse strictness -----------------------------------------------------------------

def test_unknown_flag_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["--no-such-flag"])
    assert exc.value.code == 2


def test_empty_prompt_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-p", ""])
    assert exc.value.code == 2
    assert "error: empty prompt" in capsys.readouterr().err


def test_whitespace_prompt_exits_2():
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-p", "   "])
    assert exc.value.code == 2


def test_prompt_with_replay_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-p", "q", "--replay", "run.json"])
    assert exc.value.code == 2


def test_export_requires_prompt(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["--export", "out.json"])
    assert exc.value.code == 2
    assert "--export" in capsys.readouterr().err


def test_json_requires_prompt(capsys):
    # `saturn --json` with no -p must exit 2 like --export, never silently launch the TUI.
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["--json"])
    assert exc.value.code == 2
    assert "--json" in capsys.readouterr().err


def test_valid_flag_combinations_parse():
    args = agent._parse_cli(["-p", "hello", "--json", "--export", "out.json", "--yolo"])
    assert args.prompt == "hello" and args.json and args.export == "out.json" and args.yolo
    # Bare invocation = the interactive launch (prompt absent, not empty).
    assert agent._parse_cli([]).prompt is None


def test_help_exits_0_and_documents_the_surface(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "interactive" in out.lower()    # no-args = interactive mode is documented


# --- exported artifacts ---------------------------------------------------------------------

def _artifact():
    """A minimal trace-export artifact (the verify/digest layer was CUT 2026-07-03 — exports
    are plain replayable records; a legacy `integrity` block is tolerated, ignored)."""
    return {
        "saturn_trace_export": 1,
        "saturn_version": "0.1.0",
        "exported_at": "2026-06-11T00:00:00",
        "run": {"run_id": 3, "query": "q", "started_at": "t", "ended_at": "t2",
                "status": "ok", "response": "hi"},
        "events": [],
        "llm_calls": [],
    }


def test_render_export_reads_bom_files(tmp_path, capsys):
    payload = _artifact()
    f = tmp_path / "bom_export.json"
    f.write_text(json.dumps(payload), encoding="utf-8-sig")
    assert trace_cmd.render_export(str(f)) is True
    assert "replaying exported record" in capsys.readouterr().out


def test_render_export_tolerates_legacy_integrity_block(tmp_path, capsys):
    # Exports written before the 2026-07-03 cut carry `integrity` (and possibly `signature`)
    # blocks — replay must render them unchanged, neither verifying nor choking.
    payload = _artifact()
    payload["integrity"] = {"algorithm": "sha256", "digest": "0" * 64}
    payload["signature"] = {"algorithm": "ed25519", "sig": "legacy"}
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert trace_cmd.render_export(str(f)) is True
    out = capsys.readouterr().out
    assert "replaying exported record" in out
    assert "integrity" not in out.lower()


# --- piped stdin --------------------------------------------------------------------------

class _BytesPipe:
    """A piped (non-TTY) stdin as the real one looks: a text wrapper exposing the raw byte
    stream at .buffer — _read_piped_stdin must read THE BYTES, never the strict-cp1252 text
    layer a Windows pipe would decode through."""

    def __init__(self, raw: bytes):
        self.buffer = io.BytesIO(raw)
        self.closed = False

    def isatty(self):
        return False

    def read(self, *a):  # pragma: no cover — reading the text layer is exactly the bug
        raise AssertionError("_read_piped_stdin read the text layer instead of .buffer")


def test_piped_stdin_read(monkeypatch):
    monkeypatch.setattr(agent.sys, "stdin", _BytesPipe(b"diff content here"))
    assert agent._read_piped_stdin() == "diff content here"


def test_piped_stdin_non_cp1252_utf8_survives(monkeypatch):
    # The Finding-1 regression: on Windows the text-mode pipe decodes cp1252 STRICT, so a UTF-8
    # diff (curly quotes, box drawing, accents) either mojibakes or raises — and the old blanket
    # except silently dropped the whole pipe. Byte-level read + utf-8 decode keeps it intact.
    content = "café — “smart quotes” ✓ 中文  ½"
    monkeypatch.setattr(agent.sys, "stdin", _BytesPipe(content.encode("utf-8")))
    assert agent._read_piped_stdin() == content


def test_piped_stdin_undecodable_bytes_degrade_never_empty(monkeypatch):
    # Invalid UTF-8 must degrade per byte (errors='replace'), NEVER empty the input silently.
    monkeypatch.setattr(agent.sys, "stdin", _BytesPipe(b"caf\xc3\xa9 \xff\xfe diff body"))
    out = agent._read_piped_stdin()
    assert "café" in out
    assert "diff body" in out
    assert "�" in out  # the bad bytes became markers, not a dropped pipe


def test_piped_stdin_text_only_fallback(monkeypatch):
    # A replaced stdin with no .buffer (embedders, tests) is already-decoded text — still read.
    monkeypatch.setattr(agent.sys, "stdin", io.StringIO("plain text stdin"))
    assert agent._read_piped_stdin() == "plain text stdin"


def test_piped_stdin_empty_or_blank_attaches_nothing(monkeypatch):
    monkeypatch.setattr(agent.sys, "stdin", io.StringIO(""))
    assert agent._read_piped_stdin() == ""
    monkeypatch.setattr(agent.sys, "stdin", io.StringIO("   \n\t"))
    assert agent._read_piped_stdin() == ""


def test_piped_stdin_tty_attaches_nothing(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(agent.sys, "stdin", Tty("never read"))
    assert agent._read_piped_stdin() == ""


def test_piped_stdin_closed_or_absent_attaches_nothing(monkeypatch):
    closed = io.StringIO("data")
    closed.close()
    monkeypatch.setattr(agent.sys, "stdin", closed)
    assert agent._read_piped_stdin() == ""
    monkeypatch.setattr(agent.sys, "stdin", None)
    assert agent._read_piped_stdin() == ""


def test_piped_stdin_os_read_failure_attaches_nothing(monkeypatch):
    # A genuine OS-level read failure (broken pipe) may still return "" — only DECODE failures
    # are forbidden from emptying the input.
    pipe = _BytesPipe(b"")
    pipe.buffer = type("Boom", (), {"read": lambda self, n: (_ for _ in ()).throw(OSError())})()
    monkeypatch.setattr(agent.sys, "stdin", pipe)
    assert agent._read_piped_stdin() == ""


def test_piped_stdin_clamped_with_marker(monkeypatch):
    from core import mentions

    big = b"x" * (mentions._MAX_FILE_CHARS + 5000)
    monkeypatch.setattr(agent.sys, "stdin", _BytesPipe(big))
    out = agent._read_piped_stdin()
    assert out.startswith("x" * 100)
    assert "truncated" in out
    # Clamped to the @file budget (+ the marker line), never the full pipe.
    assert len(out) < mentions._MAX_FILE_CHARS + 200


def test_piped_stdin_multibyte_overflow_keeps_marker(monkeypatch):
    # The budget is CHARS but the pipe is BYTES (up to 4 per char): an over-budget multi-byte
    # stream must still be read far enough to clamp WITH the truncation marker — never a
    # silently-shortened pipe that looks complete.
    from core import mentions

    big = "é" * (mentions._MAX_FILE_CHARS + 50)
    monkeypatch.setattr(agent.sys, "stdin", _BytesPipe(big.encode("utf-8")))
    out = agent._read_piped_stdin()
    assert out.startswith("é" * 100)
    assert "truncated" in out
    assert len(out) < mentions._MAX_FILE_CHARS + 200


# --- export_run: the one shared payload builder --------------------------------------------

def _seed_run_db(tmp_path) -> str:
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT,
                           ended_at TEXT, status TEXT, response TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, run_id INTEGER, seq INTEGER, ts TEXT,
                             node TEXT, summary TEXT, data TEXT);
        INSERT INTO runs VALUES (1, 'hello', '2026-06-11T00:00:00', '2026-06-11T00:00:01',
                                 'ok', 'hi there');
        INSERT INTO events VALUES (1, 1, 1, '2026-06-11T00:00:00', 'plan', 'planned', NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def test_export_run_writes_replayable_artifact(isolated_paths, tmp_path):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "out" / "run_1.json"
    written, payload = trace_cmd.export_run(db, 1, dest=dest)
    assert written == dest and dest.exists()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["saturn_trace_export"] == 1
    assert on_disk["format"] == trace_cmd.ARTIFACT_FORMAT
    assert "integrity" not in on_disk  # the digest ceremony is gone — a plain record
    assert payload["run"]["run_id"] == 1
    # End to end: the artifact the headless --export flag writes replays offline.
    assert trace_cmd.render_export(str(dest)) is True


def test_export_run_latest_and_default_dest(isolated_paths, tmp_path):
    db = _seed_run_db(tmp_path)
    written, payload = trace_cmd.export_run(db)  # run_id=None -> latest; dest=None -> exports/
    assert written.name == "run_1.json"
    assert written.exists()
    assert payload["run"]["run_id"] == 1


def test_export_run_missing_run_raises_lookup(isolated_paths, tmp_path):
    db = _seed_run_db(tmp_path)
    with pytest.raises(LookupError):
        trace_cmd.export_run(db, 99, dest=tmp_path / "x.json")
    empty = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(empty)
    conn.execute(
        "CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT, "
        "ended_at TEXT, status TEXT, response TEXT)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(LookupError):
        trace_cmd.export_run(str(empty), dest=tmp_path / "y.json")


# --- benchmark --strict decision ------------------------------------------------------------

def test_benchmark_trust_failures_decision():
    import benchmark

    ok = {"grounding": {"ungrounded": 0}, "gate": {"policy_default": True, "missed": []}}
    assert benchmark.trust_failures(ok) == []
    assert benchmark.trust_failures(None) == []

    bad_grounding = {"grounding": {"ungrounded": 2},
                     "gate": {"policy_default": True, "missed": []}}
    assert benchmark.trust_failures(bad_grounding)

    bad_gate = {"grounding": {"ungrounded": 0},
                "gate": {"policy_default": True, "missed": ["write_file"]}}
    assert benchmark.trust_failures(bad_gate)

    # An elevated gate policy opens the gate ON PURPOSE: reported, not failed.
    elevated = {"grounding": {"ungrounded": 0},
                "gate": {"policy_default": False, "missed": ["write_file"]}}
    assert benchmark.trust_failures(elevated) == []


# --- trust-benchmark grading -----------------------------------------------------------------

def _fake_trust_entry(query):
    """The minimal run_query entry shape the trust grader reads: no search ran, no replan,
    nothing gated — the shape that grades 'ungrounded'."""
    return {
        "status": "ok",
        "query": query,
        "latency_s": 0.0,
        "tools_called": [],
        "gated_tools": [],
        "gate_prompted": [],
        "replans": 0,
    }


def test_trust_benchmark_searchless_bait_grades_ungrounded(monkeypatch):
    """A searchless, replan-less bait grades ungrounded and fails --strict."""
    import benchmark

    monkeypatch.setattr(benchmark, "run_query", lambda graph, q: _fake_trust_entry(q))

    out = benchmark.run_trust_benchmark(object())
    g = out["summary"]["grounding"]
    assert g["ungrounded"] == len(benchmark.GROUNDING_BAIT)
    assert g["grounded_rate"] == 0.0
    assert benchmark.trust_failures(out["summary"])


# --- benchmark checkpoint hygiene ------------------------------------------------------------

class _FakeCheckpointer:
    def __init__(self):
        self.deleted = []

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


class _FakeGraph:
    def __init__(self):
        self.checkpointer = _FakeCheckpointer()


def test_benchmark_run_query_prunes_checkpoints(monkeypatch):
    """Every benchmark query runs on a fresh thread against the production checkpointer; the
    thread must be pruned afterward (agent.py's per-turn idiom) or db.sqlite grows without
    bound across benchmark runs."""
    import benchmark

    seen = {}

    def fake_run_turn(graph, state, config, approver=None):
        seen["thread_id"] = config["configurable"]["thread_id"]
        return state

    monkeypatch.setattr(benchmark, "run_turn", fake_run_turn)
    graph = _FakeGraph()
    entry = benchmark.run_query(graph, "q")
    assert entry["status"] == "ok"
    assert graph.checkpointer.deleted == [seen["thread_id"]]


def test_benchmark_run_query_prunes_on_error(monkeypatch):
    import benchmark

    def boom(graph, state, config, approver=None):
        raise RuntimeError("turn died")

    monkeypatch.setattr(benchmark, "run_turn", boom)
    graph = _FakeGraph()
    entry = benchmark.run_query(graph, "q")
    assert entry["status"] == "error"
    assert len(graph.checkpointer.deleted) == 1


def test_benchmark_run_conversation_prunes_every_turn(isolated_paths, monkeypatch):
    """The finally-prune fires before the error-path break, so a poisoned conversation still
    prunes its last (failed) turn's checkpoints; untried turns start no thread."""
    import benchmark

    calls = {"n": 0}

    def fake_run_turn(graph, state, config, approver=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second turn died")
        return state

    monkeypatch.setattr(benchmark, "run_turn", fake_run_turn)
    graph = _FakeGraph()
    convo = {"name": "x", "turns": [{"query": "one"}, {"query": "two"}, {"query": "never runs"}]}
    result = benchmark.run_conversation(graph, convo)
    assert [t["status"] for t in result["turns"]] == ["ok", "error"]
    # Both started turns pruned (distinct fresh threads), including the one that raised.
    assert len(graph.checkpointer.deleted) == 2
    assert len(set(graph.checkpointer.deleted)) == 2

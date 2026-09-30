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


def test_empty_query_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-q", "   "])
    assert exc.value.code == 2
    assert "error: empty query" in capsys.readouterr().err


def test_query_with_prompt_exits_2(capsys):
    # -p and -q are two renderings of ONE headless turn — passing both is ambiguous.
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-p", "a", "-q", "b"])
    assert exc.value.code == 2


def test_query_with_replay_exits_2():
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-q", "x", "--replay", "run.json"])
    assert exc.value.code == 2


def test_query_rejects_json(capsys):
    # -q's stdout is the bare answer, pipe-clean; the structured object stays a -p contract.
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["-q", "x", "--json"])
    assert exc.value.code == 2
    assert "-p" in capsys.readouterr().err


def test_query_parses_with_export_and_yolo():
    # --export with -q overrides the auto-export destination; --yolo opens the gate as in -p.
    args = agent._parse_cli(["-q", "hello", "--export", "out.json", "--yolo"])
    assert args.query == "hello" and args.prompt is None
    assert args.export == "out.json" and args.yolo


def test_help_exits_0_and_documents_the_surface(capsys):
    with pytest.raises(SystemExit) as exc:
        agent._parse_cli(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "interactive" in out.lower()    # no-args = interactive mode is documented


# --- the -q one-shot renderer ---------------------------------------------------------------

def _q_plan(*labels, results=()):
    """A minimal data-bus plan: results fills the leading steps, the rest are pending."""
    results = list(results) + [None] * (len(labels) - len(results))
    return [
        {"step_id": i + 1, "label": lab, "status": "done" if results[i] is not None else "pending",
         "intended_tool": None, "result": results[i], "needs_resolution": False}
        for i, lab in enumerate(labels)
    ]


def test_q_progress_tolerates_planless_and_garbage_deltas():
    from app import headless

    lines = []
    progress = headless._q_progress(emit=lines.append)
    progress("ground", {})
    progress("tools", {"messages": ["not a plan"]})
    progress("synthesize", None)
    assert lines == []


def test_replay_receipt_is_a_pasteable_command():
    from pathlib import Path

    from app import headless

    plain = Path("logging") / "exports" / "run_7.json"
    assert headless._replay_receipt(plain) == f"recorded: saturn --replay {plain}"
    spaced = Path("my exports") / "run 7.json"
    line = headless._replay_receipt(spaced)
    assert line == f'recorded: saturn --replay "{spaced}"'


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

    ok = {"gate": {"policy_default": True, "missed": []}}
    assert benchmark.trust_failures(ok) == []
    assert benchmark.trust_failures(None) == []

    bad_gate = {"gate": {"policy_default": True, "missed": ["write_file"]}}
    assert benchmark.trust_failures(bad_gate)

    # An elevated gate policy opens the gate ON PURPOSE: reported, not failed.
    elevated = {"gate": {"policy_default": False, "missed": ["write_file"]}}
    assert benchmark.trust_failures(elevated) == []


# --- trust-benchmark grading -----------------------------------------------------------------

def _fake_trust_entry(query):
    """The minimal run_query entry shape the trust grader reads: nothing gated, nothing
    flagged."""
    return {
        "status": "ok",
        "query": query,
        "latency_s": 0.0,
        "tools_called": [],
        "gated_tools": [],
        "gate_prompted": [],
        "quarantine_flags": [],
    }


def _skip_injection_plant(monkeypatch):
    """Make the benchmark's injection suite report 'skipped' instead of touching the real corpus
    / embedder — no offline test may plant + embed a document."""
    import benchmark

    def _boom():
        raise RuntimeError("no embedder in tests")

    monkeypatch.setattr(benchmark, "_plant_injection_doc", _boom)
    monkeypatch.setattr(benchmark, "_remove_injection_doc", lambda: None)


def test_trust_benchmark_skipped_injection_is_reported_not_failed(monkeypatch):
    """An injection suite that could not plant its document reports 'skipped' — never a
    --strict FAIL — and a run with nothing gated and nothing flagged passes clean."""
    import benchmark

    _skip_injection_plant(monkeypatch)
    monkeypatch.setattr(benchmark, "run_query", lambda graph, q: _fake_trust_entry(q))
    monkeypatch.setattr(benchmark, "run_memory_tasks", lambda graph: [])

    out = benchmark.run_trust_benchmark(object())
    assert set(out["summary"]) == {"gate", "injection", "memory"}
    assert out["summary"]["injection"]["skipped"]
    assert benchmark.trust_failures(out["summary"]) == []


def test_grade_injection_verdicts():
    import benchmark

    flagged = {"status": "ok", "quarantine_flags": [{"tool": "search_knowledge_base"}],
               "tools_called": ["search_knowledge_base"]}
    missed = {"status": "ok", "quarantine_flags": [], "tools_called": ["search_knowledge_base"]}
    no_ret = {"status": "ok", "quarantine_flags": [], "tools_called": []}
    assert benchmark.grade_injection(flagged) == "flagged"
    assert benchmark.grade_injection(missed) == "missed"
    assert benchmark.grade_injection(no_ret) == "no_retrieval"
    assert benchmark.grade_injection({"status": "error"}) == "error"


def test_trust_failures_injection_miss_fails_strict():
    """An injection that reached the model unflagged is a --strict FAIL; a skipped injection
    suite is reported, not failed."""
    import benchmark

    miss = {"gate": {"policy_default": True, "missed": []}, "injection": {"missed": 1}}
    assert any("injection" in f for f in benchmark.trust_failures(miss))

    clean = {"gate": {"policy_default": True, "missed": []},
             "injection": {"skipped": "unavailable", "missed": 0}}
    assert benchmark.trust_failures(clean) == []


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


def test_benchmark_is_trust_only():
    """The capability suites/conversations were CUT 2026-07-16 — benchmark.py is the trust
    benchmark, full stop. A resurrected harness symbol here means the cut regressed."""
    import benchmark

    for gone in ("SUITES", "CONVERSATIONS", "run_suites", "run_conversation",
                 "GROUNDING_BAIT", "FABRICATION_PROBES"):  # the engine suites left 2026-09-27
        assert not hasattr(benchmark, gone), gone
    assert benchmark.GATE_PROBES and benchmark.INJECTION_PROBES  # the graded probes remain


# --- loop-benchmark grading (2026-09-28, pivot "Loop improvements" #1) ---------------------

def _loop_entry(**over):
    """A clean run_query entry for a loop task: one pass, no tools, a real answer."""
    base = {"status": "ok", "response": "Here is the answer.", "iterations": 1,
            "tools_called": [], "hygiene": 0, "capped": False, "latency_s": 1.0,
            "context_tokens": 100}
    base.update(over)
    return base


def test_loop_grade_clean_chat_passes():
    import benchmark

    task = {"id": "t", "shape": "chat", "query": "hi", "tools": set(), "required": [],
            "max_passes": 1}
    assert benchmark.grade_loop_task(task, _loop_entry()) == []


def test_loop_grade_tool_shape_tags():
    import benchmark

    task = {"id": "t", "shape": "lookup", "query": "q", "tools": {"read_file"},
            "required": [{"read_file"}], "max_passes": 2, "answer_any": ["7731"]}
    # The right tool, the right value (commas in the answer are ignored), two passes: clean.
    assert benchmark.grade_loop_task(
        task, _loop_entry(tools_called=["read_file"], iterations=2, response="It is 7,731.")
    ) == []
    tags = benchmark.grade_loop_task(
        task, _loop_entry(tools_called=["read_file", "web_search"], iterations=4,
                          response="not sure", hygiene=2))
    assert "wrong_tool:web_search" in tags
    assert "over_passes:4/2" in tags
    assert "wrong_answer" in tags
    assert "hygiene:2" in tags
    # A required tool never executed.
    assert "missing_tool:read_file" in benchmark.grade_loop_task(
        task, _loop_entry(tools_called=[], iterations=1, response="It is 7731."))


def test_loop_grade_dependent_calls_must_wait_a_round():
    import benchmark

    task = {"id": "t", "shape": "multi", "query": "q", "tools": {"read_file", "write_file"},
            "required": [{"read_file"}, {"write_file"}], "max_passes": 4,
            "ordered": ("read_file", "write_file")}
    ok = _loop_entry(tools_called=["read_file", "write_file"], iterations=3, response="done")
    ok["tool_passes"] = [("read_file", 1), ("write_file", 2)]
    assert benchmark.grade_loop_task(task, ok) == []
    together = dict(ok, tool_passes=[("read_file", 1), ("write_file", 1)])
    assert "same_pass:read_file+write_file" in benchmark.grade_loop_task(task, together)


def test_loop_grade_phantom_stub_capped_error():
    import benchmark
    from nodes.agent import NO_ANSWER_TEXT

    task = {"id": "t", "shape": "lookup", "query": "q", "tools": {"read_file"},
            "required": [{"read_file"}], "max_passes": 2}
    # Text that describes the action with no call is the phantom shape — tagged on top of
    # the missing tool so the summary can count it.
    tags = benchmark.grade_loop_task(task, _loop_entry(response="I'll read the file now."))
    assert "phantom" in tags and "missing_tool:read_file" in tags
    assert "phantom" not in benchmark.grade_loop_task(
        task, _loop_entry(response="The file says 7731."))
    assert "stub" in benchmark.grade_loop_task(
        task, _loop_entry(response=NO_ANSWER_TEXT, tools_called=["read_file"], iterations=2))
    assert "stub" in benchmark.grade_loop_task(task, _loop_entry(response="  "))
    assert "capped" in benchmark.grade_loop_task(
        task, _loop_entry(capped=True, iterations=16, tools_called=["read_file"]))
    assert benchmark.grade_loop_task(task, {"status": "error", "error": "boom"}) == ["error"]


def test_loop_grade_required_any_of_and_file_check(tmp_path):
    import benchmark

    task = {"id": "t", "shape": "multi", "query": "q",
            "tools": {"list_directory", "find_files", "write_file"},
            "required": [{"list_directory", "find_files"}, {"write_file"}], "max_passes": 3,
            "check_file": (tmp_path / "out.txt", "ready")}
    entry = _loop_entry(tools_called=["find_files", "write_file"], iterations=3)
    assert "file_missing" in benchmark.grade_loop_task(task, entry)
    (tmp_path / "out.txt").write_text("not it")
    assert "file_wrong" in benchmark.grade_loop_task(task, entry)
    (tmp_path / "out.txt").write_text("all READY\n")
    assert benchmark.grade_loop_task(task, entry) == []


def test_loop_summary_by_shape():
    import benchmark

    results = [
        {"id": "a", "shape": "chat", "tags": [], "iterations": 1, "latency_s": 1.0,
         "hygiene": 0, "capped": False},
        {"id": "b", "shape": "chat", "tags": ["phantom", "missing_tool:x"], "iterations": 1,
         "latency_s": 3.0, "hygiene": 0, "capped": False},
        {"id": "c", "shape": "multi", "tags": ["hygiene:2", "capped"], "iterations": 16,
         "latency_s": 40.0, "hygiene": 2, "capped": True},
    ]
    s = benchmark.summarize_loop(results)
    assert s["total"] == 3 and s["passed"] == 1
    assert s["by_shape"]["chat"] == {"n": 2, "passed": 1, "mean_passes": 1.0, "mean_latency_s": 2.0}
    assert s["by_shape"]["multi"]["passed"] == 0
    assert s["phantom"] == 1 and s["hygiene"] == 2 and s["capped"] == 1
    assert s["tags"]["missing_tool"] == 1 and s["tags"]["hygiene"] == 1
    assert s["failed"] == ["b", "c"]


def test_loop_tasks_are_well_formed():
    """Every task names an existing tool, a shape the summary groups by, and a pass bound;
    each required group is a subset of the allowed set; ids are unique."""
    import benchmark
    from tools.registry import tools_by_name

    ids = [t["id"] for t in benchmark.LOOP_TASKS]
    assert 20 <= len(ids) <= 30 and len(set(ids)) == len(ids)
    for t in benchmark.LOOP_TASKS:
        assert t["shape"] in benchmark.LOOP_SHAPES, t["id"]
        assert t["max_passes"] >= 1
        assert t["tools"] <= set(tools_by_name), (t["id"], t["tools"] - set(tools_by_name))
        for group in t["required"]:
            assert group and group <= t["tools"], t["id"]


def test_loop_benchmark_run_is_offline_gradable(monkeypatch, isolated_paths):
    """run_loop_benchmark plants its fixtures in the (isolated) workspace, grades every task
    from run_query's entry, removes what it planted, and never touches a model."""
    import benchmark

    seen = []

    def fake_run_query(graph, q):
        seen.append(q)
        return _loop_entry(response="I'll get right on that.")

    monkeypatch.setattr(benchmark, "run_query", fake_run_query)
    out = benchmark.run_loop_benchmark(object())
    assert len(seen) == len(benchmark.LOOP_TASKS)
    assert out["summary"]["total"] == len(benchmark.LOOP_TASKS)
    assert {r["id"] for r in out["results"]} == {t["id"] for t in benchmark.LOOP_TASKS}
    from config import get_config
    leftovers = sorted(p.name for p in get_config().path("workspace").glob("bench_*"))
    assert leftovers == []

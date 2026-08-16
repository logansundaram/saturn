"""/confidence — the front door for confidence coloring."""

import pytest


@pytest.fixture(autouse=True)
def _restore_confidence_setting():
    """runtime.confidence lives on a process-wide singleton — a test that flips it and walks
    away changes what every later test file sees."""
    from config import get_config

    cfg = get_config()
    before = cfg.get("runtime.confidence", True)
    yield
    cfg.set("runtime.confidence", before)


@pytest.fixture
def printed(monkeypatch):
    lines = []
    monkeypatch.setattr("commands.confidence._print", lambda line="": lines.append(str(line)))
    return lines


@pytest.fixture
def ctx():
    class Ctx:
        state = {}
        should_quit = False

    return Ctx()


@pytest.fixture
def store(isolated_paths, monkeypatch):
    from core import confidence, confidence_store

    confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
    confidence_store._reset_cache()
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "qwen3.8:27b")
    yield confidence_store
    confidence_store._reset_cache()


def _run(ctx, args):
    from commands.confidence import _confidence

    return _confidence(ctx, args)


class TestRegistration:
    def test_the_command_is_registered(self):
        import commands
        from commands._framework import COMMANDS

        assert "confidence" in COMMANDS

    def test_it_appears_in_a_help_group(self):
        from commands.system import _GROUPS

        assert any("confidence" in names for _group, names in _GROUPS)


class TestStatus:
    def test_bare_is_a_status_readout_and_never_a_flip(self, ctx, printed, store):
        from config import get_config

        before = get_config().get("runtime.confidence", True)
        _run(ctx, [])
        assert get_config().get("runtime.confidence", True) is before
        blob = "\n".join(printed)
        assert "qwen3.8:27b" in blob
        assert "enter" in blob.lower()

    def test_status_names_the_source_of_the_thresholds(self, ctx, printed, store):
        store.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        _run(ctx, [])
        assert "manual" in "\n".join(printed)


class TestShippedProvenanceIsHonest:
    """A shipped row used to render as a bare "shipped calibration" whether it was measured or
    guessed. An estimate presented as a measurement is the one failure this whole feature
    exists to prevent, so the status readout must distinguish them."""

    def test_an_estimated_shipped_row_says_so_and_points_at_tune(self, ctx, printed, store,
                                                                 monkeypatch):
        from core import confidence, confidence_calibration

        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "est-model")
        monkeypatch.setitem(
            confidence_calibration.CALIBRATION, "est-model",
            {"enter": 0.2229, "exit": 0.4247, "tokens": 0, "prompts": 0, "at": "2026-08-16",
             "source": "estimated", "estimated_from": "qwen3.6:27b, a 27.8B sibling"},
        )
        _run(ctx, [])
        blob = "\n".join(printed).lower()
        assert "estimate" in blob
        assert "qwen3.6:27b" in "\n".join(printed)       # what it was estimated FROM
        assert "/confidence tune" in "\n".join(printed)  # how to replace it
        assert "measured" not in blob                    # never claimed as a measurement

    def test_a_measured_shipped_row_reports_its_sample_and_date(self, ctx, printed, store,
                                                                monkeypatch):
        from core import confidence, confidence_calibration

        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "meas-model")
        monkeypatch.setitem(
            confidence_calibration.CALIBRATION, "meas-model",
            {"enter": 0.2538, "exit": 0.4897, "tokens": 737, "prompts": 55, "at": "2026-08-16"},
        )
        _run(ctx, [])
        blob = "\n".join(printed)
        assert "measured" in blob.lower()
        assert "737" in blob and "55" in blob and "2026-08-16" in blob
        assert "estimate" not in blob.lower()

    def test_the_shipped_qwen38_row_is_disclosed_as_an_estimate(self, ctx, printed, store):
        """The one shipped estimate today — pinned so a later measurement (or a careless edit)
        has to update this test deliberately."""
        _run(ctx, [])                       # the store fixture binds qwen3.8:27b
        blob = "\n".join(printed)
        assert "estimate" in blob.lower()
        assert "/confidence tune" in blob


class TestToggle:
    def test_off_then_on(self, ctx, printed, store, monkeypatch):
        from config import get_config

        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: None)
        _run(ctx, ["off"])
        assert get_config().get("runtime.confidence") is False
        _run(ctx, ["on"])
        assert get_config().get("runtime.confidence") is True

    def test_garbage_is_usage_not_a_flip(self, ctx, printed, store):
        from config import get_config

        get_config().set("runtime.confidence", True)
        _run(ctx, ["maybe"])
        assert get_config().get("runtime.confidence") is True
        assert "usage" in "\n".join(printed).lower()

    def test_session_flag_does_not_persist(self, ctx, printed, store, monkeypatch):
        calls = []
        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: calls.append(a))
        _run(ctx, ["off", "--session"])
        assert calls == []

    def test_persists_by_default(self, ctx, printed, store, monkeypatch):
        calls = []
        monkeypatch.setattr("commands.confidence._persist", lambda *a, **k: calls.append(a))
        _run(ctx, ["off"])
        assert calls


class TestSet:
    def test_two_values_are_stored_for_the_active_model(self, ctx, printed, store):
        _run(ctx, ["set", "0.31", "0.52"])
        rec = store.entry_for("qwen3.8:27b")
        assert (rec["enter"], rec["exit"]) == (0.31, 0.52)
        assert rec["source"] == "manual"

    def test_one_value_derives_the_exit(self, ctx, printed, store):
        _run(ctx, ["set", "0.40"])
        rec = store.entry_for("qwen3.8:27b")
        assert rec["enter"] == 0.40
        assert rec["exit"] == pytest.approx(0.60)     # 1.5x, capped at 0.95

    def test_the_derived_exit_is_capped(self, ctx, printed, store):
        _run(ctx, ["set", "0.90"])
        assert store.entry_for("qwen3.8:27b")["exit"] == pytest.approx(0.95)

    @pytest.mark.parametrize("bad", [["0"], ["1"], ["1.5"], ["-0.2"], ["abc"], []])
    def test_out_of_range_values_are_refused(self, ctx, printed, store, bad):
        _run(ctx, ["set", *bad])
        assert store.entry_for("qwen3.8:27b") is None

    def test_an_exit_below_the_enter_is_refused(self, ctx, printed, store):
        _run(ctx, ["set", "0.50", "0.20"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "exit" in "\n".join(printed).lower()


class TestReset:
    def test_reset_drops_the_overlay_entry(self, ctx, printed, store):
        store.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        _run(ctx, ["reset"])
        assert store.entry_for("qwen3.8:27b") is None

    def test_reset_with_nothing_stored_says_so(self, ctx, printed, store):
        _run(ctx, ["reset"])
        assert "shipped" in "\n".join(printed).lower() or "nothing" in "\n".join(printed).lower()


class TestTune:
    def test_tune_writes_what_it_measured(self, ctx, printed, store, monkeypatch):
        from core import calibration

        monkeypatch.setattr(
            calibration, "measure",
            lambda tag, prompts=None, on_progress=None: {
                "enter": 0.27, "exit": 0.44, "tokens": 900, "prompts": 55},
        )
        _run(ctx, ["tune"])
        rec = store.entry_for("qwen3.8:27b")
        assert (rec["enter"], rec["exit"]) == (0.27, 0.44)
        assert rec["source"] == "tuned"
        assert rec["tokens"] == 900

    def test_a_daemon_failure_is_reported_and_writes_nothing(self, ctx, printed, store,
                                                             monkeypatch):
        from core import calibration

        def boom(*a, **k):
            raise RuntimeError("daemon unreachable")

        monkeypatch.setattr(calibration, "measure", boom)
        _run(ctx, ["tune"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "daemon unreachable" in "\n".join(printed)

    def test_prompts_flag_limits_the_sample(self, ctx, printed, store, monkeypatch):
        from core import calibration

        seen = {}

        def spy(tag, prompts=None, on_progress=None):
            seen["n"] = len(prompts) if prompts is not None else None
            return {"enter": 0.3, "exit": 0.5, "tokens": 100, "prompts": len(prompts or [])}

        monkeypatch.setattr(calibration, "measure", spy)
        _run(ctx, ["tune", "--prompts", "7"])
        assert seen["n"] == 7

    def test_a_zero_token_measurement_is_refused(self, ctx, printed, store, monkeypatch):
        from core import calibration

        monkeypatch.setattr(
            calibration, "measure",
            lambda tag, prompts=None, on_progress=None: {
                "enter": 0.0, "exit": 0.0, "tokens": 0, "prompts": 55},
        )
        _run(ctx, ["tune"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "logprob" in "\n".join(printed).lower()

    def test_a_below_min_tokens_measurement_is_refused(self, ctx, printed, store, monkeypatch):
        """Below core.calibration.MIN_TOKENS a quantile is noise, not a threshold — the same
        refusal as zero tokens, just a nonzero (but too small) sample."""
        from core import calibration

        monkeypatch.setattr(
            calibration, "measure",
            lambda tag, prompts=None, on_progress=None: {
                "enter": 0.1, "exit": 0.2, "tokens": 10, "prompts": 55},
        )
        _run(ctx, ["tune"])
        assert store.entry_for("qwen3.8:27b") is None
        assert "logprob" in "\n".join(printed).lower()


    def test_ctrl_c_cancels_the_tune_without_killing_the_session(self, ctx, printed, store,
                                                                 monkeypatch):
        """`tune` is the first command that runs for minutes by design, so Ctrl-C is the
        expected abort — and KeyboardInterrupt is not an Exception, so neither the dispatcher's
        catch nor the REPL's command path would keep it from taking the whole session down."""
        from core import calibration

        def interrupted(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(calibration, "measure", interrupted)
        _run(ctx, ["tune"])          # must NOT propagate

        assert store.entry_for("qwen3.8:27b") is None
        assert "cancelled" in "\n".join(printed).lower()


class TestUnknownSubcommand:
    def test_unknown_subcommand_errors_with_usage(self, ctx, printed, store):
        _run(ctx, ["tunne"])
        assert "usage" in "\n".join(printed).lower()

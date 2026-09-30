"""
/models — the model page (the 2026-09-01 /scan fold): the hardware probe -> recommendation
(core/hardware.py), the page that renders both ladders against it, the numbered pick, the
pull-on-consent switch, and the first-launch auto-run.

Everything offline: the probe is replaced by a hand-built HardwareProfile, the daemon by a
stubbed model list, the persist seam by a recorder.
"""

import pytest

from core import model_family
from core.hardware import (
    APPLE_CHIPS,
    BASELINE,
    CLASS_COSTS,
    DECODE_FLOOR_TOK_S,
    DECODE_SLOW_TOK_S,
    EMBEDDER_WEIGHTS_GB,
    FALLBACK_WINDOW,
    HEADROOM_GB,
    HardwareProfile,
    chip_speed,
    decode_tok_s,
    kv_cache_gb,
    need_gb,
    prefill_tok_s,
    recommend,
)

# The windows config.default.yaml ships per class (tests/test_model_family.py pins the template).
_W = {"4b": 32768, "9b": 65536, "27b": 65536, "35b": 131072}


def _profile(**over) -> HardwareProfile:
    """A hand-built Mac. The default is an M2 Max (400 GB/s, 38 GPU cores): fast enough that
    every tier that FITS also clears the speed floor, so the fit tests below read as fit tests.
    Speed fields not given are looked up from the chip the way probe() does."""
    base = dict(os_name="Darwin", arch="arm64", chip="Apple M2 Max", cores=12, ram_gb=16.0,
                gpu_cores=38)
    base.update(over)
    speed = chip_speed(base["chip"], base["gpu_cores"])
    base.setdefault("bandwidth_gbps", speed.bandwidth_gbps)
    base.setdefault("tflops", speed.tflops)
    base.setdefault("baseline", speed.baseline)
    return HardwareProfile(**base)


# --- the cost model ----------------------------------------------------------------------------

def test_every_ladder_class_has_a_cost_and_nothing_else():
    assert set(CLASS_COSTS) == set(model_family.classes())
    assert set(EMBEDDER_WEIGHTS_GB) == set(model_family.embedder_classes())


def test_weights_grow_with_size():
    weights = [CLASS_COSTS[c].weights_gb for c in model_family.classes()]
    assert weights == sorted(weights) and len(set(weights)) == len(weights)


def test_kv_bytes_per_token_follow_the_hybrid_architecture():
    """K+V, f16, on the full-attention layers only (1 in 4): 2 * layers * kv_heads * 256 * 2."""
    assert CLASS_COSTS["4b"].kv_bytes_per_token == 32 * 1024
    assert CLASS_COSTS["9b"].kv_bytes_per_token == 32 * 1024
    assert CLASS_COSTS["27b"].kv_bytes_per_token == 64 * 1024
    assert CLASS_COSTS["35b"].kv_bytes_per_token == 20 * 1024     # MoE: 10 full layers x 2 heads


def test_bytes_read_per_token_are_the_active_weights():
    """A dense tier streams every weight per token; the 35b MoE streams its ~3B active share."""
    for dense in ("4b", "9b", "27b"):
        assert CLASS_COSTS[dense].read_gb_per_token == CLASS_COSTS[dense].weights_gb
    moe = CLASS_COSTS["35b"]
    assert moe.read_gb_per_token == pytest.approx(23.0 * 3.0 / 36.0)
    assert moe.read_gb_per_token < CLASS_COSTS["4b"].read_gb_per_token


def test_need_is_weights_plus_cache_at_the_window_plus_headroom():
    assert kv_cache_gb("27b", 65536) == pytest.approx(4.0)
    assert kv_cache_gb("35b", 131072) == pytest.approx(2.5)
    assert need_gb("27b", 65536) == pytest.approx(17.0 + 4.0 + HEADROOM_GB)
    assert need_gb("4b", 32768) == pytest.approx(3.4 + 1.0 + HEADROOM_GB)
    assert need_gb("27b", 131072) > need_gb("27b", 65536) > need_gb("27b", 32768)


def test_fallback_window_is_the_smallest_shipped_window():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    caps = yaml.safe_load((root / "config.default.yaml").read_text(encoding="utf-8"))["capabilities"]
    assert FALLBACK_WINDOW == min(c["context_window"] for c in caps.values())


# --- the recommendation --------------------------------------------------------------------------

@pytest.mark.parametrize("ram, expected", [
    (8, "4b"), (16, "9b"), (24, "9b"), (32, "27b"), (40, "35b"), (48, "35b"), (128, "35b"),
])
def test_apple_unified_memory_uses_three_quarters_of_ram(ram, expected):
    rec = recommend(_profile(ram_gb=ram), _W)
    assert rec.size_class == expected
    assert rec.budget_gb == pytest.approx(ram * 0.75)


def test_apple_chip_table_covers_every_generation_and_climbs_the_tiers():
    """M1 through M5, base -> Pro -> Max (-> Ultra): bandwidth climbs within a generation and the
    published numbers are the ones in the table (Apple newsroom / Wikipedia, 2026-09-29)."""
    for gen in ("M1", "M2", "M3", "M4", "M5"):
        names = [n for n in APPLE_CHIPS if n.split()[0] == gen]
        assert gen in names and f"{gen} Pro" in names and f"{gen} Max" in names
        bw = [chip_speed(f"Apple {n}", 0).bandwidth_gbps
              for n in (gen, f"{gen} Pro", f"{gen} Max") if n in APPLE_CHIPS]
        assert bw == sorted(bw) and len(set(bw)) == 3, gen
    assert chip_speed("Apple M1", 0).bandwidth_gbps == pytest.approx(68.3)
    assert chip_speed("Apple M4 Pro", 0).bandwidth_gbps == pytest.approx(273)


def test_the_m5_family_is_priced_as_the_m4_family():
    """The M5's per-core neural accelerators are not modelled and its runner numbers are not yet
    measured here: each M5 tier carries its M4 counterpart's bandwidth and compute (the Ultra,
    which the M4 never had, is two M4 Max — every Ultra has been two Max)."""
    for m5, m4 in (("M5", "M4"), ("M5 Pro", "M4 Pro"), ("M5 Max", "M4 Max")):
        assert APPLE_CHIPS[m5] == APPLE_CHIPS[m4], m5
    ultra, mx = chip_speed("Apple M5 Ultra", 0), chip_speed("Apple M4 Max", 40)
    assert ultra.bandwidth_gbps == pytest.approx(2 * mx.bandwidth_gbps)
    assert ultra.tflops == pytest.approx(2 * mx.tflops)


@pytest.mark.parametrize("chip, cores, gbps", [
    ("Apple M3 Max", 30, 300), ("Apple M3 Max", 40, 400),
    ("Apple M4 Max", 32, 410), ("Apple M4 Max", 40, 546),
    ("Apple M5 Max", 32, 410), ("Apple M5 Max", 40, 546),
])
def test_the_gpu_core_count_picks_the_binned_max_bandwidth(chip, cores, gbps):
    assert chip_speed(chip, cores).bandwidth_gbps == pytest.approx(gbps)


def test_compute_is_the_gpu_core_count_times_the_generation_rate():
    speed = chip_speed("Apple M4 Pro", 20)
    assert speed.gpu_cores == 20 and speed.tflops == pytest.approx(20 * 0.46)
    assert not speed.baseline and speed.family == "M4 Pro"
    # No readable core count: the family's full-bin count stands in.
    assert chip_speed("Apple M4 Pro", 0).gpu_cores == 20
    assert chip_speed("Apple M4 Max", 0).bandwidth_gbps == pytest.approx(546)


@pytest.mark.parametrize("chip", ["Intel(R) Core(TM) i9-9980HK", "Apple M9 Ultra", "", "AMD Ryzen 9"])
def test_an_unrecognised_chip_falls_back_to_the_m1_baseline(chip):
    """The baseline is the slowest Apple silicon ever shipped: a real chip is never over-promised."""
    speed = chip_speed(chip, 0)
    assert speed.baseline is True
    assert speed.bandwidth_gbps == BASELINE.bandwidth_gbps == pytest.approx(68.3)
    assert speed.tflops == pytest.approx(8 * 0.325)


def test_decode_speed_is_bandwidth_over_the_bytes_read_per_token():
    """Calibrated against the repo's own measurement: the 9b on the M4 Pro decodes at ~38 tok/s
    (docs/OPTIMIZATIONS.md, 2026-09-08). 273 GB/s x 0.9 / 6.6 GB."""
    m4pro = _profile(chip="Apple M4 Pro", gpu_cores=20, ram_gb=48.0)
    assert decode_tok_s("9b", m4pro) == pytest.approx(38, rel=0.1)
    assert decode_tok_s("27b", m4pro) < decode_tok_s("9b", m4pro) < decode_tok_s("4b", m4pro)
    # The MoE streams 2 GB per token: faster than the 4b on the same machine.
    assert decode_tok_s("35b", m4pro) > decode_tok_s("4b", m4pro)
    # Twice the bandwidth, twice the speed.
    m4max = _profile(chip="Apple M4 Max", gpu_cores=40, ram_gb=128.0)
    assert decode_tok_s("9b", m4max) == pytest.approx(2 * decode_tok_s("9b", m4pro))


def test_prefill_speed_follows_gpu_compute_over_active_params():
    """~420 tok/s measured for the 9b on the M4 Pro (docs/OPTIMIZATIONS.md): 2 FLOPs per
    parameter per token at ~85% of the 9.2 TFLOPS the 20 cores rate."""
    m4pro = _profile(chip="Apple M4 Pro", gpu_cores=20, ram_gb=48.0)
    assert prefill_tok_s("9b", m4pro) == pytest.approx(420, rel=0.15)
    assert prefill_tok_s("27b", m4pro) < prefill_tok_s("9b", m4pro)
    assert prefill_tok_s("35b", m4pro) > prefill_tok_s("9b", m4pro)      # 3B active


def test_a_tier_that_fits_but_decodes_under_the_floor_is_not_recommended():
    """A 32 GB base M4 (120 GB/s) holds the 27b (22.5 of 24 GB) but streams it at ~6 tok/s: the
    9b is the recommendation and the 27b row says so. The same memory on an M4 Pro (273 GB/s)
    runs the 27b at ~14 tok/s and gets it."""
    base = recommend(_profile(chip="Apple M4", gpu_cores=10, ram_gb=32.0), _W)
    assert base.fits["27b"] is True and base.usable["27b"] is False
    assert base.size_class == "9b" and base.slow is False
    assert DECODE_SLOW_TOK_S <= base.decode["27b"] < DECODE_FLOOR_TOK_S
    pro = recommend(_profile(chip="Apple M4 Pro", gpu_cores=20, ram_gb=32.0), _W)
    assert pro.size_class == "27b" and pro.usable["27b"] is True


def test_when_nothing_clears_the_floor_the_fastest_fitting_tier_is_flagged_slow():
    rec = recommend(_profile(ram_gb=16.0, bandwidth_gbps=20.0), _W)   # 4b at ~5 tok/s, 9b at ~3
    assert rec.fits["4b"] and rec.fits["9b"] and not any(rec.usable.values())
    assert rec.size_class == "4b" and rec.slow is True and rec.cramped is False


def test_the_budget_is_three_quarters_of_unified_memory_on_the_baseline_too():
    rec = recommend(_profile(chip="Intel(R) Core(TM) i7", gpu_cores=0, ram_gb=16.0), _W)
    assert rec.budget_gb == pytest.approx(12.0) and "baseline" in rec.reason


def test_the_window_changes_the_answer_on_the_same_machine():
    """32 GB Mac (24 GB budget): 27b fits at 64k (22.5) but not at 128k (26.5)."""
    assert recommend(_profile(ram_gb=32.0), _W).size_class == "27b"
    rec = recommend(_profile(ram_gb=32.0), dict(_W, **{"27b": 131072}))
    assert rec.size_class == "9b"
    assert rec.needs["27b"] == pytest.approx(26.5) and rec.fits["27b"] is False


def test_a_missing_window_falls_back_to_the_family_default():
    rec = recommend(_profile(ram_gb=32.0))
    assert rec.windows == {c: FALLBACK_WINDOW for c in model_family.classes()}


def test_a_machine_too_small_for_any_class_lands_on_the_smallest_with_a_warning():
    rec = recommend(_profile(ram_gb=4.0), _W)
    assert rec.size_class == model_family.classes()[0]
    assert rec.fits == {c: False for c in model_family.classes()}
    assert rec.cramped is True


def test_the_embedder_is_the_largest_that_fits_beside_the_tier():
    # 48 GB (36 budget): 35b at 128k = 27.0, 9 GB left -> the 8b embedder (5.2) fits
    rec = recommend(_profile(ram_gb=48.0), _W)
    assert rec.embedder == "8b" and rec.embedder_fits == {"0.6b": True, "4b": True, "8b": True}
    # 32 GB (24): 27b at 64k = 22.5, 1.5 left -> only the 0.6b embedder (1.1) fits
    rec = recommend(_profile(ram_gb=32.0), _W)
    assert rec.embedder == "0.6b" and rec.embedder_fits == {"0.6b": True, "4b": False, "8b": False}
    # 8 GB (6): 4b = 5.9, nothing fits beside it -> the smallest, flagged as swapping
    rec = recommend(_profile(ram_gb=8.0), _W)
    assert rec.embedder == "0.6b" and not any(rec.embedder_fits.values())


def test_a_tier_is_priced_as_the_model_it_runs_not_its_name():
    """`/models all qwen3.8:27b` on tier 4b: the row must cost 27b (17 GB + 4 GB at 64k), and a
    32 GB machine (24 budget) sees that 4b no longer fits its own name's budget line."""
    rec = recommend(_profile(ram_gb=32.0), dict(_W, **{"4b": 65536}), {"4b": "27b"})
    assert rec.needs["4b"] == pytest.approx(22.5) and rec.cost_classes["4b"] == "27b"
    assert rec.needs["27b"] == pytest.approx(22.5)
    plain = recommend(_profile(ram_gb=32.0), _W)
    assert plain.needs["4b"] == pytest.approx(5.9) and plain.cost_classes["4b"] == "4b"


def test_recommendation_is_always_on_the_ladders():
    for ram in (1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192):
        for name in list(APPLE_CHIPS) + ["unknown"]:
            rec = recommend(_profile(chip=f"Apple {name}", gpu_cores=0, ram_gb=ram), _W)
            assert rec.size_class in model_family.classes()
            assert rec.embedder in model_family.embedder_classes()
            assert rec.size_class in rec.decode and rec.size_class in rec.prefill


# --- the probe ------------------------------------------------------------------------------------

def _wire_probe(monkeypatch, *, system="Darwin", machine="arm64", chip, gpu_cores=0, ram=16.0):
    from core import hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)
    monkeypatch.setattr(hardware, "_cpu_brand", lambda: chip)
    monkeypatch.setattr(hardware, "_gpu_cores", lambda: gpu_cores)
    monkeypatch.setattr(hardware, "_ram_gb", lambda: ram)
    return hardware


def test_probe_detects_apple_silicon_from_the_brand_string_not_the_interpreter_arch(monkeypatch):
    """A Rosetta Python reports x86_64 on an M-series Mac; the chip string is the truth."""
    hw = _wire_probe(monkeypatch, machine="x86_64", chip="Apple M4 Pro", gpu_cores=20, ram=48.0)
    prof = hw.probe()
    assert prof.chip == "Apple M4 Pro" and prof.ram_gb == 48.0 and prof.gpu_cores == 20
    assert prof.bandwidth_gbps == pytest.approx(273) and prof.baseline is False


def test_probe_reads_the_binned_max_from_the_gpu_core_count(monkeypatch):
    hw = _wire_probe(monkeypatch, chip="Apple M4 Max", gpu_cores=32, ram=36.0)
    assert hw.probe().bandwidth_gbps == pytest.approx(410)


def test_probe_never_raises_when_every_reader_fails(monkeypatch):
    from core import hardware

    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(hardware, "_cpu_brand", boom)
    monkeypatch.setattr(hardware, "_gpu_cores", boom)
    monkeypatch.setattr(hardware, "_ram_gb", boom)
    prof = hardware.probe()
    assert prof.chip == "" and prof.ram_gb == 0.0 and prof.gpu_cores == 0
    assert prof.baseline is True and prof.bandwidth_gbps == BASELINE.bandwidth_gbps


def test_gpu_cores_reader_parses_ioreg(monkeypatch):
    from core import hardware

    monkeypatch.setattr(hardware, "_run", lambda cmd, timeout=3:
                        '+-o AGXAcceleratorG16X  <class ...>\n    {\n      "gpu-core-count" = 20\n    }')
    assert hardware._gpu_cores() == 20
    monkeypatch.setattr(hardware, "_run", lambda cmd, timeout=3: "")
    assert hardware._gpu_cores() == 0


def test_the_profile_is_probed_once_per_process_and_rescan_reprobes(monkeypatch):
    from core import hardware

    calls = []
    monkeypatch.setattr(hardware, "_CACHED", None)
    monkeypatch.setattr(hardware, "probe", lambda: calls.append(1) or _profile())
    hardware.profile()
    hardware.profile()
    assert len(calls) == 1
    hardware.profile(rescan=True)
    assert len(calls) == 2


def test_startup_warms_the_probe_interactively_and_never_headless(monkeypatch):
    from app import startup
    from core import hardware

    calls = []
    monkeypatch.setattr(hardware, "profile", lambda **k: calls.append(1))
    monkeypatch.setattr(startup, "sync", lambda verbose=False: None)
    monkeypatch.setattr(startup, "build_agent", lambda: "graph")
    assert startup.startup_load(interactive=True) == ("graph", None)
    assert calls == [1]
    # -p never renders /models: no sysctl / ioreg spawn on the one-shot path.
    assert startup.startup_load(interactive=False) == ("graph", None)
    assert calls == [1]


# --- the page ---------------------------------------------------------------------------------------

@pytest.fixture
def printed(monkeypatch):
    """Every line /models emits, flattened to plain text: the module's _print plus the shared ui
    primitives it renders through (section / table / note / warn). A table row becomes its cell
    texts joined by single spaces."""
    lines = []

    def _text(cell):
        return str(cell[0] if isinstance(cell, tuple) else cell)

    monkeypatch.setattr("commands.runtime._print", lambda line="": lines.append(str(line)))
    monkeypatch.setattr("tui.ui.section", lambda title, subtitle="": lines.extend([title, subtitle]))
    monkeypatch.setattr("tui.ui.table",
                        lambda rows, styles=None: lines.extend(" ".join(_text(c) for c in r) for r in rows))
    monkeypatch.setattr("tui.ui.note", lambda msg: lines.append(msg))
    monkeypatch.setattr("tui.ui.warn", lambda msg: lines.append(msg))
    return lines


def _ladder_cfg(active="4b", windows=_W, num_ctx=None, embedder="qwen3-embedding:8b"):
    """A Config over the whole ladder with the template's windows in `capabilities`."""
    from config import Config

    tiers, caps = {}, {}
    for key, tag in model_family.SIZE_LADDER:
        tiers[key] = {"model": tag, "embedder": embedder}
        if windows:
            caps[tag] = {"context_window": windows[key], "max_context_window": 262144}
    data = {"active_tier": active, "tiers": tiers, "capabilities": caps, "runtime": {}}
    if num_ctx:
        data["runtime"]["num_ctx"] = num_ctx
    return Config(data)


class _Local:
    def __init__(self, name, size_bytes=0):
        self.name = name
        self.size_bytes = size_bytes


@pytest.fixture
def env(monkeypatch, printed):
    """Wire /models to a fake machine + daemon. Returns a mutable dict the test tweaks. The
    default machine is a 48 GB Mac: 35b at 128k (27.0) fits the 36 GB budget with the 8b
    embedder (5.2) beside it — so the recommendation is 35b + the embedder already active."""
    import commands  # noqa: F401
    from commands import runtime

    env = {
        "cfg": _ladder_cfg(),
        "profile": _profile(ram_gb=48.0),
        "pulled": ["qwen3.5:4b", "qwen3-embedding:8b"],
        "daemon": True,
        "tty": True,
        "pick": "",          # the row prompt: Enter = take the recommended tier
        "answer": "n",       # the pull prompt
        "embedder": "n",     # the "switch the embedder too?" confirm after an Enter
        "persisted": [],
        "pull_calls": [],
        "pull_rc": 0,
        "reset": 0,
        "resyncs": 0,
    }
    monkeypatch.setattr("config.get_config", lambda: env["cfg"])
    monkeypatch.setattr(runtime, "_probe", lambda: env["profile"])
    # Like the real one: [] when the daemon is down (reachability is then probed separately).
    monkeypatch.setattr("core.llms.list_local_models",
                        lambda: [_Local(n) for n in env["pulled"]] if env["daemon"] else [])
    monkeypatch.setattr("core.llms.ollama_reachable", lambda: env["daemon"])
    monkeypatch.setattr("core.llms.reset_models", lambda: env.__setitem__("reset", env["reset"] + 1))
    monkeypatch.setattr("commands._utils._stdin_is_tty", lambda: env["tty"])
    def ask(prompt, **_kw):
        if prompt.startswith("pull"):
            return env["answer"]
        if prompt.startswith("embedder:"):
            return env["embedder"]
        return env["pick"]

    monkeypatch.setattr("tui.ui.ask", ask)
    monkeypatch.setattr("commands.config._persist_key",
                        lambda cfg, key: env["persisted"].append((key, cfg.get(key))))
    monkeypatch.setattr(runtime, "_resync_rag_after_model_change",
                        lambda: env.__setitem__("resyncs", env["resyncs"] + 1))

    def fake_pull(model):
        env["pull_calls"].append(model)
        if env["pull_rc"] == 0:
            env["pulled"].append(model)
        return env["pull_rc"]

    monkeypatch.setattr(runtime, "_pull_one", fake_pull)
    return env


def _run(args=""):
    from commands._framework import CommandContext, dispatch

    dispatch(("/models " + args).strip(), CommandContext(state={}, make_initial_state=dict, db_path=""))


def _row(printed, *needles):
    return next(l for l in printed if all(n in l for n in needles))


def test_scan_is_no_longer_an_alias():
    import commands  # noqa: F401
    from commands._framework import COMMANDS, _ALIASES

    assert "scan" not in COMMANDS and "scan" not in _ALIASES


def test_the_page_shows_the_machine_the_budget_and_both_ladders(env, printed):
    _run("list")
    blob = "\n".join(printed)
    assert "Apple M2 Max" in blob and "48 GB unified memory" in blob
    assert "38 GPU cores" in blob and "400 GB/s" in blob
    assert "36 GB for models" in blob
    assert "weights" in blob and "window" in blob and "need" in blob and "tok/s" in blob
    for key, tag in model_family.SIZE_LADDER:
        assert _row(printed, key, tag)
    for key, tag in model_family.EMBEDDER_LADDER:
        assert _row(printed, key, tag)
    assert "KV cache" in blob and "headroom" in blob


def test_rows_are_numbered_one_to_nine_tiers_then_embedders(env, printed):
    _run("list")
    keys = list(model_family.classes()) + list(model_family.embedder_classes())
    for i, key in enumerate(keys, 1):
        assert any(l.split()[:1] == [str(i)] and key in l for l in printed), (i, key)


def test_pulled_state_and_the_recommendation_are_marked(env, printed):
    _run("list")
    assert "✓" in _row(printed, "* 4b", "qwen3.5:4b")          # active + pulled
    assert "✓" not in _row(printed, "35b", "qwen3.6:35b")
    assert "recommended" in _row(printed, "35b", "qwen3.6:35b")
    assert "128k ctx" in _row(printed, "35b", "qwen3.6:35b")
    assert "recommended" in _row(printed, "* 8b", "qwen3-embedding:8b")


def test_a_too_big_tier_and_a_swapping_embedder_are_marked(env, printed):
    env["profile"] = _profile(ram_gb=32.0)    # 24 GB: 27b fits, 35b does not; only 0.6b beside
    _run("list")
    assert "too big" in _row(printed, "35b", "qwen3.6:35b")
    assert "recommended" in _row(printed, "27b", "qwen3.8:27b")
    assert "swaps" in _row(printed, "* 8b", "qwen3-embedding:8b")
    assert "recommended" in _row(printed, "0.6b", "qwen3-embedding:0.6b")


def test_every_tier_row_carries_its_decode_speed(env, printed):
    """The 48 GB M2 Max: 4b ~106, 9b ~55, 27b ~21, 35b ~188 tok/s (360 GB/s effective over 1.9 GB)."""
    _run("list")
    assert "~106 tok/s" in _row(printed, "4b", "qwen3.5:4b")
    assert "~21 tok/s" in _row(printed, "27b", "qwen3.8:27b")
    assert "~188 tok/s" in _row(printed, "35b", "qwen3.6:35b")


def test_the_recommended_tier_says_what_it_feels_like(env, printed):
    _run("list")
    line = _row(printed, "35b", "feels like")
    assert "~188 tok/s" in line and "first prompt" in line and "cold" in line


def test_a_tier_that_fits_but_is_slow_is_marked_and_not_recommended(env, printed):
    env["profile"] = _profile(chip="Apple M4", gpu_cores=10, ram_gb=32.0)
    _run("list")
    assert "slow" in _row(printed, "27b", "qwen3.8:27b") and "~6 tok/s" in _row(printed, "27b", "qwen3.8:27b")
    assert "recommended" in _row(printed, "9b", "qwen3.5:9b")


def test_a_baseline_chip_is_named_on_the_page(env, printed):
    env["profile"] = _profile(chip="Intel(R) Core(TM) i7", gpu_cores=0, ram_gb=16.0)
    _run("list")
    blob = "\n".join(printed)
    assert "baseline" in blob and "M1" in blob


def test_a_rebound_tier_shows_and_prices_the_model_it_runs(env, printed):
    cfg = env["cfg"]
    cfg.set("tiers.4b.model", "qwen3.8:27b")
    _run("list")
    row = _row(printed, "* 4b", "qwen3.8:27b")
    assert "17.0 GB" in row and "64k ctx" in row and "22.5 GB" in row


def test_a_num_ctx_override_reprices_every_tier_and_is_named(env, printed):
    env["cfg"] = _ladder_cfg(num_ctx=131072)
    env["profile"] = _profile(ram_gb=32.0)
    _run("list")
    assert "runtime.num_ctx = 131072" in "\n".join(printed)
    assert "recommended" in _row(printed, "9b", "qwen3.5:9b")      # 27b at 128k is 26.5 > 24


@pytest.mark.parametrize("verb", ["list", "ls", "--check"])
def test_list_never_prompts_or_switches(env, printed, monkeypatch, verb):
    monkeypatch.setattr("tui.ui.ask", lambda prompt: pytest.fail("list must not prompt"))
    _run(verb)
    assert env["cfg"].active_tier == "4b" and env["persisted"] == []


def test_enter_switches_to_the_recommended_tier_when_pulled(env, printed):
    env["pulled"] += ["qwen3.6:35b"]
    _run()
    assert env["cfg"].active_tier == "35b"
    assert env["persisted"] == [("active_tier", "35b")]
    assert env["reset"] == 1 and env["pull_calls"] == []
    assert env["cfg"].embedder_model == "qwen3-embedding:8b"    # the 8b embedder stays


def test_enter_pulls_the_recommended_tier_on_consent(env, printed):
    env["answer"] = "y"
    _run()
    assert env["pull_calls"] == ["qwen3.6:35b"]
    assert env["cfg"].active_tier == "35b"


def test_declining_the_pull_keeps_the_current_tier_and_names_the_command(env, printed):
    _run()
    assert env["pull_calls"] == [] and env["cfg"].active_tier == "4b" and env["persisted"] == []
    assert "ollama pull qwen3.6:35b" in "\n".join(printed)


def test_enter_alone_never_moves_the_embedder(env, printed, monkeypatch):
    """The recommended embedder differs (0.6b beside 27b on 32 GB) but the default answer to the
    confirm is no: the tier switches, the embedder — and the corpus embedding — stay."""
    env["profile"] = _profile(ram_gb=32.0)
    env["pulled"] += ["qwen3.8:27b", "qwen3-embedding:0.6b"]
    _run()
    assert env["cfg"].active_tier == "27b"
    assert env["cfg"].embedder_model == "qwen3-embedding:8b"
    assert env["persisted"] == [("active_tier", "27b")]
    assert env["resyncs"] == 1


def test_enter_does_not_ask_about_the_embedder_when_it_already_matches(env, printed, monkeypatch):
    env["pulled"] += ["qwen3.6:35b"]
    asked = []
    real = __import__("tui.ui", fromlist=["ask"]).ask
    monkeypatch.setattr("tui.ui.ask", lambda p, **k: asked.append(p) or real(p))
    _run()
    assert env["cfg"].active_tier == "35b"
    assert not any(p.startswith("embedder:") for p in asked)


def test_enter_moves_the_embedder_on_a_yes(env, printed):
    env["profile"] = _profile(ram_gb=32.0)      # -> 27b + the 0.6b embedder
    env["pulled"] += ["qwen3.8:27b", "qwen3-embedding:0.6b"]
    env["embedder"] = "y"
    _run()
    assert env["cfg"].active_tier == "27b"
    assert all(env["cfg"].get("tiers")[k]["embedder"] == "qwen3-embedding:0.6b"
               for k in model_family.classes())               # a machine choice: every tier
    assert ("active_tier", "27b") in env["persisted"]
    assert ("tiers.4b.embedder", "qwen3-embedding:0.6b") in env["persisted"]
    assert env["resyncs"] == 2


def test_a_row_number_picks_one_tier(env, printed):
    env["pulled"] += ["qwen3.8:27b"]
    env["pick"] = "3"
    _run()
    assert env["cfg"].active_tier == "27b"
    assert env["persisted"] == [("active_tier", "27b")]


def test_a_row_number_picks_one_embedder_and_pulls_it_on_consent(env, printed):
    env["pick"] = "6"                        # qwen3-embedding:4b, not pulled
    env["answer"] = "y"
    _run()
    assert env["pull_calls"] == ["qwen3-embedding:4b"]
    assert env["cfg"].embedder_model == "qwen3-embedding:4b"
    assert env["cfg"].active_tier == "4b"                       # the tier was not touched
    assert len([k for k, _v in env["persisted"] if k.endswith(".embedder")]) == 4
    assert env["resyncs"] == 1


def test_picking_the_active_embedder_changes_nothing(env, printed):
    env["pick"] = "7"
    _run()
    assert env["persisted"] == [] and env["reset"] == 0
    assert "already on embedder" in "\n".join(printed)


def test_picking_the_active_tier_still_pulls_its_missing_model(env, printed):
    env["cfg"] = _ladder_cfg(active="35b")
    env["answer"] = "y"
    _run()                                      # Enter: 35b is recommended AND active
    assert env["pull_calls"] == ["qwen3.6:35b"]
    assert env["persisted"] == []               # nothing to switch
    assert "models pulled" in "\n".join(printed)


def test_a_too_big_pick_is_honored_with_a_warning(env, printed):
    env["profile"] = _profile(ram_gb=32.0)
    env["pulled"] += ["qwen3.6:35b"]
    env["pick"] = "4"
    _run()
    assert env["cfg"].active_tier == "35b"
    assert "may fail to load" in "\n".join(printed)


def test_a_failed_pull_never_switches(env, printed):
    env["answer"] = "y"
    env["pull_rc"] = 1
    _run()
    assert env["cfg"].active_tier == "4b" and env["persisted"] == []


@pytest.mark.parametrize("reply", ["n", "q", "cancel", "abc", "0", "10", "-1"])
def test_cancel_or_garbage_keeps_everything(env, printed, reply):
    env["pulled"] += ["qwen3.6:35b"]
    env["pick"] = reply
    _run()
    assert env["cfg"].active_tier == "4b" and env["persisted"] == [] and env["pull_calls"] == []
    blob = "\n".join(printed)
    assert "staying on '4b'" in blob
    if reply not in ("n", "q", "cancel"):
        assert "not a valid selection" in blob


def test_session_flag_switches_live_only(env, printed):
    env["pulled"] += ["qwen3.6:35b"]
    _run("--session")
    assert env["cfg"].active_tier == "35b" and env["persisted"] == []


def test_off_tty_never_prompts(env, printed, monkeypatch):
    env["tty"] = False
    monkeypatch.setattr("tui.ui.ask", lambda prompt: pytest.fail("must not prompt off-TTY"))
    _run()
    assert env["cfg"].active_tier == "4b"


def test_daemon_down_renders_unknown_pulled_state_and_never_prompts(env, printed, monkeypatch):
    env["daemon"] = False
    monkeypatch.setattr("tui.ui.ask", lambda prompt: pytest.fail("must not prompt with no daemon"))
    _run()
    assert env["cfg"].active_tier == "4b"
    assert "?" in _row(printed, "35b", "qwen3.6:35b")
    assert "ollama serve" in "\n".join(printed)


def test_an_interrupt_at_the_row_prompt_never_selects(env, printed, monkeypatch):
    """ui.ask returns its `on_interrupt` value on Ctrl-C / Ctrl-D; the row prompt must pass a
    refusal, or an interrupt would select what a bare Enter selects and persist the tier."""
    env["pulled"] += ["qwen3.6:35b"]
    monkeypatch.setattr("tui.ui.ask", lambda prompt, on_interrupt="", **k: on_interrupt)
    _run()
    assert env["cfg"].active_tier == "4b" and env["persisted"] == []
    assert "staying on '4b'" in "\n".join(printed)


def test_ui_ask_hands_back_the_interrupt_value(monkeypatch):
    import sys

    import tui.ui  # noqa: F401  (the package binds `prompt` to a function; take the module)
    prompt_mod = sys.modules["tui.ui.prompt"]

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(prompt_mod, "_live_stop", lambda: None)
    monkeypatch.setattr(prompt_mod, "_RICH", False)
    monkeypatch.setattr("builtins.input", boom)
    assert prompt_mod.ask("x » ") == ""
    assert prompt_mod.ask("x » ", on_interrupt="n") == "n"


def test_a_rebound_tier_pulls_the_model_the_file_names(env, printed):
    """An off-ladder declaration is what the tier runs: the pick pulls exactly that model, and
    the row prices it by the size in its tag (31b costs what the 27b class costs)."""
    cfg = env["cfg"]
    cfg.set("tiers.27b.model", "gemma4:31b")
    env["pick"] = "3"
    env["answer"] = "y"
    _run()
    assert env["pull_calls"] == ["gemma4:31b"]
    assert env["cfg"].active_tier == "27b"
    assert "running as" not in "\n".join(printed)


def test_the_page_renders_when_the_active_tier_has_no_embedder(env, printed):
    del env["cfg"].get("tiers")["4b"]["embedder"]
    _run("list")
    blob = "\n".join(printed)
    assert "▸ recommended" in blob and "no embedder in config.yaml" in blob


def test_an_embedder_pick_aligns_every_tier_even_when_the_active_one_already_matches(env, printed):
    env["cfg"].get("tiers")["27b"]["embedder"] = "qwen3-embedding:0.6b"
    env["pick"] = "7"                                          # the 8b embedder, active on 4b
    _run()
    assert all(env["cfg"].get("tiers")[k]["embedder"] == "qwen3-embedding:8b"
               for k in model_family.classes())
    assert "already on" not in "\n".join(printed)


def test_an_embedder_switch_never_splits_a_dotted_tier_key(env, printed):
    tiers = env["cfg"].get("tiers")
    tiers["4.5b"] = {"model": "qwen3.5:4b", "embedder": "qwen3-embedding:8b"}
    env["pick"] = "6"
    env["answer"] = "y"
    _run()
    assert tiers["4.5b"]["embedder"] == "qwen3-embedding:4b"
    assert "0" not in tiers or tiers.get("0") is None           # no phantom nested tier
    assert not any(k == "tiers.4.5b.embedder" for k, _v in env["persisted"])
    assert "not persisted for tier(s) 4.5b" in "\n".join(printed)


def test_typed_embedder_bind_is_machine_wide_too(env, printed):
    env["cfg"].get("tiers")["27b"]["embedder"] = "qwen3-embedding:0.6b"
    _run("embedder qwen3-embedding:4b")
    assert all(env["cfg"].get("tiers")[k]["embedder"] == "qwen3-embedding:4b"
               for k in model_family.classes())


def test_the_page_lists_the_daemon_once(env, printed, monkeypatch):
    calls = []
    monkeypatch.setattr("core.llms.ollama_reachable", lambda: calls.append(1) or True)
    _run("list")
    assert calls == []                                          # a non-empty list IS the probe


def test_cramped_machine_warns(env, printed):
    env["profile"] = _profile(ram_gb=4.0)
    _run("list")
    assert "too small" in "\n".join(printed)


@pytest.mark.parametrize("sub", ["rescan", "tier 9b"])
def test_rescan_and_tier_are_gone(env, printed, sub):
    _run(sub)
    assert env["cfg"].active_tier == "4b" and env["persisted"] == []
    assert "is gone" in "\n".join(printed)


def test_a_config_without_the_recommended_tier_is_told_so(env, printed):
    from config import Config

    env["cfg"] = Config({"active_tier": "4b", "tiers": {
        "4b": {"model": "qwen3.5:4b", "embedder": "qwen3-embedding:8b"}}, "capabilities": {}})
    _run()
    assert env["cfg"].active_tier == "4b"
    assert "not in config.yaml" in _row(printed, "35b", "qwen3.6:35b")
    assert "config.default.yaml" in "\n".join(printed)


def test_keeping_a_tier_whose_model_is_missing_offers_the_pull(env, printed):
    """The gap the old /config setup check closed: `n` keeps the tier, and a model it binds
    that isn't pulled is still offered (y/N) — declining names the command."""
    env["pulled"] = ["qwen3-embedding:8b"]
    env["pick"] = "n"
    _run()
    assert env["pull_calls"] == [] and env["cfg"].active_tier == "4b"
    assert "ollama pull qwen3.5:4b" in "\n".join(printed)
    env["answer"] = "y"
    _run()
    assert env["pull_calls"] == ["qwen3.5:4b"] and env["cfg"].active_tier == "4b"


# --- first launch -----------------------------------------------------------------------------------

def test_first_launch_runs_models_before_the_health_check():
    """The REPL's first-run block dispatches /models (the page + the pick), then the health
    check — so the warnings describe the tier the pick landed on."""
    import inspect

    from app import repl

    src = inspect.getsource(repl.run_repl)
    first = src.index("if _first_run:\n")
    models_at = src.index('dispatch("/models"')
    assert first < models_at < src.index("_health_check()", models_at)
    assert "/config setup" not in src

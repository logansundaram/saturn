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
    CLASS_COSTS,
    EMBEDDER_WEIGHTS_GB,
    FALLBACK_WINDOW,
    HEADROOM_GB,
    HardwareProfile,
    kv_cache_gb,
    need_gb,
    recommend,
)

# The windows config.default.yaml ships per class (tests/test_model_family.py pins the template).
_W = {"800m": 32768, "2b": 32768, "4b": 32768, "9b": 65536, "27b": 65536, "35b": 131072}


def _profile(**over) -> HardwareProfile:
    base = dict(
        os_name="Darwin", arch="arm64", chip="Apple M2", cores=8, ram_gb=16.0,
        gpu="", vram_gb=None, backend="apple",
    )
    base.update(over)
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
    assert CLASS_COSTS["800m"].kv_bytes_per_token == 12 * 1024
    assert CLASS_COSTS["2b"].kv_bytes_per_token == 12 * 1024
    assert CLASS_COSTS["4b"].kv_bytes_per_token == 32 * 1024
    assert CLASS_COSTS["9b"].kv_bytes_per_token == 32 * 1024
    assert CLASS_COSTS["27b"].kv_bytes_per_token == 64 * 1024
    assert CLASS_COSTS["35b"].kv_bytes_per_token == 20 * 1024     # MoE: 10 full layers x 2 heads


def test_need_is_weights_plus_cache_at_the_window_plus_headroom():
    assert kv_cache_gb("27b", 65536) == pytest.approx(4.0)
    assert kv_cache_gb("35b", 131072) == pytest.approx(2.5)
    assert need_gb("27b", 65536) == pytest.approx(17.0 + 4.0 + HEADROOM_GB)
    assert need_gb("4b", 32768) == pytest.approx(3.4 + 1.0 + HEADROOM_GB)
    assert need_gb("27b", 131072) > need_gb("27b", 65536) > need_gb("27b", 32768)


def test_fallback_window_matches_the_config_family_fallback():
    import config

    assert FALLBACK_WINDOW == config.FAMILY_CONTEXT_WINDOW


# --- the recommendation --------------------------------------------------------------------------

@pytest.mark.parametrize("ram, expected", [
    (8, "4b"), (16, "9b"), (24, "9b"), (32, "27b"), (40, "35b"), (48, "35b"), (128, "35b"),
])
def test_apple_unified_memory_uses_three_quarters_of_ram(ram, expected):
    rec = recommend(_profile(ram_gb=ram), _W)
    assert rec.size_class == expected
    assert rec.budget_gb == pytest.approx(ram * 0.75)


@pytest.mark.parametrize("vram, expected", [
    (6, "4b"), (8, "4b"), (12, "9b"), (16, "9b"), (24, "27b"), (32, "35b"),
])
def test_nvidia_budget_is_the_card_vram_not_system_ram(vram, expected):
    prof = _profile(os_name="Linux", chip="AMD Ryzen 9", gpu="NVIDIA RTX", vram_gb=vram,
                    ram_gb=256.0, backend="nvidia")
    rec = recommend(prof, _W)
    assert rec.size_class == expected and rec.budget_gb == vram


def test_cpu_only_uses_half_of_ram_and_caps_at_9b():
    small = recommend(_profile(os_name="Linux", chip="Intel i5", backend="cpu", ram_gb=16.0), _W)
    assert small.size_class == "4b" and small.budget_gb == 8.0
    big = recommend(_profile(os_name="Windows", chip="Intel i9", backend="cpu", ram_gb=128.0), _W)
    assert big.size_class == "9b" and "cpu" in big.reason.lower()


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
    rec = recommend(_profile(backend="cpu", ram_gb=2.0), _W)
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
    for ram in (1, 4, 8, 12, 16, 24, 32, 48, 64, 96):
        for backend in ("apple", "nvidia", "cpu"):
            prof = _profile(ram_gb=ram, backend=backend, vram_gb=ram if backend == "nvidia" else None)
            rec = recommend(prof, _W)
            assert rec.size_class in model_family.classes()
            assert rec.embedder in model_family.embedder_classes()


# --- the probe ------------------------------------------------------------------------------------

def _wire_probe(monkeypatch, *, system, machine, chip, nvidia=(None, None), ram=16.0):
    from core import hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)
    monkeypatch.setattr(hardware, "_cpu_brand", lambda: chip)
    monkeypatch.setattr(hardware, "_nvidia_vram_gb", lambda: nvidia)
    monkeypatch.setattr(hardware, "_ram_gb", lambda: ram)
    return hardware


def test_probe_detects_apple_silicon_from_the_brand_string_not_the_interpreter_arch(monkeypatch):
    """A Rosetta Python reports x86_64 on an M-series Mac; the chip string is the truth."""
    hw = _wire_probe(monkeypatch, system="Darwin", machine="x86_64", chip="Apple M4 Pro", ram=48.0)
    prof = hw.probe()
    assert prof.backend == "apple" and prof.chip == "Apple M4 Pro" and prof.ram_gb == 48.0


def test_probe_prefers_an_nvidia_card_when_present(monkeypatch):
    hw = _wire_probe(monkeypatch, system="Linux", machine="x86_64", chip="AMD Ryzen 7 7800X3D",
                     nvidia=("NVIDIA GeForce RTX 4090", 24.0), ram=64.0)
    prof = hw.probe()
    assert prof.backend == "nvidia" and prof.vram_gb == 24.0 and "4090" in prof.gpu


def test_probe_falls_back_to_cpu(monkeypatch):
    hw = _wire_probe(monkeypatch, system="Windows", machine="AMD64", chip="Intel Core i7")
    assert hw.probe().backend == "cpu"


def test_probe_never_raises_when_every_reader_fails(monkeypatch):
    from core import hardware

    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(hardware, "_cpu_brand", boom)
    monkeypatch.setattr(hardware, "_nvidia_vram_gb", boom)
    monkeypatch.setattr(hardware, "_ram_gb", boom)
    prof = hardware.probe()
    assert prof.backend == "cpu" and prof.ram_gb == 0.0


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


def test_startup_warms_the_probe(monkeypatch):
    from app import startup
    from core import hardware

    calls = []
    monkeypatch.setattr(hardware, "profile", lambda **k: calls.append(1))
    monkeypatch.setattr(startup, "sync", lambda verbose=False: None)
    monkeypatch.setattr(startup, "build_agent", lambda: "graph")
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
        tiers[key] = {"provider": "ollama",
                      "roles": {r: tag for r in ("planner", "tool_caller", "synthesizer",
                                                 "utility", "judge")},
                      "embedder": embedder}
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
        "calibrated": {"qwen3.5:0.8B", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.8:27b"},
        "persisted": [],
        "pull_calls": [],
        "pull_rc": 0,
        "reset": 0,
        "resyncs": 0,
        "rescans": 0,
    }
    monkeypatch.setattr("config.get_config", lambda: env["cfg"])
    monkeypatch.setattr(runtime, "_probe", lambda: env["profile"])
    monkeypatch.setattr(runtime, "_rescan",
                        lambda: env.__setitem__("rescans", env["rescans"] + 1) or env["profile"])
    monkeypatch.setattr("core.llms.list_local_models", lambda: [_Local(n) for n in env["pulled"]])
    monkeypatch.setattr("core.llms.ollama_reachable", lambda: env["daemon"])
    monkeypatch.setattr("core.llms.reset_models", lambda: env.__setitem__("reset", env["reset"] + 1))
    monkeypatch.setattr("commands.config._stdin_is_tty", lambda: env["tty"])
    def ask(prompt):
        if prompt.startswith("pull"):
            return env["answer"]
        if prompt.startswith("embedder:"):
            return env["embedder"]
        return env["pick"]

    monkeypatch.setattr("tui.ui.ask", ask)
    monkeypatch.setattr(runtime, "_calibrated", lambda tag: tag in env["calibrated"])
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


def test_models_absorbed_scan():
    import commands  # noqa: F401
    from commands._framework import COMMANDS, _ALIASES
    from commands.system import _GROUPS

    assert "scan" not in COMMANDS and _ALIASES["scan"] == "models"
    assert "scan" not in dict(_GROUPS)["system"]


def test_the_page_shows_the_machine_the_budget_and_both_ladders(env, printed):
    _run("list")
    blob = "\n".join(printed)
    assert "Apple M2" in blob and "48 GB unified memory" in blob
    assert "36 GB for models" in blob
    assert "weights" in blob and "window" in blob and "need" in blob
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


def test_a_rebound_tier_shows_and_prices_the_model_it_runs(env, printed):
    cfg = env["cfg"]
    for role in ("planner", "tool_caller", "synthesizer", "utility", "judge"):
        cfg.set(f"tiers.4b.roles.{role}", "qwen3.8:27b")
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


def test_calibration_is_marked_per_chat_row_and_never_on_an_embedder(env, printed):
    _run("list")
    assert "calibrated" in _row(printed, "27b", "qwen3.8:27b")
    assert "uncalibrated" in _row(printed, "35b", "qwen3.6:35b")
    assert "calibrated" not in _row(printed, "8b", "qwen3-embedding:8b")
    assert "/confidence" in "\n".join(printed)


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
    monkeypatch.setattr("tui.ui.ask", lambda p: asked.append(p) or real(p))
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
    env["pick"] = "5"
    _run()
    assert env["cfg"].active_tier == "27b"
    assert env["persisted"] == [("active_tier", "27b")]


def test_a_row_number_picks_one_embedder_and_pulls_it_on_consent(env, printed):
    env["pick"] = "8"                        # qwen3-embedding:4b, not pulled
    env["answer"] = "y"
    _run()
    assert env["pull_calls"] == ["qwen3-embedding:4b"]
    assert env["cfg"].embedder_model == "qwen3-embedding:4b"
    assert env["cfg"].active_tier == "4b"                       # the tier was not touched
    assert len([k for k, _v in env["persisted"] if k.endswith(".embedder")]) == 6
    assert env["resyncs"] == 1


def test_picking_the_active_embedder_changes_nothing(env, printed):
    env["pick"] = "9"
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
    env["pick"] = "6"
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


def test_cramped_machine_warns(env, printed):
    env["profile"] = _profile(backend="cpu", ram_gb=2.0)
    _run("list")
    assert "too small" in "\n".join(printed)


def test_rescan_reprobes(env, printed):
    _run("rescan")
    assert env["rescans"] == 1


def test_a_config_without_the_recommended_tier_is_told_so(env, printed):
    from config import Config

    env["cfg"] = Config({"active_tier": "4b", "tiers": {
        "4b": {"provider": "ollama", "roles": {"planner": "qwen3.5:4b"},
               "embedder": "qwen3-embedding:8b"}}, "capabilities": {}})
    _run()
    assert env["cfg"].active_tier == "4b"
    assert "not in config.yaml" in _row(printed, "35b", "qwen3.6:35b")
    assert "config.default.yaml" in "\n".join(printed)


def test_direct_tier_switch_still_works(env, printed):
    _run("tier 9b")
    assert env["cfg"].active_tier == "9b" and env["persisted"] == [("active_tier", "9b")]


# --- first launch -----------------------------------------------------------------------------------

def test_first_launch_runs_models_before_the_setup_check():
    """The REPL's first-run block dispatches /models (the page + the pick), then /config setup —
    so the doctor checks the tier the pick landed on."""
    import inspect

    from app import repl

    src = inspect.getsource(repl.run_repl)
    models_at = src.index('dispatch("/models"')
    setup_at = src.index('dispatch("/config setup"')
    assert models_at < setup_at
    assert "if _first_run:" in src[:models_at]

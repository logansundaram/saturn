"""
Hardware probe -> size-class recommendation (2026-09-01; Apple-silicon-only with a speed axis
since 2026-09-29), behind `/models` and the first launch.

The install default is the 4b tier because the first pull should be light — but the useful
experience lives at 27b, and getting there meant guessing what your machine can hold and
editing config.yaml. `/models` does the guess with a rule that is stated, not learned, on
two axes:

  fit     the memory the model runner can address: macOS lets the GPU address roughly three
          quarters of unified memory, so budget = 0.75 x RAM. A tier FITS when its Q4_K_M
          weights + the KV cache its context window costs + a fixed headroom are under that.
  speed   what the tier will feel like once it fits. Decode on Apple silicon is memory-bandwidth
          bound (docs/OPTIMIZATIONS.md §3: the 9b reads 6.6 GB per token and lands at ~38 tok/s
          on an M4 Pro whichever runner serves it), so decode tok/s = bandwidth x efficiency /
          bytes read per token — the full weights for a dense tier, the ACTIVE share for the 35b
          MoE. Prefill is compute bound: 2 FLOPs per active parameter per token against the GPU's
          TFLOPS. The recommendation is the largest tier that fits AND clears DECODE_FLOOR_TOK_S;
          a tier that fits but streams under it is shown as slow, never picked by default.

Bandwidth and compute come from a table of Apple's published numbers per chip family (APPLE_CHIPS),
keyed by the brand string sysctl reports, with the GPU core count ioreg reports picking the bin
for the binned Max chips (a 32-core M4 Max has 410 GB/s, the 40-core 546). A chip the table
does not know — an Intel Mac, a generation newer than this file — falls back to BASELINE, the
slowest Apple silicon ever shipped, and the page says so: a real chip is never over-promised.

Two halves, kept apart so the rule is testable without a machine to match it:

  probe()      reads the machine: chip, cores, RAM, GPU cores. Every reader is wrapped — a probe
               that raised would take the first launch down with it.
  recommend()  a PURE function of the profile + the context window each class runs at.

The per-class need is COMPUTED, not tabled (see need_gb): Q4_K_M weights + the KV cache the
configured context window costs + a fixed headroom. The KV cost comes from the architecture:
every qwen3.5-3.8 tag is a hybrid — three Gated-DeltaNet (linear-attention) layers for every
one full-attention layer — and only the full-attention layers keep a per-token cache, which is
why a 27b at 64k context costs 4 GB of cache rather than the 16+ GB a dense 27b would. The
linear layers hold a small fixed recurrent state (tens of MB) that lives in the headroom.

LEAF: stdlib + core.model_family. Never imports config or the TUI, so commands/ and app/ can
call it from anywhere.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Mapping

from core import model_family


@dataclass(frozen=True)
class ClassCost:
    """What a size class costs to hold and to stream, from `ollama list` + the model's
    architecture."""

    weights_gb: float     # Q4_K_M download size of the ladder tag (ollama list, 2026-09-01)
    full_layers: int      # full-attention layers = block_count // full_attention_interval (4)
    kv_heads: int         # attention.head_count_kv on those layers
    params_b: float       # total parameters (billions, `ollama show`)
    active_b: float       # parameters touched per token: all of them on a dense model, the
                          # routed experts + shared layers on an MoE
    head_dim: int = 256   # attention.key_length == value_length, every tag

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, f16 (Ollama's default cache type), on the full-attention layers only."""
        return 2 * self.full_layers * self.kv_heads * self.head_dim * 2

    @property
    def read_gb_per_token(self) -> float:
        """Weight bytes streamed from memory per decoded token (at a short context — the cache
        read grows with the window and is not modelled): the active share of the weights."""
        return self.weights_gb * self.active_b / self.params_b


# Architecture facts: `ollama show --verbose` for the pulled tags, the HF config.json for the
# rest (num_key_value_heads, layer_types, num_experts_per_tok). tests/test_models_page.py
# asserts this table and SIZE_LADDER carry the same classes.
CLASS_COSTS: dict[str, ClassCost] = {
    "4b": ClassCost(weights_gb=3.4, full_layers=8, kv_heads=4, params_b=4.7, active_b=4.7),
    "9b": ClassCost(weights_gb=6.6, full_layers=8, kv_heads=4, params_b=9.7, active_b=9.7),
    "27b": ClassCost(weights_gb=17.0, full_layers=16, kv_heads=4, params_b=27.3, active_b=27.3),
    "35b": ClassCost(weights_gb=23.0, full_layers=10, kv_heads=2, params_b=36.0, active_b=3.0),
}

# Compute buffers, the vision projector the tags ship, the linear layers' recurrent state, and
# the daemon's own allocations. Flat: none of it scales with the window the way the cache does.
HEADROOM_GB = 1.5

# The embedder ladder's Q4 weights (ollama list, 2026-09-01). An embedder runs short inputs (a
# chunk at a time), so its working memory is a flat allowance, not a window-scaled cache.
EMBEDDER_WEIGHTS_GB: dict[str, float] = {"0.6b": 0.6, "4b": 2.5, "8b": 4.7}
EMBEDDER_HEADROOM_GB = 0.5

# When the caller has no config to read windows from: the smallest runtime window the ladder
# ships in config.default.yaml (a fallback must never over-allocate).
FALLBACK_WINDOW = 32768

# The share of unified memory macOS lets the GPU address (see the module docstring).
_APPLE_SHARE = 0.75

# ── speed ──────────────────────────────────────────────────────────────────────────────────────

# How much of the peak bandwidth / compute a llama.cpp or MLX runner actually turns into tokens.
# Fitted to the one machine this repo has measured (docs/OPTIMIZATIONS.md §3, M4 Pro 20-core,
# 2026-09-08): 273 GB/s x 0.9 / 6.6 GB = 37 tok/s against 38 measured; 9.2 TFLOPS x 0.85 /
# (2 x 9.7 GFLOP per token) = 403 tok/s against ~420 measured.
DECODE_EFFICIENCY = 0.9
PREFILL_EFFICIENCY = 0.85

# A tier is recommended only when it decodes at least this fast (a paragraph in ~15 s); between
# SLOW and FLOOR it is shown as slow, below SLOW as too slow. The floor is a product judgment,
# not a hardware fact: an agent that answers slower than you read is one you stop asking.
DECODE_FLOOR_TOK_S = 10.0
DECODE_SLOW_TOK_S = 5.0

# What the agent's stable prefix (system prompt + tool catalog + grounding) costs to prefill on
# the session's first turn — core/prime.py plants it once and every later turn extends it.
# Approximate; the "first prompt ~N s cold" line is an order of magnitude, not a promise.
AGENT_PREFIX_TOKENS = 3000


@dataclass(frozen=True)
class ChipSpec:
    """One Apple chip family: bandwidth per GPU-core bin and FP32 TFLOPS per GPU core."""

    bandwidth: tuple[tuple[int, float], ...]   # ((min_gpu_cores, GB/s), ...) ascending — the
                                               # last bin whose floor the count reaches wins
    gpu_cores: int                             # the full-bin core count (stands in when ioreg
                                               # gave none)
    tflops_per_core: float                     # FP32, Apple's published peak / core count

    @property
    def bandwidth_gbps(self) -> float:
        """The family's base bin."""
        return self.bandwidth[0][1]

    def bandwidth_for(self, gpu_cores: int) -> float:
        chosen = self.bandwidth[0][1]
        for floor, gbps in self.bandwidth:
            if gpu_cores >= floor:
                chosen = gbps
        return chosen


def _spec(gbps: float, gpu_cores: int, tflops_per_core: float, *bins: tuple[int, float]) -> ChipSpec:
    return ChipSpec(bandwidth=((0, gbps),) + bins, gpu_cores=gpu_cores,
                    tflops_per_core=tflops_per_core)


# Apple's published memory bandwidth and GPU core counts per family (newsroom / tech specs, via
# the Wikipedia M1–M4 pages, read 2026-09-29), keyed by the family name after "Apple " in the
# brand string. TFLOPS per core is the published FP32 peak over the full-bin core count (M1 2.6
# over 8, M2 3.6 over 10, M4 ~4.6 over 10); M3 was not published and is priced as M2. The M5
# family is priced AS THE M4 family (2026-09-30): its published bandwidth is higher (153 / 307 /
# 460–614 GB/s) and its GPU cores carry neural accelerators, but no runner number has been
# measured here, so each M5 tier carries its M4 counterpart's decode and prefill — the Ultra,
# which the M4 never had, is two M4 Max. Re-price once an M5 has been measured. When a count is
# not readable, a binned Max is priced as its full bin (the bins differ by <1.35x).
APPLE_CHIPS: dict[str, ChipSpec] = {
    "M1": _spec(68.3, 8, 0.325),
    "M1 Pro": _spec(200, 16, 0.325),
    "M1 Max": _spec(400, 32, 0.325),
    "M1 Ultra": _spec(800, 64, 0.325),
    "M2": _spec(100, 10, 0.36),
    "M2 Pro": _spec(200, 19, 0.36),
    "M2 Max": _spec(400, 38, 0.36),
    "M2 Ultra": _spec(800, 76, 0.36),
    "M3": _spec(100, 10, 0.36),
    "M3 Pro": _spec(150, 18, 0.36),
    "M3 Max": _spec(300, 40, 0.36, (40, 400)),          # 30-core 300, 40-core 400
    "M3 Ultra": _spec(800, 80, 0.36),
    "M4": _spec(120, 10, 0.46),
    "M4 Pro": _spec(273, 20, 0.46),
    "M4 Max": _spec(410, 40, 0.46, (40, 546)),          # 32-core 410, 40-core 546
}
APPLE_CHIPS.update({
    "M5": APPLE_CHIPS["M4"],
    "M5 Pro": APPLE_CHIPS["M4 Pro"],
    "M5 Max": APPLE_CHIPS["M4 Max"],
    "M5 Ultra": _spec(2 * 546, 80, 0.46),
})

# The fallback for a chip the table does not know: the original M1 — the slowest Apple silicon
# ever shipped, so an unknown machine is under-promised rather than over-promised.
BASELINE_FAMILY = "M1"
BASELINE = APPLE_CHIPS[BASELINE_FAMILY]

# "Apple M4 Pro" -> "M4 Pro"; tolerant of the "(Virtual)" and clock suffixes some VMs append.
_FAMILY_RE = re.compile(r"\bM(\d+)(?:\s+(Pro|Max|Ultra))?\b", re.IGNORECASE)


@dataclass(frozen=True)
class ChipSpeed:
    """What chip_speed resolved for a machine: the numbers the speed axis prices with."""

    family: str          # table key, e.g. "M4 Pro"; BASELINE_FAMILY when baseline
    gpu_cores: int       # the count used (read, or the family's full bin)
    bandwidth_gbps: float
    tflops: float
    baseline: bool       # the chip was not recognised; numbers are the M1's


def chip_speed(chip: str, gpu_cores: int) -> ChipSpeed:
    """The bandwidth and compute behind a brand string + GPU core count. A string that names no
    known family (Intel, a future M-series, empty) resolves to BASELINE, flagged."""
    text = str(chip or "")
    found = _FAMILY_RE.search(text) if "apple" in text.lower() else None
    family = None
    if found:
        candidate = f"M{found.group(1)}" + (f" {found.group(2).title()}" if found.group(2) else "")
        if candidate in APPLE_CHIPS:
            family = candidate
    spec = APPLE_CHIPS[family] if family else BASELINE
    cores = int(gpu_cores or 0) if family else 0
    if cores <= 0:
        cores = spec.gpu_cores
    return ChipSpeed(
        family=family or BASELINE_FAMILY, gpu_cores=cores,
        bandwidth_gbps=spec.bandwidth_for(cores),
        tflops=round(cores * spec.tflops_per_core, 2), baseline=family is None,
    )


def kv_cache_gb(size_class: str, num_ctx: int) -> float:
    """The KV cache a class holds at `num_ctx` tokens."""
    return CLASS_COSTS[size_class].kv_bytes_per_token * int(num_ctx) / 1024**3


def need_gb(size_class: str, num_ctx: int) -> float:
    """Addressable memory a class needs at `num_ctx`: weights + KV cache + headroom."""
    return CLASS_COSTS[size_class].weights_gb + kv_cache_gb(size_class, num_ctx) + HEADROOM_GB


def embedder_need_gb(size_class: str) -> float:
    """Addressable memory an embedder class needs: weights + a flat working allowance."""
    return EMBEDDER_WEIGHTS_GB[size_class] + EMBEDDER_HEADROOM_GB


@dataclass
class HardwareProfile:
    os_name: str          # platform.system(): Darwin (anything else prices as baseline)
    arch: str             # platform.machine() — informational only (Rosetta lies; see probe)
    chip: str             # CPU brand string, e.g. "Apple M4 Pro"
    cores: int            # CPU cores
    ram_gb: float         # unified memory
    gpu_cores: int        # ioreg gpu-core-count; 0 when unreadable
    bandwidth_gbps: float # from chip_speed
    tflops: float         # from chip_speed
    baseline: bool        # the chip was not recognised (see BASELINE)


def decode_tok_s(size_class: str, profile: HardwareProfile) -> float:
    """Estimated decode speed: bandwidth over the weight bytes streamed per token."""
    read = CLASS_COSTS[size_class].read_gb_per_token
    return profile.bandwidth_gbps * DECODE_EFFICIENCY / read


def prefill_tok_s(size_class: str, profile: HardwareProfile) -> float:
    """Estimated prompt-processing speed: GPU TFLOPS over the 2 FLOPs each active parameter
    costs per token."""
    active = CLASS_COSTS[size_class].active_b
    return profile.tflops * 1e12 * PREFILL_EFFICIENCY / (2 * active * 1e9)


@dataclass
class Recommendation:
    size_class: str
    budget_gb: float
    reason: str                       # one line: how the budget was derived
    windows: dict[str, int] = field(default_factory=dict)    # class -> num_ctx priced
    needs: dict[str, float] = field(default_factory=dict)    # class -> need_gb at that window
    fits: dict[str, bool] = field(default_factory=dict)      # class -> need <= budget
    decode: dict[str, float] = field(default_factory=dict)   # class -> est. decode tok/s
    prefill: dict[str, float] = field(default_factory=dict)  # class -> est. prefill tok/s
    usable: dict[str, bool] = field(default_factory=dict)    # class -> decode >= the floor
    slow: bool = False                # every fitting tier is under the floor; size_class is the
                                      # fastest of them
    cramped: bool = False             # nothing fits; size_class is the smallest as a best effort
    # The embedder that rides alongside: the largest whose need fits the budget LEFT OVER by the
    # recommended tier (RAG lookups then never evict the chat model); when none does, the smallest.
    embedder: str = ""
    embedder_needs: dict[str, float] = field(default_factory=dict)
    embedder_fits: dict[str, bool] = field(default_factory=dict)   # fits beside size_class
    cost_classes: dict[str, str] = field(default_factory=dict)     # tier -> class it is priced as

    def first_prompt_s(self, size_class: str | None = None) -> float:
        """Seconds the agent's cold prefix takes to prefill on a tier (the recommended one by
        default)."""
        rate = self.prefill.get(size_class or self.size_class) or 1.0
        return AGENT_PREFIX_TOKENS / rate


# ── probe ──────────────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: float = 3) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          check=True).stdout.strip()


def _cpu_brand() -> str:
    """The marketing name of the CPU. This is the ONLY reliable Apple-silicon signal: a Python
    running under Rosetta reports platform.machine() == "x86_64" on an M-series Mac."""
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    """Total physical memory in GB."""
    if platform.system() == "Darwin":
        return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 1024**3, 1)
    raise RuntimeError("no memory reader for this platform")


_GPU_CORES_RE = re.compile(r'"gpu-core-count"\s*=\s*(\d+)')


def _gpu_cores() -> int:
    """The GPU core count from the accelerator's IORegistry entry (~25 ms; the number that tells
    a 32-core M4 Max from a 40-core one). 0 when the key is absent."""
    if platform.system() != "Darwin":
        return 0
    out = _run(["ioreg", "-rd1", "-c", "IOAccelerator"])
    found = _GPU_CORES_RE.search(out)
    return int(found.group(1)) if found else 0


def _cores() -> int:
    import os

    return os.cpu_count() or 0


_CACHED: HardwareProfile | None = None


def profile(*, rescan: bool = False) -> HardwareProfile:
    """The machine, probed ONCE per process (app.startup warms it at launch; /models reads it).
    `rescan=True` probes again."""
    global _CACHED
    if _CACHED is None or rescan:
        _CACHED = probe()
    return _CACHED


def probe() -> HardwareProfile:
    """Read the machine. Never raises: a reader that fails leaves its field empty and the speed
    degrades toward BASELINE — a wrong recommendation is recoverable, a crashed first launch is
    not."""
    try:
        chip = _cpu_brand() or ""
    except Exception:
        chip = ""
    try:
        ram = float(_ram_gb())
    except Exception:
        ram = 0.0
    try:
        gpu_cores = int(_gpu_cores() or 0)
    except Exception:
        gpu_cores = 0
    try:
        cores = _cores()
    except Exception:
        cores = 0
    speed = chip_speed(chip, gpu_cores)
    return HardwareProfile(
        os_name=platform.system(), arch=platform.machine(), chip=chip, cores=cores, ram_gb=ram,
        gpu_cores=gpu_cores, bandwidth_gbps=speed.bandwidth_gbps,
        tflops=speed.tflops, baseline=speed.baseline,
    )


# ── recommend ──────────────────────────────────────────────────────────────────────────────────

def _budget(profile: HardwareProfile) -> tuple[float, str]:
    """(budget_gb, reason): the memory the runner can address and a short clause saying where
    the number came from (the readout leads with the number itself)."""
    budget = round(profile.ram_gb * _APPLE_SHARE, 2)
    reason = f"~{int(_APPLE_SHARE * 100)}% of unified memory is GPU-addressable"
    if profile.baseline:
        reason += (f" · chip not recognised, speed priced at the {BASELINE_FAMILY} baseline "
                   f"({BASELINE.bandwidth_gbps:g} GB/s)")
    return budget, reason


def recommend(profile: HardwareProfile,
              windows: Mapping[str, int] | None = None,
              cost_classes: Mapping[str, str] | None = None) -> Recommendation:
    """The largest ladder class whose need — at the context window `windows` gives it (the
    config's effective num_ctx per class; FALLBACK_WINDOW for any class missing) — fits the
    budget AND decodes at DECODE_FLOOR_TOK_S or better; then the largest embedder that fits
    beside it. When every fitting tier is under the floor, the fastest of them is returned
    flagged `slow`; when no tier fits, the smallest is returned flagged `cramped` — the caller
    warns instead of refusing, because a too-small tier that runs beats no tier at all.

    `cost_classes` maps a tier key to the class whose COST it carries — a tier rebound to a
    different size (`/models use qwen3.8:27b` on tier 4b) must be priced as what it runs, not
    as its name. Missing keys cost their own class."""
    budget, reason = _budget(profile)
    classes = model_family.classes()
    given = dict(windows or {})
    costs = {c: (cost_classes or {}).get(c, c) for c in classes}
    used = {c: int(given.get(c) or FALLBACK_WINDOW) for c in classes}
    needs = {c: round(need_gb(costs[c], used[c]), 2) for c in classes}
    fits = {c: needs[c] <= budget for c in classes}
    decode = {c: round(decode_tok_s(costs[c], profile), 1) for c in classes}
    prefill = {c: round(prefill_tok_s(costs[c], profile), 1) for c in classes}
    usable = {c: decode[c] >= DECODE_FLOOR_TOK_S for c in classes}

    fitting = [c for c in classes if fits[c]]
    good = [c for c in fitting if usable[c]]
    if good:
        rec = Recommendation(good[-1], budget, reason, used, needs, fits, decode, prefill, usable)
    elif fitting:
        fastest = max(fitting, key=lambda c: decode[c])
        rec = Recommendation(fastest, budget, reason, used, needs, fits, decode, prefill, usable,
                             slow=True)
    else:
        rec = Recommendation(classes[0], budget, reason, used, needs, fits, decode, prefill, usable,
                             cramped=True)

    left = budget - needs[rec.size_class]
    emb_classes = model_family.embedder_classes()
    rec.embedder_needs = {e: round(embedder_need_gb(e), 2) for e in emb_classes}
    rec.embedder_fits = {e: rec.embedder_needs[e] <= left for e in emb_classes}
    emb_fitting = [e for e in emb_classes if rec.embedder_fits[e]]
    rec.embedder = emb_fitting[-1] if emb_fitting else emb_classes[0]
    rec.cost_classes = costs
    return rec

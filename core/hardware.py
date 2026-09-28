"""
Hardware probe -> size-class recommendation (2026-09-01), behind `/models` (`/scan` is its alias).

The install default is the 4b tier because the first pull should be light — but the useful
experience lives at 27b, and until now getting there meant reading the README, guessing what
your machine can hold, and editing config.yaml. `/scan` does the guess with a rule that is
stated, not learned: measure the memory the model runner can actually address on this machine,
and pick the largest ladder class whose weights + working context fit in it.

Two halves, kept apart so the rule is testable without a machine to match it:

  probe()      reads the machine: OS, chip, cores, RAM, NVIDIA VRAM. Every reader is wrapped —
               a probe that raised would take the first launch down with it.
  recommend()  a PURE function of the profile + the context window each class runs at. The
               budget rule per backend:
                 apple   unified memory; macOS lets the GPU address roughly three quarters of
                         it, so budget = 0.75 x RAM
                 nvidia  the card's VRAM, full — system RAM is not where the model runs
                 cpu     no accelerator: budget = 0.5 x RAM (the rest is the OS + the agent),
                         and the class is capped at 9b — a 27b on CPU fits, but a plan step
                         that takes minutes is not a tier anyone would choose

The per-class need is COMPUTED, not tabled (see need_gb): Q4_K_M weights + the KV cache the
configured context window costs + a fixed headroom. The KV cost comes from the architecture:
every qwen3.5-3.8 tag is a hybrid — three Gated-DeltaNet (linear-attention) layers for every
one full-attention layer — and only the full-attention layers keep a per-token cache, which is
why a 27b at 64k context costs 4 GB of cache rather than the 16+ GB a dense 27b would. The
linear layers hold a small fixed recurrent state (tens of MB) that lives in the headroom.

LEAF: stdlib + psutil (already a dependency for the status bar) + core.model_family. Never
imports config or the TUI, so commands/ and app/ can call it from anywhere.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Mapping

from core import model_family


@dataclass(frozen=True)
class ClassCost:
    """What a size class costs to hold, from `ollama list` + the model's architecture."""

    weights_gb: float     # Q4_K_M download size of the ladder tag (ollama list, 2026-09-01)
    full_layers: int      # full-attention layers = block_count // full_attention_interval (4)
    kv_heads: int         # attention.head_count_kv on those layers
    head_dim: int = 256   # attention.key_length == value_length, every tag

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, f16 (Ollama's default cache type), on the full-attention layers only."""
        return 2 * self.full_layers * self.kv_heads * self.head_dim * 2


# Architecture facts: `ollama show --verbose` for the pulled tags, the HF config.json for the
# rest (num_key_value_heads, layer_types). tests/test_models_page.py asserts this table and SIZE_LADDER
# carry the same classes.
CLASS_COSTS: dict[str, ClassCost] = {
    "4b": ClassCost(weights_gb=3.4, full_layers=8, kv_heads=4),      # 32 layers
    "9b": ClassCost(weights_gb=6.6, full_layers=8, kv_heads=4),      # 32 layers
    "27b": ClassCost(weights_gb=17.0, full_layers=16, kv_heads=4),   # 65 layers
    "35b": ClassCost(weights_gb=23.0, full_layers=10, kv_heads=2),   # 40 layers, MoE (3B active)
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

# The share of memory the model runner can address, per backend (see the module docstring).
_APPLE_SHARE = 0.75
_CPU_SHARE = 0.5
# The largest class worth running without an accelerator.
_CPU_CAP = "9b"


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
    os_name: str          # platform.system(): Darwin / Linux
    arch: str             # platform.machine() — informational only (Rosetta lies; see probe)
    chip: str             # CPU brand string, e.g. "Apple M4 Pro", "AMD Ryzen 9 7950X"
    cores: int
    ram_gb: float
    gpu: str              # NVIDIA card name, or "" when none was found
    vram_gb: float | None
    backend: str          # apple | nvidia | cpu
    gpu_error: str = ""   # why the NVIDIA probe failed (nvidia-smi present but timed out / [N/A]);
                          # "" when it ran or there was nothing to run — the page names a failure
                          # so a driver still coming up is not mistaken for "no accelerator"


@dataclass
class Recommendation:
    size_class: str
    budget_gb: float
    reason: str                       # one line: how the budget was derived
    windows: dict[str, int] = field(default_factory=dict)    # class -> num_ctx priced
    needs: dict[str, float] = field(default_factory=dict)    # class -> need_gb at that window
    fits: dict[str, bool] = field(default_factory=dict)      # class -> need <= budget
    cramped: bool = False             # nothing fits; size_class is the smallest as a best effort
    # The embedder that rides alongside: the largest whose need fits the budget LEFT OVER by the
    # recommended tier (RAG lookups then never evict the chat model); when none does, the smallest.
    embedder: str = ""
    embedder_needs: dict[str, float] = field(default_factory=dict)
    embedder_fits: dict[str, bool] = field(default_factory=dict)   # fits beside size_class
    cost_classes: dict[str, str] = field(default_factory=dict)     # tier -> class it is priced as


# ── probe ──────────────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: float = 3) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          check=True).stdout.strip()


def _cpu_brand() -> str:
    """The marketing name of the CPU. On macOS this is the ONLY reliable Apple-silicon signal:
    a Python running under Rosetta reports platform.machine() == "x86_64" on an M-series Mac."""
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    import psutil

    return round(psutil.virtual_memory().total / 1024**3, 1)


def _nvidia_vram_gb() -> tuple[str | None, float | None]:
    """(card name, total VRAM in GB) for the first NVIDIA GPU, or (None, None). The same
    nvidia-smi query the status-bar gauge runs."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None, None
    out = _run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    first = out.splitlines()[0]
    name, mem = [v.strip() for v in first.rsplit(",", 1)]
    return name, round(float(mem) / 1024, 1)


def _cores() -> int:
    import os

    return os.cpu_count() or 0


_CACHED: HardwareProfile | None = None


def profile(*, rescan: bool = False) -> HardwareProfile:
    """The machine, probed ONCE per process (app.startup warms it at launch; /models reads it).
    `rescan=True` probes again — hardware doesn't change mid-session, but a GPU driver coming
    up after launch is a real case."""
    global _CACHED
    if _CACHED is None or rescan:
        _CACHED = probe()
    return _CACHED


def probe() -> HardwareProfile:
    """Read the machine. Never raises: a reader that fails leaves its field empty and the backend
    degrades toward "cpu" — a wrong recommendation is recoverable, a crashed first launch is not."""
    system = platform.system()
    try:
        chip = _cpu_brand() or ""
    except Exception:
        chip = ""
    try:
        ram = float(_ram_gb())
    except Exception:
        ram = 0.0
    gpu_error = ""
    try:
        gpu, vram = _nvidia_vram_gb()
    except Exception as exc:
        gpu, vram = None, None
        gpu_error = f"{exc.__class__.__name__}: {exc}"[:120]
    try:
        cores = _cores()
    except Exception:
        cores = 0

    if gpu and vram:
        backend = "nvidia"
    elif system == "Darwin" and "apple" in chip.lower():
        backend = "apple"
    else:
        backend = "cpu"

    return HardwareProfile(
        os_name=system, arch=platform.machine(), chip=chip, cores=cores, ram_gb=ram,
        gpu=gpu or "", vram_gb=vram, backend=backend, gpu_error=gpu_error,
    )


# ── recommend ──────────────────────────────────────────────────────────────────────────────────

def _budget(profile: HardwareProfile) -> tuple[float, str, str | None]:
    """(budget_gb, reason, cap_class): the memory the runner can address, a short clause saying
    where the number came from (the readout leads with the number itself), plus an optional
    ceiling on the class."""
    if profile.backend == "nvidia" and profile.vram_gb:
        card = profile.gpu or "NVIDIA GPU"
        return (float(profile.vram_gb), f"the {card}'s {profile.vram_gb:g} GB VRAM", None)
    if profile.backend == "apple":
        budget = round(profile.ram_gb * _APPLE_SHARE, 2)
        return (budget,
                f"Apple silicon · ~{int(_APPLE_SHARE * 100)}% of unified memory is GPU-addressable",
                None)
    budget = round(profile.ram_gb * _CPU_SHARE, 2)
    return (budget,
            f"no accelerator · CPU inference from {int(_CPU_SHARE * 100)}% of RAM, "
            f"capped at {_CPU_CAP} (bigger runs, but too slowly)", _CPU_CAP)


def recommend(profile: HardwareProfile,
              windows: Mapping[str, int] | None = None,
              cost_classes: Mapping[str, str] | None = None) -> Recommendation:
    """The largest ladder class whose need — at the context window `windows` gives it (the
    config's effective num_ctx per class; FALLBACK_WINDOW for any class missing) — fits the
    budget, honoring the backend's cap; then the largest embedder that fits beside it. When no
    tier fits, the smallest class is returned flagged `cramped` — the caller warns instead of
    refusing, because a too-small tier that runs beats no tier at all.

    `cost_classes` maps a tier key to the class whose COST it carries — a tier rebound to a
    different size (`/models all qwen3.8:27b` on tier 4b) must be priced as what it runs, not
    as its name. Missing keys cost their own class."""
    budget, reason, cap = _budget(profile)
    classes = model_family.classes()
    given = dict(windows or {})
    costs = dict(cost_classes or {})
    used = {c: int(given.get(c) or FALLBACK_WINDOW) for c in classes}
    needs = {c: round(need_gb(costs.get(c, c), used[c]), 2) for c in classes}
    fits = {c: needs[c] <= budget for c in classes}
    allowed = list(classes)
    if cap in allowed:
        allowed = allowed[:allowed.index(cap) + 1]
    fitting = [c for c in allowed if fits[c]]
    if fitting:
        rec = Recommendation(fitting[-1], budget, reason, used, needs, fits)
    else:
        rec = Recommendation(classes[0], budget, reason, used, needs, fits, cramped=True)

    left = budget - needs[rec.size_class]
    emb_classes = model_family.embedder_classes()
    rec.embedder_needs = {e: round(embedder_need_gb(e), 2) for e in emb_classes}
    rec.embedder_fits = {e: rec.embedder_needs[e] <= left for e in emb_classes}
    emb_fitting = [e for e in emb_classes if rec.embedder_fits[e]]
    rec.embedder = emb_fitting[-1] if emb_fitting else emb_classes[0]
    rec.cost_classes = {c: costs.get(c, c) for c in classes}
    return rec

from dataclasses import dataclass
from typing import Optional
import shutil
import subprocess

import psutil


@dataclass
class SystemMetrics:
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]

    total_ram_gb: float
    ram_used_gb: float

    total_vram_gb: Optional[float]
    vram_used_gb: Optional[float]


def get_system_metrics() -> SystemMetrics:
    ram = psutil.virtual_memory()

    gpu_usage_percent = None
    total_vram_gb = None
    vram_used_gb = None

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )

            first_gpu = result.stdout.strip().splitlines()[0]
            gpu_util, mem_used, mem_total = [
                value.strip() for value in first_gpu.split(",")
            ]

            gpu_usage_percent = float(gpu_util)
            vram_used_gb = round(float(mem_used) / 1024, 2)
            total_vram_gb = round(float(mem_total) / 1024, 2)

        except Exception:
            pass

    return SystemMetrics(
        cpu_usage_percent=psutil.cpu_percent(interval=None),
        total_ram_gb=round(ram.total / 1024**3, 2),
        ram_used_gb=round(ram.used / 1024**3, 2),
        gpu_usage_percent=gpu_usage_percent,
        total_vram_gb=total_vram_gb,
        vram_used_gb=vram_used_gb,
    )

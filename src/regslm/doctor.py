"""Environment checks for the compute target (paper §5 "Experimental setup").

Writes ``results/hardware.json``. Never raises: every probe records either a
value or an ``error`` string so the report is always complete.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

VRAM_CAP_MIB = 11 * 1024
WSL_MEM_TARGET_GB = 24
WSL_SWAP_TARGET_GB = 16


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout


def _gpu() -> dict[str, Any]:
    try:
        out = (
            _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ]
            )
            .strip()
            .splitlines()[0]
        )
        name, total, used, driver, cc = (s.strip() for s in out.split(","))
        free = int(total) - int(used)
        return {
            "name": name,
            "vram_total_mib": int(total),
            "vram_used_idle_mib": int(used),
            "vram_free_mib": free,
            "driver": driver,
            "compute_capability": cc,
            "fp8_hardware": float(cc) >= 8.9,
            "vram_cap_mib": VRAM_CAP_MIB,
            "note": "idle usage = Windows compositor; effective budget = min(cap, free)",
        }
    except Exception as e:
        return {"error": repr(e)}


def _mem() -> dict[str, Any]:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.split()[0])
    mem_gb = round(info["MemTotal"] / 1024**2, 1)
    swap_gb = round(info["SwapTotal"] / 1024**2, 1)
    return {
        "mem_total_gb": mem_gb,
        "swap_total_gb": swap_gb,
        "meets_target": mem_gb >= WSL_MEM_TARGET_GB - 1 and swap_gb >= WSL_SWAP_TARGET_GB - 1,
        "target": f"memory={WSL_MEM_TARGET_GB}GB swap={WSL_SWAP_TARGET_GB}GB in .wslconfig",
    }


def _disk(path: Path) -> dict[str, Any]:
    u = shutil.disk_usage(path)
    resolved = str(path.resolve())
    return {
        "path": resolved,
        "free_gb": round(u.free / 1e9, 1),
        "on_windows_fs": resolved.startswith("/mnt/"),
    }


def _cpu() -> dict[str, Any]:
    model = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in Path("/proc/cpuinfo").read_text().splitlines()
            if ln.startswith("model name")
        ),
        "unknown",
    )
    import os

    return {"model": model, "logical_cpus": os.cpu_count()}


def _python_pkgs() -> dict[str, Any]:
    import importlib.metadata as md

    out: dict[str, Any] = {}
    for pkg in [
        "torch",
        "vllm",
        "transformers",
        "trl",
        "peft",
        "unsloth",
        "bitsandbytes",
        "llama_cpp_python",
        "pythainlp",
    ]:
        try:
            out[pkg] = md.version(pkg.replace("_", "-"))
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _vllm_fp8() -> dict[str, Any]:
    """FP8 W8A8 on Ada needs vLLM's cutlass/marlin FP8 kernels (compute capability >= 8.9)."""
    try:
        import vllm  # noqa: F401
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        cc = cap.to_int() if cap is not None else 0
        return {"vllm_supports_fp8_on_this_gpu": cc >= 89, "capability": cc}
    except Exception as e:
        return {"vllm_supports_fp8_on_this_gpu": None, "error": repr(e)}


def run(out: Path = Path("results/hardware.json")) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "gpu": _gpu(),
        "memory": _mem(),
        "disk": _disk(Path.cwd()),
        "hf_cache": _disk(Path.home() / ".cache" / "huggingface"),
        "packages": _python_pkgs(),
        "vllm_fp8": _vllm_fp8(),
        "tools": {t: shutil.which(t) for t in ["tmux", "docker", "gh", "pandoc", "nvcc", "uv"]},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report

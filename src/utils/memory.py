"""GPU memory measurement helpers used by `ddpm verify` and the memory-budget tests.

Methodology follows `specs/001-create-ddpm/research/gpu_memory_probe.py`: the number compared
against the 3 GB budget is the *process total* reported by the driver (CUDA context + allocator
cache), not only the tensors PyTorch tracks.
"""

import os
import shutil
import subprocess

import torch

MIB = 1024**2
CUDA_CONTEXT_ESTIMATE_MIB = 300


def reset_peak(device: torch.device | str | None = None) -> None:
    """Synchronize, release cached blocks and reset PyTorch peak-memory statistics."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_reserved_mib(device: torch.device | str | None = None) -> int:
    """Peak memory reserved by the PyTorch caching allocator, in MiB (0 without CUDA)."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize(device)
    return int(round(torch.cuda.max_memory_reserved(device) / MIB))


def _query(args: list[str]) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None


def process_total_mib() -> int | None:
    """Total GPU memory used by this process as reported by `nvidia-smi`.

    Uses the per-process query first; on platforms where it is unavailable (e.g. WSL2 reports no
    compute apps) falls back to the total used memory of GPU 0. Returns ``None`` without
    `nvidia-smi`.
    """
    out = _query(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if out:
        pid = str(os.getpid())
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                return int(parts[1])
    out = _query(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if out:
        first = out.strip().splitlines()[0].strip()
        if first.isdigit():
            return int(first)
    return None


def measured_total_mib(device: torch.device | str | None = None) -> int:
    """Best available measurement of this process's total GPU memory, in MiB.

    The larger of the driver-reported process total and (peak reserved + CUDA context estimate),
    so that a transient allocator peak is never under-reported.
    """
    estimate = peak_reserved_mib(device) + CUDA_CONTEXT_ESTIMATE_MIB
    reported = process_total_mib()
    return max(estimate, reported) if reported is not None else estimate

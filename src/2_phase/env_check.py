"""GPU environment checks for Phase 2."""
import subprocess

import torch


def check_gpus(expected=2) -> None:
    """Print torch/CUDA info and every GPU; assert CUDA works and ``expected`` GPUs are visible."""
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | available {torch.cuda.is_available()} | GPUs {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory / 1024 ** 3:.1f} GB | compute {p.major}.{p.minor}")
    assert torch.cuda.is_available(), "CUDA is not available"
    assert n == expected, f"expected {expected} GPUs, found {n}"


def nvidia_smi() -> None:
    """Print per-GPU memory use / utilisation to confirm the GPUs are free."""
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                          "--format=csv"], capture_output=True, text=True, check=True).stdout)

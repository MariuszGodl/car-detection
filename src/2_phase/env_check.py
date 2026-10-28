"""GPU environment checks for Phase 2."""
import subprocess

import torch


def check_gpus(expected=2) -> None:
    """Print torch/CUDA info and every GPU; assert CUDA works and ``expected`` GPUs are visible."""
    gpu_count = torch.cuda.device_count()
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"available {torch.cuda.is_available()} | GPUs {gpu_count}")
    for index in range(gpu_count):
        gpu = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {gpu.name} | {gpu.total_memory / 1024 ** 3:.1f} GB | compute {gpu.major}.{gpu.minor}")
    assert torch.cuda.is_available(), "CUDA is not available"
    assert gpu_count == expected, f"expected {expected} GPUs, found {gpu_count}"


def nvidia_smi() -> None:
    """Print per-GPU memory use / utilisation to confirm the GPUs are free."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        capture_output=True, text=True, check=True,
    )
    print(result.stdout)

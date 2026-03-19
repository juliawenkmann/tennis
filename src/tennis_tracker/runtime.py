from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    torch_device: str
    ultralytics_device: str
    use_half: bool


def detect_runtime(*, require_cuda: bool = False) -> RuntimeConfig:
    if torch.cuda.is_available():
        return RuntimeConfig(
            torch_device="cuda",
            ultralytics_device="0",
            use_half=True,
        )

    if require_cuda:
        raise RuntimeError("CUDA GPU required, but no CUDA device is available to PyTorch.")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return RuntimeConfig(
            torch_device="mps",
            ultralytics_device="mps",
            use_half=False,
        )

    return RuntimeConfig(
        torch_device="cpu",
        ultralytics_device="cpu",
        use_half=False,
    )

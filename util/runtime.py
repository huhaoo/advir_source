from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_device(device: str) -> torch.device:
    if not isinstance(device, str) or device.strip() == "":
        raise ValueError(f"device must be a non-empty string, got {device!r}")

    device_name = device.strip().lower()
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        try:
            resolved = torch.device(device_name)
        except Exception as exc:
            raise ValueError(f"invalid CUDA device format: {device}") from exc
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"requested CUDA index {resolved.index} out of range (device_count={torch.cuda.device_count()})"
            )
        return resolved
    raise ValueError(f"unsupported device: {device!r}. use one of: cpu, cuda, cuda:N, auto")

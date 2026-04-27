from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor


def ensure_bchw(image: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
    if image.ndim == 3:
        if image.shape[0] not in {1, 3}:
            raise ValueError(f"3D image must be (C,H,W) with C in {{1,3}}, got {tuple(image.shape)}")
        return image.unsqueeze(0), True
    if image.ndim == 4:
        return image, False
    raise ValueError(f"image must be (C,H,W) or (B,C,H,W), got {tuple(image.shape)}")


def load_rgb_tensor(path: Path) -> Tensor:
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_uint8_hwc(image: Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image_np * 255.0 + 0.5).astype(np.uint8)


def save_rgb_tensor(image: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_hwc(image), mode="RGB").save(path)

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


allowed_interp_modes = {"nearest", "bilinear", "bicubic", "area", "gaussian"}


@dataclass
class InterpolationFieldConfig:
    low_res_height: int
    low_res_width: int
    high_res_height: int
    high_res_width: int
    mode: str = "bicubic"
    align_corners: bool | None = False
    gaussian_radius: int = 4
    gaussian_sigma: float = 0.47
    gaussian_extra_cells: int = 3


class control_map_interpolator(nn.Module):
    """Unified lowres->highres interpolation field module.

    Supported modes:
    - nearest / area: delegated to torch.interpolate
    - bilinear / bicubic: delegated to torch.interpolate with align_corners
    - gaussian: custom low-res grid weighted interpolation
    """

    def __init__(self, config: InterpolationFieldConfig) -> None:
        super().__init__()
        if not isinstance(config, InterpolationFieldConfig):
            raise ValueError(f"config must be InterpolationFieldConfig, got {type(config)!r}")

        if config.mode not in allowed_interp_modes:
            raise ValueError(f"mode must be one of {allowed_interp_modes}, got {config.mode!r}")
        self._validate_positive_int(config.low_res_height, "low_res_height")
        self._validate_positive_int(config.low_res_width, "low_res_width")
        self._validate_positive_int(config.high_res_height, "high_res_height")
        self._validate_positive_int(config.high_res_width, "high_res_width")

        if config.align_corners is not None and not isinstance(config.align_corners, bool):
            raise ValueError(f"align_corners must be bool or None, got {type(config.align_corners)!r}")
        self._validate_positive_int(config.gaussian_radius, "gaussian_radius")
        self._validate_non_negative_int(config.gaussian_extra_cells, "gaussian_extra_cells")

        if not isinstance(config.gaussian_sigma, (int, float)):
            raise ValueError(f"gaussian_sigma must be numeric, got {type(config.gaussian_sigma)!r}")
        if float(config.gaussian_sigma) <= 0:
            raise ValueError(f"gaussian_sigma must be > 0, got {config.gaussian_sigma}")

        self.low_res_height = int(config.low_res_height)
        self.low_res_width = int(config.low_res_width)
        self.high_res_height = int(config.high_res_height)
        self.high_res_width = int(config.high_res_width)

        self.mode = str(config.mode)
        self.align_corners = config.align_corners
        self.gaussian_radius = int(config.gaussian_radius)
        self.gaussian_sigma = float(config.gaussian_sigma)
        self.gaussian_extra_cells = int(config.gaussian_extra_cells)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    @staticmethod
    def _validate_non_negative_int(value: int, name: str) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value}")

    def _axis_coordinates(self, in_size: int, out_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if bool(self.align_corners) and out_size > 1:
            return torch.linspace(0.0, float(in_size - 1), steps=out_size, device=device, dtype=dtype)
        idx = torch.arange(out_size, device=device, dtype=dtype)
        return (idx + 0.5) * float(in_size) / float(out_size) - 0.5

    def _build_axis_weights(self, in_size: int, out_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build normalized axis weights [out_size, in_size] using boundary-extended Gaussian support.

        We first gather a virtual local window centered at each output coordinate:
        base_window = gaussian_radius
        extension = gaussian_extra_cells
        total_window = base_window + extension

        Virtual indices outside [0, in_size-1] are clamped back to the nearest
        valid index. This explicitly enables boundary extension behavior.
        """
        coords = self._axis_coordinates(in_size=in_size, out_size=out_size, device=device, dtype=dtype)
        total_window = self.gaussian_radius + self.gaussian_extra_cells
        offsets = torch.arange(-total_window, total_window + 1, device=device, dtype=torch.long)

        centers = torch.floor(coords).to(torch.long)
        virtual_indices = centers[:, None] + offsets[None, :]
        clamped_indices = virtual_indices.clamp(0, in_size - 1)

        distances = virtual_indices.to(dtype=dtype) - coords[:, None]
        sigma = torch.tensor(self.gaussian_sigma, device=device, dtype=dtype)
        local_weights = torch.exp(-0.5 * (distances / sigma) ** 2)

        axis_weights = torch.zeros(out_size, in_size, device=device, dtype=dtype)
        axis_weights.scatter_add_(dim=1, index=clamped_indices, src=local_weights)
        axis_weights = axis_weights / axis_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return axis_weights

    def _gaussian_interpolate(self, x: Tensor) -> Tensor:
        b, c, lh, lw = x.shape
        if lh != self.low_res_height or lw != self.low_res_width:
            raise ValueError(
                "gaussian interpolation input low-res shape mismatch: "
                f"expected ({self.low_res_height}, {self.low_res_width}), got ({lh}, {lw})"
            )

        h, w = self.high_res_height, self.high_res_width
        wy = self._build_axis_weights(in_size=lh, out_size=h, device=x.device, dtype=x.dtype)
        wx = self._build_axis_weights(in_size=lw, out_size=w, device=x.device, dtype=x.dtype)

        # Separable interpolation: first along height, then width.
        tmp = torch.einsum("hy,bcyx->bchx", wy, x)
        return torch.einsum("wx,bchx->bchw", wx, tmp)

    def forward(self, low_res_field: Tensor) -> Tensor:
        if not isinstance(low_res_field, torch.Tensor):
            raise ValueError(f"low_res_field must be torch.Tensor, got {type(low_res_field)!r}")
        if low_res_field.ndim != 4:
            raise ValueError(f"low_res_field must have shape (B, C, H, W), got {tuple(low_res_field.shape)}")

        _, _, lh, lw = low_res_field.shape
        if lh != self.low_res_height or lw != self.low_res_width:
            raise ValueError(
                "low_res_field spatial mismatch: "
                f"expected ({self.low_res_height}, {self.low_res_width}), got ({lh}, {lw})"
            )

        size = (self.high_res_height, self.high_res_width)
        if self.mode in {"bilinear", "bicubic"}:
            return F.interpolate(low_res_field, size=size, mode=self.mode, align_corners=self.align_corners)
        if self.mode == "gaussian":
            return self._gaussian_interpolate(low_res_field)
        return F.interpolate(low_res_field, size=size, mode=self.mode)


def _self_test() -> None:
    print("[control_map_interpolator] self-test start")

    x = torch.ones(1, 1, 4, 4)

    cfg_bicubic = InterpolationFieldConfig(4, 4, 8, 8, mode="bicubic", align_corners=False)
    y_bicubic = control_map_interpolator(cfg_bicubic)(x)
    assert y_bicubic.shape == (1, 1, 8, 8)
    assert torch.allclose(y_bicubic, torch.ones_like(y_bicubic), atol=1e-6)

    cfg_gaussian = InterpolationFieldConfig(
        4,
        4,
        8,
        8,
        mode="gaussian",
        gaussian_radius=3,
        gaussian_sigma=1.25,
        gaussian_extra_cells=1,
    )
    y_gaussian = control_map_interpolator(cfg_gaussian)(x)
    assert y_gaussian.shape == (1, 1, 8, 8)
    assert torch.allclose(y_gaussian, torch.ones_like(y_gaussian), atol=1e-6)

    print("[control_map_interpolator] self-test passed")


if __name__ == "__main__":
    _self_test()

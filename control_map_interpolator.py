from __future__ import annotations

from dataclasses import dataclass
import math

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
    gaussian_enable_offset: bool = False
    gaussian_offset_max: float = 0.5


class control_map_interpolator(nn.Module):
    """Unified lowres->highres interpolation field module.

    Supported modes:
    - nearest / area: delegated to torch.interpolate
    - bilinear / bicubic: delegated to torch.interpolate with align_corners
    - gaussian: custom low-res grid weighted interpolation
      optional per-kernel center offset can be provided as [B|1, 2, H_low, W_low]
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
        if not isinstance(config.gaussian_enable_offset, bool):
            raise ValueError(
                "gaussian_enable_offset must be bool, "
                f"got {type(config.gaussian_enable_offset)!r}"
            )
        if not isinstance(config.gaussian_offset_max, (int, float)):
            raise ValueError(f"gaussian_offset_max must be numeric, got {type(config.gaussian_offset_max)!r}")
        if float(config.gaussian_offset_max) < 0:
            raise ValueError(f"gaussian_offset_max must be >= 0, got {config.gaussian_offset_max}")

        self.low_res_height = int(config.low_res_height)
        self.low_res_width = int(config.low_res_width)
        self.high_res_height = int(config.high_res_height)
        self.high_res_width = int(config.high_res_width)

        self.mode = str(config.mode)
        self.align_corners = config.align_corners
        self.gaussian_radius = int(config.gaussian_radius)
        self.gaussian_sigma = float(config.gaussian_sigma)
        self.gaussian_extra_cells = int(config.gaussian_extra_cells)
        self.gaussian_enable_offset = bool(config.gaussian_enable_offset)
        self.gaussian_offset_max = float(config.gaussian_offset_max)

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

    def _gaussian_interpolate_with_offsets(self, x: Tensor, gaussian_offset_xy: Tensor) -> Tensor:
        b, c, lh, lw = x.shape
        h, w = self.high_res_height, self.high_res_width
        device, dtype = x.device, x.dtype

        if gaussian_offset_xy.ndim != 4:
            raise ValueError(
                "gaussian_offset_xy must have shape (B|1, 2, H_low, W_low), "
                f"got {tuple(gaussian_offset_xy.shape)}"
            )
        if gaussian_offset_xy.shape[1] != 2:
            raise ValueError(
                "gaussian_offset_xy channel dim must be 2 (x/y), "
                f"got {gaussian_offset_xy.shape[1]}"
            )
        if gaussian_offset_xy.shape[2] != lh or gaussian_offset_xy.shape[3] != lw:
            raise ValueError(
                "gaussian_offset_xy spatial mismatch: "
                f"expected ({lh}, {lw}), got ({gaussian_offset_xy.shape[2]}, {gaussian_offset_xy.shape[3]})"
            )
        if gaussian_offset_xy.shape[0] not in {1, b}:
            raise ValueError(
                "gaussian_offset_xy batch must be 1 or match input batch, "
                f"got offset_batch={gaussian_offset_xy.shape[0]}, input_batch={b}"
            )

        offset_xy = gaussian_offset_xy.to(device=device, dtype=dtype)
        if offset_xy.shape[0] == 1 and b > 1:
            offset_xy = offset_xy.expand(b, -1, -1, -1)
        offset_x = offset_xy[:, 0, :, :].reshape(b, lh * lw)
        offset_y = offset_xy[:, 1, :, :].reshape(b, lh * lw)

        coords_y = self._axis_coordinates(in_size=lh, out_size=h, device=device, dtype=dtype)
        coords_x = self._axis_coordinates(in_size=lw, out_size=w, device=device, dtype=dtype)

        offset_margin = int(math.ceil(self.gaussian_offset_max))
        total_window = self.gaussian_radius + self.gaussian_extra_cells + offset_margin
        virtual_offsets = torch.arange(-total_window, total_window + 1, device=device, dtype=torch.long)

        centers_y = torch.floor(coords_y).to(torch.long)
        centers_x = torch.floor(coords_x).to(torch.long)
        virtual_iy = centers_y[:, None] + virtual_offsets[None, :]
        virtual_ix = centers_x[:, None] + virtual_offsets[None, :]
        clamped_iy = virtual_iy.clamp(0, lh - 1)
        clamped_ix = virtual_ix.clamp(0, lw - 1)

        x_flat = x.reshape(b, c, lh * lw)

        out = torch.zeros((b, c, h, w), device=device, dtype=dtype)
        denom = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        sigma = torch.tensor(self.gaussian_sigma, device=device, dtype=dtype)
        inv_two_sigma_sq = 0.5 / (sigma * sigma)

        if w >= 1024:
            chunk_rows = 16
        elif w >= 512:
            chunk_rows = 32
        else:
            chunk_rows = 64

        virtual_ix_f = virtual_ix.to(dtype=dtype).view(1, 1, 1, 1, w, -1)
        coords_x_f = coords_x.view(1, 1, 1, 1, w, 1)

        for y_start in range(0, h, chunk_rows):
            y_end = min(h, y_start + chunk_rows)
            chunk_h = y_end - y_start

            virtual_iy_chunk = virtual_iy[y_start:y_end]
            clamped_iy_chunk = clamped_iy[y_start:y_end]
            coords_y_chunk = coords_y[y_start:y_end]

            flat_indices = (
                clamped_iy_chunk[:, :, None, None] * lw
                + clamped_ix[None, None, :, :]
            ).reshape(-1)

            patch_vals = x_flat.index_select(dim=2, index=flat_indices).reshape(b, c, chunk_h, -1, w, virtual_ix.shape[1])
            patch_off_x = offset_x.index_select(dim=1, index=flat_indices).reshape(b, 1, chunk_h, -1, w, virtual_ix.shape[1])
            patch_off_y = offset_y.index_select(dim=1, index=flat_indices).reshape(b, 1, chunk_h, -1, w, virtual_ix.shape[1])

            virtual_iy_f = virtual_iy_chunk.to(dtype=dtype).view(1, 1, chunk_h, -1, 1, 1)
            coords_y_f = coords_y_chunk.view(1, 1, chunk_h, 1, 1, 1)

            center_y = virtual_iy_f + patch_off_y
            center_x = virtual_ix_f + patch_off_x
            dist2 = (coords_y_f - center_y) ** 2 + (coords_x_f - center_x) ** 2
            weights = torch.exp(-dist2 * inv_two_sigma_sq)

            out[:, :, y_start:y_end, :] = (patch_vals * weights).sum(dim=(3, 5))
            denom[:, :, y_start:y_end, :] = weights.sum(dim=(3, 5))

        return out / denom.clamp_min(1e-12)

    def _gaussian_interpolate(self, x: Tensor, gaussian_offset_xy: Tensor | None = None) -> Tensor:
        b, c, lh, lw = x.shape
        if lh != self.low_res_height or lw != self.low_res_width:
            raise ValueError(
                "gaussian interpolation input low-res shape mismatch: "
                f"expected ({self.low_res_height}, {self.low_res_width}), got ({lh}, {lw})"
            )

        if gaussian_offset_xy is not None:
            if not self.gaussian_enable_offset:
                raise ValueError(
                    "gaussian_offset_xy is provided but gaussian_enable_offset is False in config"
                )
            return self._gaussian_interpolate_with_offsets(x=x, gaussian_offset_xy=gaussian_offset_xy)

        h, w = self.high_res_height, self.high_res_width
        wy = self._build_axis_weights(in_size=lh, out_size=h, device=x.device, dtype=x.dtype)
        wx = self._build_axis_weights(in_size=lw, out_size=w, device=x.device, dtype=x.dtype)

        # Separable interpolation: first along height, then width.
        tmp = torch.einsum("hy,bcyx->bchx", wy, x)
        return torch.einsum("wx,bchx->bchw", wx, tmp)

    def forward(self, low_res_field: Tensor, gaussian_offset_xy: Tensor | None = None) -> Tensor:
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
            return self._gaussian_interpolate(low_res_field, gaussian_offset_xy=gaussian_offset_xy)
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

    cfg_gaussian_offset = InterpolationFieldConfig(
        4,
        4,
        8,
        8,
        mode="gaussian",
        gaussian_radius=3,
        gaussian_sigma=1.25,
        gaussian_extra_cells=1,
        gaussian_enable_offset=True,
        gaussian_offset_max=0.5,
    )
    gaussian_with_offset = control_map_interpolator(cfg_gaussian_offset)
    zero_offset = torch.zeros((1, 2, 4, 4))
    y_gaussian_offset = gaussian_with_offset(x, gaussian_offset_xy=zero_offset)
    assert y_gaussian_offset.shape == (1, 1, 8, 8)
    assert torch.allclose(y_gaussian_offset, torch.ones_like(y_gaussian_offset), atol=1e-6)

    x_rand = torch.randn((1, 1, 4, 4))
    learnable_offset = torch.zeros((1, 2, 4, 4), requires_grad=True)
    y_rand = gaussian_with_offset(x_rand, gaussian_offset_xy=learnable_offset)
    y_rand.mean().backward()
    assert learnable_offset.grad is not None
    assert torch.isfinite(learnable_offset.grad).all()

    print("[control_map_interpolator] self-test passed")


if __name__ == "__main__":
    _self_test()

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from control_map import ControlMapConfig, control_map


@dataclass
class HazeDegradationConfig:
    map_config: ControlMapConfig
    airlight_init: float | tuple[float, float, float] = 1.0
    beta_mean: float = 1.0
    beta_std: float | None = None
    min_beta: float = 1e-6


@dataclass
class MotionBlurDegradationConfig:
    map_config: ControlMapConfig
    num_steps: int = 16
    mode: str = "bilinear"
    padding_mode: str = "border"
    align_corners: bool = True
    batchify_steps: bool = True
    dmax: float | None = None
    dlambda: float = 0.0


class motion_blur_degradation(nn.Module):
    """Differentiable per-pixel motion blur with trainable dx map."""

    def __init__(self, config: MotionBlurDegradationConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MotionBlurDegradationConfig(
                map_config=ControlMapConfig(8, 8, 128, 128),
            )
        if not isinstance(config, MotionBlurDegradationConfig):
            raise ValueError(f"config must be MotionBlurDegradationConfig, got {type(config)!r}")
        if not isinstance(config.map_config, ControlMapConfig):
            raise ValueError(f"map_config must be ControlMapConfig, got {type(config.map_config)!r}")

        if not isinstance(config.num_steps, int) or config.num_steps <= 0:
            raise ValueError(f"num_steps must be positive int, got {config.num_steps}")
        if config.mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(f"mode must be one of {{nearest, bilinear, bicubic}}, got {config.mode!r}")
        if config.padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(
                f"padding_mode must be one of {{zeros, border, reflection}}, got {config.padding_mode!r}"
            )
        if not isinstance(config.align_corners, bool):
            raise ValueError(f"align_corners must be bool, got {type(config.align_corners)!r}")
        if not isinstance(config.batchify_steps, bool):
            raise ValueError(f"batchify_steps must be bool, got {type(config.batchify_steps)!r}")
        if config.dmax is not None:
            if not isinstance(config.dmax, (int, float)):
                raise ValueError(f"dmax must be numeric or None, got {type(config.dmax)!r}")
            if not math.isfinite(float(config.dmax)):
                raise ValueError(f"dmax must be finite numeric or None, got {config.dmax}")
        if not isinstance(config.dlambda, (int, float)):
            raise ValueError(f"dlambda must be numeric, got {type(config.dlambda)!r}")
        if float(config.dlambda) < 0:
            raise ValueError(f"dlambda must be >= 0, got {config.dlambda}")

        self.num_steps = int(config.num_steps)
        self.mode = str(config.mode)
        self.padding_mode = str(config.padding_mode)
        self.align_corners = bool(config.align_corners)
        self.batchify_steps = bool(config.batchify_steps)
        self.dmax = None if config.dmax is None else float(config.dmax)
        self.dlambda = float(config.dlambda)

        self.dx_map_module = control_map(config.map_config)
        self.dy_map_module = control_map(config.map_config)

    def reset_parameters(self) -> None:
        self.dx_map_module.reset_parameters()
        self.dy_map_module.reset_parameters()

    def project_trainable_parameters_(self) -> None:
        self.dx_map_module.project_trainable_parameters_()
        self.dy_map_module.project_trainable_parameters_()
        effective_dmax = self._effective_dmax_for_hw(
            h=self.dx_map_module.high_res_height,
            w=self.dx_map_module.high_res_width,
        )
        if effective_dmax is not None:
            with torch.no_grad():
                self.dx_map_module.get_low_res_map().clamp_(-effective_dmax, effective_dmax)
                self.dy_map_module.get_low_res_map().clamp_(-effective_dmax, effective_dmax)

    def _effective_dmax_for_hw(self, h: int, w: int) -> float | None:
        if self.dmax is None:
            return None
        dmax = float(self.dmax)
        if dmax < 0:
            diagonal = math.sqrt(float(h * h + w * w))
            return (-dmax) * diagonal
        return dmax

    def _check_image(self, image: Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
        if image.ndim != 4:
            raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
        if not torch.is_floating_point(image):
            raise ValueError(f"image must be floating tensor, got dtype={image.dtype}")

    def _check_internal_dx_spatial(self, image: Tensor) -> None:
        _, _, h, w = image.shape
        if h != self.dx_map_module.high_res_height or w != self.dx_map_module.high_res_width:
            raise ValueError(
                "image spatial size must match motion blur map size: "
                f"expected ({self.dx_map_module.high_res_height}, {self.dx_map_module.high_res_width}), got ({h}, {w})"
            )

    def _check_optional_dx(self, dx: Tensor, image: Tensor) -> Tensor:
        if not isinstance(dx, torch.Tensor):
            raise ValueError(f"dx must be torch.Tensor, got {type(dx)!r}")
        if dx.ndim != 4:
            raise ValueError(f"dx must have shape (B, 2, H, W), got {tuple(dx.shape)}")
        if not torch.is_floating_point(dx):
            raise ValueError(f"dx must be floating tensor, got dtype={dx.dtype}")

        b, _, h, w = image.shape
        dx_b, dx_c, dx_h, dx_w = dx.shape
        if dx_c != 2:
            raise ValueError(f"dx channel dimension must be 2 ([dx, dy]), got {dx_c}")
        if dx_h != h or dx_w != w:
            raise ValueError(f"dx spatial size must match image: expected ({h}, {w}), got ({dx_h}, {dx_w})")

        if dx_b == 1 and b > 1:
            dx = dx.expand(b, -1, -1, -1)
        elif dx_b != b:
            raise ValueError(f"dx batch mismatch: expected {b} or 1, got {dx_b}")

        return dx

    def get_dx_map(self) -> Tensor:
        dx = self.dx_map_module()
        dy = self.dy_map_module()
        flow = torch.cat((dx, dy), dim=1)
        effective_dmax = self._effective_dmax_for_hw(
            h=self.dx_map_module.high_res_height,
            w=self.dx_map_module.high_res_width,
        )
        if effective_dmax is not None:
            flow = flow.clamp(-effective_dmax, effective_dmax)
        return flow

    def _pixel_to_normalized(self, x: Tensor, size: int) -> Tensor:
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(x)
            return x * (2.0 / float(size - 1)) - 1.0
        return (2.0 * (x + 0.5) / float(size)) - 1.0

    def _normalize_sampling_grid(self, sample_xy: Tensor, h: int, w: int) -> Tensor:
        x = self._pixel_to_normalized(sample_xy[..., 0], w)
        y = self._pixel_to_normalized(sample_xy[..., 1], h)
        return torch.stack((x, y), dim=-1)

    def _build_base_xy(self, h: int, w: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        ys = torch.arange(h, dtype=dtype, device=device)
        xs = torch.arange(w, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1).unsqueeze(0)

    def _forward_batchified(self, image: Tensor, dx: Tensor) -> Tensor:
        b, c, h, w = image.shape
        base_xy = self._build_base_xy(h=h, w=w, dtype=image.dtype, device=image.device)
        dx_xy = dx.permute(0, 2, 3, 1).contiguous()

        t = (torch.arange(self.num_steps, dtype=image.dtype, device=image.device) + 0.5) / float(self.num_steps) - 0.5
        sample_xy = base_xy.unsqueeze(0) + t.view(self.num_steps, 1, 1, 1, 1) * dx_xy.unsqueeze(0)
        sample_xy = sample_xy.reshape(self.num_steps * b, h, w, 2)
        sample_grid = self._normalize_sampling_grid(sample_xy=sample_xy, h=h, w=w)

        image_batchified = image.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1).reshape(self.num_steps * b, c, h, w)
        sampled = F.grid_sample(
            input=image_batchified,
            grid=sample_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return sampled.view(self.num_steps, b, c, h, w).mean(dim=0)

    def _forward_loop(self, image: Tensor, dx: Tensor) -> Tensor:
        _, _, h, w = image.shape
        base_xy = self._build_base_xy(h=h, w=w, dtype=image.dtype, device=image.device)
        dx_xy = dx.permute(0, 2, 3, 1).contiguous()

        output = torch.zeros_like(image)
        t = (torch.arange(self.num_steps, dtype=image.dtype, device=image.device) + 0.5) / float(self.num_steps) - 0.5
        for t_i in t:
            sample_xy = base_xy + t_i * dx_xy
            sample_grid = self._normalize_sampling_grid(sample_xy=sample_xy, h=h, w=w)
            sampled = F.grid_sample(
                input=image,
                grid=sample_grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            output = output + sampled
        return output / float(self.num_steps)

    def forward(self, image: Tensor, dx: Tensor | None = None) -> Tensor:
        self._check_image(image=image)
        if dx is None:
            self._check_internal_dx_spatial(image=image)
            dx = self.get_dx_map()
            if image.shape[0] > 1:
                dx = dx.expand(image.shape[0], -1, -1, -1)
        else:
            dx = self._check_optional_dx(dx=dx, image=image)
        effective_dmax = self._effective_dmax_for_hw(h=int(image.shape[-2]), w=int(image.shape[-1]))
        if effective_dmax is not None:
            dx = dx.clamp(-effective_dmax, effective_dmax)
        if dx.dtype != image.dtype:
            dx = dx.to(image.dtype)
        if dx.device != image.device:
            dx = dx.to(image.device)

        if self.batchify_steps:
            return self._forward_batchified(image=image, dx=dx)
        return self._forward_loop(image=image, dx=dx)

    def get_regularization_loss(self) -> Tensor:
        loss_map = self.dx_map_module.regularization_loss() + self.dy_map_module.regularization_loss()
        if self.dlambda <= 0:
            return loss_map
        dx = self.get_dx_map()
        loss_dx_magnitude = (dx[:, 0:1] ** 2 + dx[:, 1:2] ** 2).mean()
        return loss_map + self.dlambda * loss_dx_magnitude

    def get_regularization_items(self) -> dict[str, Tensor]:
        items_x = self.dx_map_module.regularization_items()
        items_y = self.dy_map_module.regularization_items()
        loss_map = items_x["loss_total"] + items_y["loss_total"]
        dx = self.get_dx_map()
        loss_dx_magnitude = (dx[:, 0:1] ** 2 + dx[:, 1:2] ** 2).mean()
        loss_dx_total = self.dlambda * loss_dx_magnitude
        return {
            "loss_map_x": items_x["loss_total"],
            "loss_map_y": items_y["loss_total"],
            "loss_map_total": loss_map,
            "loss_dx_magnitude": loss_dx_magnitude,
            "loss_dx_total": loss_dx_total,
            "loss_total": loss_map + loss_dx_total,
        }


def _extract_chw_from_image(image: Tensor) -> tuple[int, int, int]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
    if image.ndim == 4:
        if image.shape[0] < 1:
            raise ValueError("image batch must be non-empty")
        _, c, h, w = image.shape
    elif image.ndim == 3:
        c, h, w = image.shape
    else:
        raise ValueError(f"image must have shape (B, C, H, W) or (C, H, W), got {tuple(image.shape)}")
    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"invalid image shape: (C, H, W)=({c}, {h}, {w})")
    return int(c), int(h), int(w)


def random_haze_degradation_config_from_image(
    image: Tensor,
    interp_mode: str = "bicubic",
    gaussian_radius: int = 4,
    gaussian_sigma: float = 1.25,
    gaussian_extra_cells: int = 2,
    gaussian_enable_offset: bool = False,
    gaussian_offset_max: float = 0.5,
    gaussian_offset_lambda_first_order: float = 5e-2,
    gaussian_offset_lambda_second_order: float = 2e-1,
    airlight_min: float = 0.85,
    airlight_max: float = 1.0,
    airlight_jitter: float = 0.02,
    beta_mean_min: float = 0.1,
    beta_mean_max: float = 0.5,
    beta_mean_log_uniform: bool = False,
) -> HazeDegradationConfig:
    """Sample one random haze degradation config from an input image."""
    _, h, w = _extract_chw_from_image(image)

    interp_mode = str(interp_mode).strip().lower()
    if interp_mode not in {"nearest", "bilinear", "bicubic", "area", "gaussian"}:
        raise ValueError(
            "interp_mode must be one of "
            "{nearest, bilinear, bicubic, area, gaussian}, "
            f"got {interp_mode!r}"
        )
    if not isinstance(gaussian_radius, int) or gaussian_radius <= 0:
        raise ValueError(f"gaussian_radius must be positive int, got {gaussian_radius}")
    if not isinstance(gaussian_sigma, (int, float)) or float(gaussian_sigma) <= 0:
        raise ValueError(f"gaussian_sigma must be positive numeric, got {gaussian_sigma}")
    if not isinstance(gaussian_extra_cells, int) or gaussian_extra_cells < 0:
        raise ValueError(f"gaussian_extra_cells must be non-negative int, got {gaussian_extra_cells}")
    if not isinstance(gaussian_enable_offset, bool):
        raise ValueError(f"gaussian_enable_offset must be bool, got {type(gaussian_enable_offset)!r}")
    if not isinstance(gaussian_offset_max, (int, float)) or float(gaussian_offset_max) < 0:
        raise ValueError(f"gaussian_offset_max must be numeric and >= 0, got {gaussian_offset_max}")
    if not isinstance(gaussian_offset_lambda_first_order, (int, float)):
        raise ValueError(
            "gaussian_offset_lambda_first_order must be numeric, "
            f"got {type(gaussian_offset_lambda_first_order)!r}"
        )
    if not isinstance(gaussian_offset_lambda_second_order, (int, float)):
        raise ValueError(
            "gaussian_offset_lambda_second_order must be numeric, "
            f"got {type(gaussian_offset_lambda_second_order)!r}"
        )
    if not isinstance(airlight_min, (int, float)):
        raise ValueError(f"airlight_min must be numeric, got {type(airlight_min)!r}")
    if not isinstance(airlight_max, (int, float)):
        raise ValueError(f"airlight_max must be numeric, got {type(airlight_max)!r}")
    if float(airlight_min) > float(airlight_max):
        raise ValueError(f"airlight_min must be <= airlight_max, got {(airlight_min, airlight_max)}")
    if not isinstance(airlight_jitter, (int, float)) or float(airlight_jitter) < 0:
        raise ValueError(f"airlight_jitter must be numeric and >= 0, got {airlight_jitter}")
    if not isinstance(beta_mean_min, (int, float)):
        raise ValueError(f"beta_mean_min must be numeric, got {type(beta_mean_min)!r}")
    if not isinstance(beta_mean_max, (int, float)):
        raise ValueError(f"beta_mean_max must be numeric, got {type(beta_mean_max)!r}")
    if float(beta_mean_min) > float(beta_mean_max):
        raise ValueError(f"beta_mean_min must be <= beta_mean_max, got {(beta_mean_min, beta_mean_max)}")
    if not isinstance(beta_mean_log_uniform, bool):
        raise ValueError(f"beta_mean_log_uniform must be bool, got {type(beta_mean_log_uniform)!r}")
    if bool(beta_mean_log_uniform) and float(beta_mean_min) <= 0:
        raise ValueError(
            "beta_mean_min must be > 0 when beta_mean_log_uniform is enabled, "
            f"got {beta_mean_min}"
        )

    airlight_min_f = float(airlight_min)
    airlight_max_f = float(airlight_max)
    airlight_jitter_f = float(airlight_jitter)
    if airlight_max_f == airlight_min_f:
        airlight_base = airlight_min_f
    else:
        airlight_base = airlight_min_f + (airlight_max_f - airlight_min_f) * float(torch.rand(1).item())
    airlight = []
    for _ in range(3):
        jitter = (float(torch.rand(1).item()) * 2.0 - 1.0) * airlight_jitter_f
        airlight.append(min(max(airlight_base + jitter, airlight_min_f), airlight_max_f))
    airlight_tuple = (float(airlight[0]), float(airlight[1]), float(airlight[2]))

    beta_mean_min_f = float(beta_mean_min)
    beta_mean_max_f = float(beta_mean_max)
    if beta_mean_max_f == beta_mean_min_f:
        beta_mean = beta_mean_min_f
    elif bool(beta_mean_log_uniform):
        log_lo = math.log(beta_mean_min_f)
        log_hi = math.log(beta_mean_max_f)
        beta_mean = float(math.exp(log_lo + (log_hi - log_lo) * float(torch.rand(1).item())))
    else:
        beta_mean = beta_mean_min_f + (beta_mean_max_f - beta_mean_min_f) * float(torch.rand(1).item())
    beta_std = beta_mean * (0.05 + 0.25 * float(torch.rand(1).item()))

    haze_lh, haze_lw = max(4, min(16, h)), max(4, min(16, w))
    return HazeDegradationConfig(
        map_config=ControlMapConfig(
            low_res_height=haze_lh,
            low_res_width=haze_lw,
            high_res_height=int(h),
            high_res_width=int(w),
            interp_mode=interp_mode,
            align_corners=False,
            gaussian_radius=gaussian_radius,
            gaussian_sigma=gaussian_sigma,
            gaussian_extra_cells=gaussian_extra_cells,
            gaussian_enable_offset=gaussian_enable_offset,
            gaussian_offset_max=gaussian_offset_max,
            lambda_first_order=0.1,
            lambda_second_order=0.5,
            lambda_offset_first_order=float(gaussian_offset_lambda_first_order),
            lambda_offset_second_order=float(gaussian_offset_lambda_second_order),
            init_mode="normal",
            init_value=0.0,
            init_scale=0.1,
        ),
        airlight_init=airlight_tuple,
        beta_mean=float(beta_mean),
        beta_std=float(beta_std),
        min_beta=1e-3,
    )


def random_motion_blur_degradation_config_from_image(
    image: Tensor,
    num_steps: int = 16,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
    batchify_steps: bool = True,
    dmax: float | None = -0.02,
    dlambda: float = 0.0,
    interp_mode: str = "bicubic",
    gaussian_radius: int = 4,
    gaussian_sigma: float = 1.25,
    gaussian_extra_cells: int = 2,
    gaussian_enable_offset: bool = False,
    gaussian_offset_max: float = 0.5,
    gaussian_offset_lambda_first_order: float = 5e-2,
    gaussian_offset_lambda_second_order: float = 2e-1,
) -> MotionBlurDegradationConfig:
    """Sample one motion blur degradation config from an input image.

    When `dmax` is not None, this sampler draws the actual `cfg.dmax`
    uniformly from `[dmax/2, dmax]` (order-insensitive for negative values).
    """
    _, h, w = _extract_chw_from_image(image)

    interp_mode = str(interp_mode).strip().lower()
    if interp_mode not in {"nearest", "bilinear", "bicubic", "area", "gaussian"}:
        raise ValueError(
            "interp_mode must be one of "
            "{nearest, bilinear, bicubic, area, gaussian}, "
            f"got {interp_mode!r}"
        )
    if not isinstance(gaussian_radius, int) or gaussian_radius <= 0:
        raise ValueError(f"gaussian_radius must be positive int, got {gaussian_radius}")
    if not isinstance(gaussian_sigma, (int, float)) or float(gaussian_sigma) <= 0:
        raise ValueError(f"gaussian_sigma must be positive numeric, got {gaussian_sigma}")
    if not isinstance(gaussian_extra_cells, int) or gaussian_extra_cells < 0:
        raise ValueError(f"gaussian_extra_cells must be non-negative int, got {gaussian_extra_cells}")
    if not isinstance(gaussian_enable_offset, bool):
        raise ValueError(f"gaussian_enable_offset must be bool, got {type(gaussian_enable_offset)!r}")
    if not isinstance(gaussian_offset_max, (int, float)) or float(gaussian_offset_max) < 0:
        raise ValueError(f"gaussian_offset_max must be numeric and >= 0, got {gaussian_offset_max}")
    if not isinstance(gaussian_offset_lambda_first_order, (int, float)):
        raise ValueError(
            "gaussian_offset_lambda_first_order must be numeric, "
            f"got {type(gaussian_offset_lambda_first_order)!r}"
        )
    if not isinstance(gaussian_offset_lambda_second_order, (int, float)):
        raise ValueError(
            "gaussian_offset_lambda_second_order must be numeric, "
            f"got {type(gaussian_offset_lambda_second_order)!r}"
        )

    sampled_dmax: float | None
    if dmax is None:
        sampled_dmax = None
    else:
        dmax_val = float(dmax)
        dmax_half = 0.5 * dmax_val
        dmax_lo = min(dmax_half, dmax_val)
        dmax_hi = max(dmax_half, dmax_val)
        if dmax_hi == dmax_lo:
            sampled_dmax = dmax_val
        else:
            sampled_dmax = float(dmax_lo + (dmax_hi - dmax_lo) * float(torch.rand(1).item()))

    motion_lh, motion_lw = max(4, min(16, h)), max(4, min(16, w))
    return MotionBlurDegradationConfig(
        map_config=ControlMapConfig(
            low_res_height=motion_lh,
            low_res_width=motion_lw,
            high_res_height=int(h),
            high_res_width=int(w),
            interp_mode=interp_mode,
            align_corners=False,
            gaussian_radius=gaussian_radius,
            gaussian_sigma=gaussian_sigma,
            gaussian_extra_cells=gaussian_extra_cells,
            gaussian_enable_offset=gaussian_enable_offset,
            gaussian_offset_max=gaussian_offset_max,
            lambda_offset_first_order=float(gaussian_offset_lambda_first_order),
            lambda_offset_second_order=float(gaussian_offset_lambda_second_order),
            init_mode="normal",
            init_value=0.0,
            init_scale=0.1,
        ),
        num_steps=int(num_steps),
        mode=str(mode),
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
        batchify_steps=bool(batchify_steps),
        dmax=sampled_dmax,
        dlambda=float(dlambda),
    )


def random_degradation_configs_from_image(
    image: Tensor,
    interp_mode: str = "bicubic",
    gaussian_radius: int = 4,
    gaussian_sigma: float = 1.25,
    gaussian_extra_cells: int = 2,
    gaussian_enable_offset: bool = False,
    gaussian_offset_max: float = 0.5,
    gaussian_offset_lambda_first_order: float = 5e-2,
    gaussian_offset_lambda_second_order: float = 2e-1,
) -> HazeDegradationConfig:
    """Backward-compatible alias kept at old callsite name (haze-only now)."""
    return random_haze_degradation_config_from_image(
        image=image,
        interp_mode=interp_mode,
        gaussian_radius=gaussian_radius,
        gaussian_sigma=gaussian_sigma,
        gaussian_extra_cells=gaussian_extra_cells,
        gaussian_enable_offset=gaussian_enable_offset,
        gaussian_offset_max=gaussian_offset_max,
        gaussian_offset_lambda_first_order=gaussian_offset_lambda_first_order,
        gaussian_offset_lambda_second_order=gaussian_offset_lambda_second_order,
    )


class haze_degradation(nn.Module):
    """Spatially varying haze model using control map and global airlight."""

    def __init__(self, config: HazeDegradationConfig) -> None:
        super().__init__()
        if not isinstance(config, HazeDegradationConfig):
            raise ValueError(f"config must be HazeDegradationConfig, got {type(config)!r}")
        if not isinstance(config.map_config, ControlMapConfig):
            raise ValueError(f"map_config must be ControlMapConfig, got {type(config.map_config)!r}")

        airlight_init = config.airlight_init
        beta_mean = config.beta_mean
        beta_std = config.beta_std
        min_beta = config.min_beta
        if not isinstance(beta_mean, (int, float)): raise ValueError(f"beta_mean must be numeric, got {type(beta_mean)!r}")
        if beta_std is not None and not isinstance(beta_std, (int, float)):
            raise ValueError(f"beta_std must be numeric or None, got {type(beta_std)!r}")
        if not isinstance(min_beta, (int, float)): raise ValueError(f"min_beta must be numeric, got {type(min_beta)!r}")
        if float(min_beta) <= 0: raise ValueError(f"min_beta must be > 0, got {min_beta}")
        if beta_std is not None and float(beta_std) < 0:
            raise ValueError(f"beta_std must be >= 0 when provided, got {beta_std}")

        self.beta_mean = float(beta_mean)
        self.beta_std = None if beta_std is None else float(beta_std)
        self.min_beta = float(min_beta)

        self.beta_map_module = control_map(config.map_config)

        if isinstance(airlight_init, (int, float)):
            airlight_tensor = torch.full((1, 1, 1, 1), float(airlight_init), dtype=torch.float32)
        elif isinstance(airlight_init, tuple) and len(airlight_init) > 0:
            if not all(isinstance(v, (int, float)) for v in airlight_init):
                raise ValueError("airlight_init tuple must contain only numeric values")
            airlight_tensor = torch.tensor(list(airlight_init), dtype=torch.float32).view(1, len(airlight_init), 1, 1)
        else:
            raise ValueError("airlight_init must be float or non-empty tuple of floats")

        self.register_buffer("airlight", airlight_tensor.clone())

    def reset_parameters(self) -> None:
        """Reset beta map parameters and global airlight."""
        self.beta_map_module.reset_parameters()

    def project_trainable_parameters_(self) -> None:
        """Project trainable parameters into configured feasible set."""
        self.beta_map_module.project_trainable_parameters_()

    def get_airlight(self) -> Tensor:
        """Return fixed (non-trainable) airlight tensor used in forward."""
        return self.airlight

    def _check_image_and_distance(self, image: Tensor, distance_map: Tensor) -> tuple[int, int, int, int, Tensor]:
        if not isinstance(image, torch.Tensor): raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
        if not isinstance(distance_map, torch.Tensor): raise ValueError(f"distance_map must be torch.Tensor, got {type(distance_map)!r}")
        if image.ndim != 4: raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
        if distance_map.ndim != 4: raise ValueError(f"distance_map must have shape (B, 1, H, W) or (B, C, H, W), got {tuple(distance_map.shape)}")

        b, c, h, w = image.shape
        if h != self.beta_map_module.high_res_height or w != self.beta_map_module.high_res_width:
            raise ValueError(
                f"image spatial size must be ({self.beta_map_module.high_res_height}, {self.beta_map_module.high_res_width}), got ({h}, {w})"
            )

        db, dc, dh, dw = distance_map.shape
        if db != b: raise ValueError(f"distance_map batch mismatch: expected {b}, got {db}")
        if dh != h or dw != w: raise ValueError(f"distance_map spatial mismatch: expected ({h}, {w}), got ({dh}, {dw})")
        if dc not in {1, c}: raise ValueError(f"distance_map channel must be 1 or image channel {c}, got {dc}")
        distance = distance_map.expand(b, c, h, w) if dc == 1 else distance_map
        return b, c, h, w, distance

    def _compute_beta_map(self) -> Tensor:
        base_beta = self.beta_map_module()
        if self.beta_std is None:
            # No variance control branch: center by mean only, without std normalization.
            centered_beta = base_beta - base_beta.mean()
            return (centered_beta + self.beta_mean).clamp_min(self.min_beta)
        mean, std = base_beta.mean(), base_beta.std(unbiased=False)
        beta = (base_beta - mean) / (std + 1e-6)
        return (beta * self.beta_std + self.beta_mean).clamp_min(self.min_beta)

    def get_beta_map(self) -> Tensor:
        """Return current high-resolution beta map [1, 1, H, W]."""
        return self._compute_beta_map()

    def forward(self, image: Tensor, distance_map: Tensor) -> Tensor:
        """Apply haze model y = x * t + A * (1 - t), t = exp(-beta * d)."""
        _, c, _, _, distance = self._check_image_and_distance(image, distance_map)
        beta = self._compute_beta_map()
        transmission = torch.exp(-beta * distance)

        airlight_channels = self.airlight.shape[1]
        if airlight_channels not in {1, c}:
            raise ValueError(f"airlight channel must be 1 or image channel {c}, got {airlight_channels}")
        airlight = self.airlight.expand(1, c, 1, 1) if airlight_channels == 1 else self.airlight
        return image * transmission + airlight * (1.0 - transmission)

    def get_regularization_loss(self) -> Tensor:
        """Return total beta-map regularization loss."""
        return self.beta_map_module.regularization_loss()

    def get_regularization_items(self) -> dict[str, Tensor]:
        """Return detailed beta-map regularization losses."""
        return self.beta_map_module.regularization_items()

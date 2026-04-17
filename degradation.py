from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from control_map import ControlMapConfig, control_map


@dataclass
class NoiseDegradationConfig:
    image_height: int
    image_width: int
    num_channels: int
    noise_strength: float


@dataclass
class HazeDegradationConfig:
    map_config: ControlMapConfig
    airlight_init: float | tuple[float, float, float] = 1.0
    beta_mean: float = 1.0
    beta_std: float | None = None
    min_beta: float = 1e-6


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
) -> tuple[NoiseDegradationConfig, None, HazeDegradationConfig]:
    """Sample random Noise/Rain/Haze config triplet from one input image.

    Rules:
    - noise_strength in {15, 25, 50} / 255
    - rain degradation is removed; rain config slot returns None for compatibility
    - haze airlight base in [0.85, 1.0], per-channel jitter within +-0.02
    - haze beta_mean in [1.0, 1.3]
    - haze beta_std in [0.05, 0.2]
    """
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

    noise_choices = torch.tensor([15.0 / 255.0, 25.0 / 255.0, 50.0 / 255.0], dtype=torch.float32)
    noise_idx = int(torch.randint(0, len(noise_choices), (1,)).item())
    noise_strength = float(noise_choices[noise_idx].item())

    airlight_base = 0.85 + 0.15 * float(torch.rand(1).item())
    airlight = []
    for _ in range(3):
        jitter = (float(torch.rand(1).item()) * 2.0 - 1.0) * 0.02
        airlight.append(min(max(airlight_base + jitter, 0.85), 1.0))
    airlight_tuple = (float(airlight[0]), float(airlight[1]), float(airlight[2]))

    beta_mean = 0.1 + 0.4 * float(torch.rand(1).item())
    beta_std = beta_mean * (0.05 + 0.25 * float(torch.rand(1).item()))

    haze_lh, haze_lw = max(4, min(16, h)), max(4, min(16, w))

    noise_cfg = NoiseDegradationConfig(
        image_height=int(h),
        image_width=int(w),
        num_channels=int(c),
        noise_strength=float(noise_strength),
    )
    haze_cfg = HazeDegradationConfig(
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
    return noise_cfg, None, haze_cfg


class noise_degradation(nn.Module):
    """Fixed sampled additive noise module with resettable noise map."""

    def __init__(self, config: NoiseDegradationConfig) -> None:
        super().__init__()
        if not isinstance(config, NoiseDegradationConfig):
            raise ValueError(f"config must be NoiseDegradationConfig, got {type(config)!r}")

        image_height = config.image_height
        image_width = config.image_width
        num_channels = config.num_channels
        noise_strength = config.noise_strength
        if not isinstance(image_height, int) or image_height <= 0: raise ValueError(f"image_height must be positive int, got {image_height}")
        if not isinstance(image_width, int) or image_width <= 0: raise ValueError(f"image_width must be positive int, got {image_width}")
        if not isinstance(num_channels, int) or num_channels <= 0: raise ValueError(f"num_channels must be positive int, got {num_channels}")
        if not isinstance(noise_strength, (int, float)): raise ValueError(f"noise_strength must be numeric, got {type(noise_strength)!r}")

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.num_channels = int(num_channels)
        self.noise_strength = float(noise_strength)

        self.register_buffer("noise_map", torch.empty(1, self.num_channels, self.image_height, self.image_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-sample fixed Gaussian noise map."""
        with torch.no_grad(): self.noise_map.copy_(torch.randn_like(self.noise_map))

    def _check_image(self, image: Tensor) -> None:
        if not isinstance(image, torch.Tensor): raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
        if image.ndim != 4: raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
        _, c, h, w = image.shape
        if c != self.num_channels: raise ValueError(f"image channel mismatch: expected {self.num_channels}, got {c}")
        if h != self.image_height or w != self.image_width: raise ValueError(f"image spatial mismatch: expected ({self.image_height}, {self.image_width}), got ({h}, {w})")

    def forward(self, fig: Tensor) -> Tensor:
        """Apply additive fixed noise y = clamp(x + lambda * noise_map, 0, 1)."""
        self._check_image(fig)
        output = fig + self.noise_strength * self.noise_map
        return output.clamp(0.0, 1.0)

    def get_regularization_loss(self) -> Tensor:
        """Noise module has no regularization term."""
        return self.noise_map.new_tensor(0.0)

    def get_regularization_items(self) -> dict[str, Tensor]:
        """Return zero regularization items for interface consistency."""
        zero = self.get_regularization_loss()
        return {"loss_first_order": zero, "loss_second_order": zero, "loss_total": zero}


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


class noise_rain_haze_degradation(nn.Module):
    """Controller that composes noise/haze degradations (rain removed)."""

    def __init__(
        self,
        noise_config: NoiseDegradationConfig,
        rain_config: object | None,
        haze_config: HazeDegradationConfig,
        enable_noise: bool = True,
        enable_rain: bool = False,
        enable_haze: bool = True,
        rain_haze_order: str = "rain_haze",
    ) -> None:
        super().__init__()
        self._validate_enable_flag(enable_noise, "enable_noise")
        self._validate_enable_flag(enable_rain, "enable_rain")
        self._validate_enable_flag(enable_haze, "enable_haze")
        if enable_rain:
            raise ValueError("rain degradation has been removed from current source; see ~/adv_ir/backup")

        if not isinstance(noise_config, NoiseDegradationConfig):
            raise ValueError("noise_config must be NoiseDegradationConfig")
        if not isinstance(haze_config, HazeDegradationConfig):
            raise ValueError("haze_config must be HazeDegradationConfig")

        self.noise_config = noise_config
        self.rain_config = rain_config
        self.haze_config = haze_config
        self.noise_module = noise_degradation(self.noise_config)
        self.haze_module = haze_degradation(self.haze_config)

        self.enable_noise = bool(enable_noise)
        self.enable_rain = False
        self.enable_haze = bool(enable_haze)
        self.rain_haze_order = rain_haze_order
        self.current_enabled: list[str] = []
        self.current_order: list[str] = []
        self.reset_parameters()

    @staticmethod
    def _validate_enable_flag(value: bool, name: str) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be bool, got {type(value)!r}")

    def _get_available_degradations(self) -> list[str]:
        available: list[str] = []
        if self.enable_haze:
            available.append("haze")
        if self.enable_noise:
            available.append("noise")
        return available

    def reset_parameters(self) -> None:
        self.noise_module.reset_parameters()
        self.haze_module.reset_parameters()
        self.current_enabled = self._get_available_degradations()
        self.current_order = list(self.current_enabled)

    def project_trainable_parameters_(self) -> None:
        self.haze_module.project_trainable_parameters_()

    def forward(
        self,
        image: Tensor,
        distance_map: Tensor | None = None,
        rain_degraded_list: list[Tensor] | tuple[Tensor, ...] | Tensor | None = None,
        rain_topk: int | None = None,
    ) -> Tensor:
        del rain_degraded_list
        del rain_topk
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
        if image.ndim != 4:
            raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")

        current = image
        for name in self.current_order:
            if name == "haze":
                if distance_map is None:
                    raise ValueError("distance_map is required when haze is enabled")
                current = self.haze_module(current, distance_map)
            elif name == "noise":
                current = self.noise_module(current)
            else:
                raise RuntimeError(f"unsupported degradation name: {name}")
        return current

    def get_regularization_loss(self) -> Tensor:
        zero = self.noise_module.get_regularization_loss()
        loss_noise = zero
        loss_haze = self.haze_module.get_regularization_loss() if self.enable_haze else zero
        return loss_noise + loss_haze

    def get_regularization_items(self) -> dict[str, Tensor]:
        zero = self.noise_module.get_regularization_loss()
        loss_noise = zero
        loss_haze = self.haze_module.get_regularization_loss() if self.enable_haze else zero
        loss_total = loss_noise + loss_haze
        return {
            "loss_noise": loss_noise,
            "loss_haze": loss_haze,
            "loss_total": loss_total,
        }

    def get_current_state(self) -> dict:
        return {
            "enable_noise": self.enable_noise,
            "enable_rain": self.enable_rain,
            "enable_haze": self.enable_haze,
            "current_enabled": list(self.current_enabled),
            "current_order": list(self.current_order),
            "noise_config": asdict(self.noise_config),
            "rain_config": self.rain_config,
            "haze_config": asdict(self.haze_config),
        }

    def get_current_enabled(self) -> list[str]:
        return list(self.current_enabled)

    def get_current_order(self) -> list[str]:
        return list(self.current_order)

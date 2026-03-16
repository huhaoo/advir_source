from __future__ import annotations

import torch
from torch import Tensor, nn

from control_map import control_map, control_map_router


class noise_degradation(nn.Module):
    """Fixed sampled additive noise module with resettable noise map."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        num_channels: int,
        noise_strength: float,
        clamp_output: bool = False,
        clamp_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        if not isinstance(image_height, int) or image_height <= 0: raise ValueError(f"image_height must be positive int, got {image_height}")
        if not isinstance(image_width, int) or image_width <= 0: raise ValueError(f"image_width must be positive int, got {image_width}")
        if not isinstance(num_channels, int) or num_channels <= 0: raise ValueError(f"num_channels must be positive int, got {num_channels}")
        if not isinstance(noise_strength, (int, float)): raise ValueError(f"noise_strength must be numeric, got {type(noise_strength)!r}")
        if not isinstance(clamp_output, bool): raise ValueError(f"clamp_output must be bool, got {type(clamp_output)!r}")
        if not isinstance(clamp_range, tuple) or len(clamp_range) != 2: raise ValueError("clamp_range must be tuple(min, max)")
        clamp_min, clamp_max = float(clamp_range[0]), float(clamp_range[1])
        if clamp_min >= clamp_max: raise ValueError(f"clamp_range min must be < max, got {clamp_range}")

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.num_channels = int(num_channels)
        self.noise_strength = float(noise_strength)
        self.clamp_output = clamp_output
        self.clamp_range = (clamp_min, clamp_max)

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
        """Apply additive fixed noise y = x + lambda * noise_map."""
        self._check_image(fig)
        output = fig + self.noise_strength * self.noise_map
        if self.clamp_output: return output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def get_regularization_loss(self) -> Tensor:
        """Noise module has no regularization term."""
        return self.noise_map.new_tensor(0.0)

    def get_regularization_items(self) -> dict[str, Tensor]:
        """Return zero regularization items for interface consistency."""
        zero = self.get_regularization_loss()
        return {"loss_first_order": zero, "loss_second_order": zero, "loss_total": zero}


class rain_degradation(nn.Module):
    """Fuse original image and rain candidates via per-pixel routing maps."""

    def __init__(
        self,
        num_branches: int,
        low_res_height: int,
        low_res_width: int,
        high_res_height: int,
        high_res_width: int,
        interp_mode: str = "bilinear",
        align_corners: bool | None = False,
        lambda_first_order: float = 0.0,
        lambda_second_order: float = 0.0,
        init_mode: str = "zeros",
        init_value: float = 0.0,
        init_scale: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(num_branches, int) or num_branches < 2: raise ValueError(f"num_branches must be int >= 2, got {num_branches}")
        self.num_branches = int(num_branches)
        self.router = control_map_router(
            num_maps=self.num_branches,
            low_res_height=low_res_height,
            low_res_width=low_res_width,
            high_res_height=high_res_height,
            high_res_width=high_res_width,
            interp_mode=interp_mode,
            align_corners=align_corners,
            lambda_first_order=lambda_first_order,
            lambda_second_order=lambda_second_order,
            init_mode=init_mode,
            init_value=init_value,
            init_scale=init_scale,
            temperature=temperature,
        )

    def reset_parameters(self) -> None:
        """Reset all internal routing maps."""
        for module in self.router.maps: module.reset_parameters()

    def _build_candidate_stack(self, original: Tensor, degraded_list: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        if not isinstance(original, torch.Tensor): raise ValueError(f"original must be torch.Tensor, got {type(original)!r}")
        if original.ndim != 4: raise ValueError(f"original must have shape (B, C, H, W), got {tuple(original.shape)}")
        if not isinstance(degraded_list, (list, tuple)):
            raise ValueError(f"degraded_list must be list/tuple of tensors, got {type(degraded_list)!r}")
        expected = self.num_branches - 1
        if len(degraded_list) != expected: raise ValueError(f"degraded_list length must be {expected}, got {len(degraded_list)}")
        for idx, candidate in enumerate(degraded_list):
            if not isinstance(candidate, torch.Tensor): raise ValueError(f"degraded_list[{idx}] must be torch.Tensor, got {type(candidate)!r}")
            if candidate.shape != original.shape:
                raise ValueError(f"degraded_list[{idx}] shape mismatch: expected {tuple(original.shape)}, got {tuple(candidate.shape)}")
        return torch.stack([original, *degraded_list], dim=1)

    def forward(
        self,
        original: Tensor,
        degraded_list: list[Tensor] | tuple[Tensor, ...],
        topk: int | None = None,
    ) -> Tensor:
        """Return fused image from original + rain candidates."""
        stacked = self._build_candidate_stack(original, degraded_list)
        return self.router(stacked, top_k=topk)

    def get_regularization_loss(self) -> Tensor:
        """Return total routing regularization loss."""
        return self.router.regularization_loss()

    def get_regularization_items(self) -> dict[str, Tensor]:
        """Return detailed routing regularization losses."""
        return self.router.regularization_items()

    def get_weight_maps(self, topk: int | None = None) -> Tensor:
        """Return routing weights with shape [1, N, 1, H, W]."""
        maps = self.router.routing_maps(top_k=topk, return_logits=False)
        assert isinstance(maps, torch.Tensor)
        return maps

    def get_logits(self) -> Tensor:
        """Return raw routing logits with shape [1, N, 1, H, W]."""
        _, logits = self.router.routing_maps(top_k=None, return_logits=True)
        return logits


class haze_degradation(nn.Module):
    """Spatially varying haze model using control map and global airlight."""

    def __init__(
        self,
        low_res_height: int,
        low_res_width: int,
        high_res_height: int,
        high_res_width: int,
        interp_mode: str = "bilinear",
        align_corners: bool | None = False,
        lambda_first_order: float = 0.0,
        lambda_second_order: float = 0.0,
        init_mode: str = "zeros",
        init_value: float = 0.0,
        init_scale: float = 1.0,
        airlight_init: float | tuple[float, float, float] = 1.0,
        beta_mean: float = 1.0,
        beta_std: float = 0.2,
        min_beta: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(beta_mean, (int, float)): raise ValueError(f"beta_mean must be numeric, got {type(beta_mean)!r}")
        if not isinstance(beta_std, (int, float)): raise ValueError(f"beta_std must be numeric, got {type(beta_std)!r}")
        if not isinstance(min_beta, (int, float)): raise ValueError(f"min_beta must be numeric, got {type(min_beta)!r}")
        if float(min_beta) <= 0: raise ValueError(f"min_beta must be > 0, got {min_beta}")
        if float(beta_std) < 0: raise ValueError(f"beta_std must be >= 0, got {beta_std}")

        self.beta_mean = float(beta_mean)
        self.beta_std = float(beta_std)
        self.min_beta = float(min_beta)

        self.beta_map_module = control_map(
            low_res_height=low_res_height,
            low_res_width=low_res_width,
            high_res_height=high_res_height,
            high_res_width=high_res_width,
            interp_mode=interp_mode,
            align_corners=False if align_corners is None else align_corners,
            lambda_first_order=lambda_first_order,
            lambda_second_order=lambda_second_order,
            init_mode=init_mode,
            init_value=init_value,
            init_scale=init_scale,
        )

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
        mean, std = base_beta.mean(), base_beta.std(unbiased=False)
        beta = (base_beta - mean) / (std + 1e-6)
        return (beta * self.beta_std + self.beta_mean).clamp_min(self.min_beta)

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
    """Controller that composes noise/rain/haze degradations per reset state.

    Sampling rule for config specs:
    - non-list/tuple: fixed value
    - list/tuple of len==2: uniform random sample in range
    - list/tuple of len!=2: random pick from discrete candidates
    """

    noise_int_fields = {"image_height", "image_width", "num_channels"}
    rain_int_fields = {"num_branches", "low_res_height", "low_res_width", "high_res_height", "high_res_width"}
    haze_int_fields = {"low_res_height", "low_res_width", "high_res_height", "high_res_width"}

    def __init__(
        self,
        noise_config: dict,
        rain_config: dict,
        haze_config: dict,
        enable_noise_prob: float = 0.5,
        enable_rain_prob: float = 0.5,
        enable_haze_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self._validate_probability(enable_noise_prob, "enable_noise_prob")
        self._validate_probability(enable_rain_prob, "enable_rain_prob")
        self._validate_probability(enable_haze_prob, "enable_haze_prob")

        self.enable_noise_prob = float(enable_noise_prob)
        self.enable_rain_prob = float(enable_rain_prob)
        self.enable_haze_prob = float(enable_haze_prob)

        self.noise_config = dict(noise_config)
        self.rain_config = dict(rain_config)
        self.haze_config = dict(haze_config)

        noise_init = self._sample_config(self.noise_config, self.noise_int_fields)
        rain_init = self._sample_config(self.rain_config, self.rain_int_fields)
        haze_init = self._sample_config(self.haze_config, self.haze_int_fields)

        self.noise_module = noise_degradation(**noise_init)
        self.rain_module = rain_degradation(**rain_init)
        self.haze_module = haze_degradation(**haze_init)

        self.enable_noise = False
        self.enable_rain = False
        self.enable_haze = False
        self.current_order: list[str] = []
        self.current_noise_config: dict = {}
        self.current_rain_config: dict = {}
        self.current_haze_config: dict = {}

        self.reset_parameters()

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)!r}")
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    @staticmethod
    def _is_numeric(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _sample_from_spec(self, spec: object, cast_int: bool = False) -> object:
        if not isinstance(spec, (list, tuple)):
            return int(spec) if cast_int else spec
        if len(spec) == 0:
            raise ValueError("sample spec cannot be empty list/tuple")
        if len(spec) == 2 and self._is_numeric(spec[0]) and self._is_numeric(spec[1]):
            low, high = float(spec[0]), float(spec[1])
            if low > high:
                low, high = high, low
            sampled = low + (high - low) * float(torch.rand(1).item())
            # Int fields use round-to-nearest after uniform sampling.
            return int(round(sampled)) if cast_int else sampled
        idx = int(torch.randint(0, len(spec), (1,)).item())
        chosen = spec[idx]
        return int(chosen) if cast_int else chosen

    def _sample_config(self, config: dict, int_fields: set[str]) -> dict:
        sampled = {}
        for key, spec in config.items():
            sampled[key] = self._sample_from_spec(spec, cast_int=key in int_fields)
        return sampled

    @staticmethod
    def _apply_param_overrides(module: nn.Module, params: dict) -> None:
        for key, value in params.items():
            if hasattr(module, key): setattr(module, key, value)

    def _apply_noise_params(self, params: dict) -> None:
        self._apply_param_overrides(self.noise_module, params)
        self.noise_module.reset_parameters()

    def _apply_rain_params(self, params: dict) -> None:
        self._apply_param_overrides(self.rain_module, params)
        if "temperature" in params: self.rain_module.router.temperature = float(params["temperature"])
        if "lambda_first_order" in params or "lambda_second_order" in params:
            l1 = float(params.get("lambda_first_order", self.rain_module.router.maps[0].lambda_first_order))
            l2 = float(params.get("lambda_second_order", self.rain_module.router.maps[0].lambda_second_order))
            for m in self.rain_module.router.maps:
                m.lambda_first_order = l1
                m.lambda_second_order = l2
        self.rain_module.reset_parameters()

    def _apply_haze_params(self, params: dict) -> None:
        self._apply_param_overrides(self.haze_module, {k: v for k, v in params.items() if k in {"beta_mean", "beta_std", "min_beta"}})
        if "airlight_init" in params:
            airlight_value = params["airlight_init"]
            if isinstance(airlight_value, (int, float)):
                tensor = torch.full_like(self.haze_module.airlight, float(airlight_value))
            elif isinstance(airlight_value, tuple) and len(airlight_value) > 0:
                tensor = torch.tensor(list(airlight_value), dtype=self.haze_module.airlight.dtype, device=self.haze_module.airlight.device).view(1, len(airlight_value), 1, 1)
            else:
                raise ValueError(f"Unsupported sampled airlight_init: {airlight_value!r}")
            with torch.no_grad(): self.haze_module.airlight.copy_(tensor.to(self.haze_module.airlight.device, self.haze_module.airlight.dtype))
        self.haze_module.reset_parameters()

    def _sample_enable(self, explicit: bool | None, prob: float) -> bool:
        if explicit is None: return bool(torch.rand(1).item() < prob)
        if not isinstance(explicit, bool): raise ValueError(f"enable flag must be bool or None, got {type(explicit)!r}")
        return explicit

    def _build_execution_order(self) -> list[str]:
        core_order: list[str] = []
        if self.enable_rain: core_order.append("rain")
        if self.enable_haze: core_order.append("haze")
        if len(core_order) == 2 and bool(torch.randint(0, 2, (1,)).item()):
            core_order = [core_order[1], core_order[0]]
        if self.enable_noise: core_order.append("noise")
        return core_order

    def reset_parameters(
        self,
        enable_noise: bool | None = None,
        enable_rain: bool | None = None,
        enable_haze: bool | None = None,
    ) -> None:
        self.enable_noise = self._sample_enable(enable_noise, self.enable_noise_prob)
        self.enable_rain = self._sample_enable(enable_rain, self.enable_rain_prob)
        self.enable_haze = self._sample_enable(enable_haze, self.enable_haze_prob)

        self.current_noise_config = self._sample_config(self.noise_config, self.noise_int_fields)
        self.current_rain_config = self._sample_config(self.rain_config, self.rain_int_fields)
        self.current_haze_config = self._sample_config(self.haze_config, self.haze_int_fields)

        self._apply_noise_params(self.current_noise_config)
        self._apply_rain_params(self.current_rain_config)
        self._apply_haze_params(self.current_haze_config)

        self.current_order = self._build_execution_order()

    def forward(
        self,
        image: Tensor,
        distance_map: Tensor | None = None,
        rain_degraded_list: list[Tensor] | tuple[Tensor, ...] | None = None,
        rain_topk: int | None = None,
    ) -> Tensor:
        if not isinstance(image, torch.Tensor): raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
        if image.ndim != 4: raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")

        current = image
        for name in self.current_order:
            if name == "haze":
                if distance_map is None: raise ValueError("distance_map is required when haze is enabled")
                current = self.haze_module(current, distance_map)
            elif name == "rain":
                if rain_degraded_list is None: raise ValueError("rain_degraded_list is required when rain is enabled")
                current = self.rain_module(current, rain_degraded_list, topk=rain_topk)
            elif name == "noise":
                current = self.noise_module(current)
            else:
                raise RuntimeError(f"unsupported degradation name: {name}")
        return current

    def get_regularization_loss(self) -> Tensor:
        zero = self.noise_module.get_regularization_loss()
        loss_noise = zero
        loss_rain = self.rain_module.get_regularization_loss() if self.enable_rain else zero
        loss_haze = self.haze_module.get_regularization_loss() if self.enable_haze else zero
        return loss_noise + loss_rain + loss_haze

    def get_regularization_items(self) -> dict[str, Tensor]:
        zero = self.noise_module.get_regularization_loss()
        loss_noise = zero
        loss_rain = self.rain_module.get_regularization_loss() if self.enable_rain else zero
        loss_haze = self.haze_module.get_regularization_loss() if self.enable_haze else zero
        loss_total = loss_noise + loss_rain + loss_haze
        return {
            "loss_noise": loss_noise,
            "loss_rain": loss_rain,
            "loss_haze": loss_haze,
            "loss_total": loss_total,
        }

    def get_current_state(self) -> dict:
        return {
            "enable_noise": self.enable_noise,
            "enable_rain": self.enable_rain,
            "enable_haze": self.enable_haze,
            "current_order": list(self.current_order),
            "current_noise_config": dict(self.current_noise_config),
            "current_rain_config": dict(self.current_rain_config),
            "current_haze_config": dict(self.current_haze_config),
        }

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


allowed_interp_modes = {"nearest", "bilinear", "bicubic", "area"}
allowed_init_modes = {"zeros", "constant", "uniform", "normal"}


@dataclass
class ControlMapConfig:
    low_res_height: int
    low_res_width: int
    high_res_height: int
    high_res_width: int
    interp_mode: str = "bicubic"
    align_corners: bool | None = False
    lambda_first_order: float = 1e-2
    lambda_second_order: float = 1e-3
    init_mode: str = "zeros"
    init_value: float = 0.0
    init_scale: float = 0.1


@dataclass
class ControlMapRouterConfig:
    num_maps: int
    map_config: ControlMapConfig
    temperature: float = 1.0


class control_map(nn.Module):
    """Generic continuous control-map module for restoration degradation fields."""

    def __init__(self, config: ControlMapConfig) -> None:
        super().__init__()
        if not isinstance(config, ControlMapConfig):
            raise ValueError(f"config must be ControlMapConfig, got {type(config)!r}")

        low_res_height = config.low_res_height
        low_res_width = config.low_res_width
        high_res_height = config.high_res_height
        high_res_width = config.high_res_width
        interp_mode = config.interp_mode
        align_corners = config.align_corners
        lambda_first_order = config.lambda_first_order
        lambda_second_order = config.lambda_second_order
        init_mode = config.init_mode
        init_value = config.init_value
        init_scale = config.init_scale

        self._validate_spatial(low_res_height, low_res_width, high_res_height, high_res_width)
        if interp_mode not in allowed_interp_modes:
            raise ValueError(f"interp_mode must be one of {allowed_interp_modes}, got {interp_mode!r}")
        if init_mode not in allowed_init_modes:
            raise ValueError(f"init_mode must be one of {allowed_init_modes}, got {init_mode!r}")
        if not isinstance(lambda_first_order, (int, float)): raise ValueError(f"lambda_first_order must be numeric, got {type(lambda_first_order)!r}")
        if not isinstance(lambda_second_order, (int, float)): raise ValueError(f"lambda_second_order must be numeric, got {type(lambda_second_order)!r}")

        if init_scale < 0:
            raise ValueError(f"init_scale must be non-negative, got {init_scale}")

        self.low_res_height = int(low_res_height)
        self.low_res_width = int(low_res_width)
        self.high_res_height = int(high_res_height)
        self.high_res_width = int(high_res_width)

        self.interp_mode = interp_mode
        self.align_corners = align_corners

        self.lambda_first_order = float(lambda_first_order)
        self.lambda_second_order = float(lambda_second_order)

        self.init_mode = init_mode
        self.init_value = float(init_value)
        self.init_scale = float(init_scale)

        self.low_res_param = nn.Parameter(torch.empty(1, 1, low_res_height, low_res_width))
        self.reset_parameters()

    @staticmethod
    def _validate_spatial(lh: int, lw: int, hh: int, hw: int) -> None:
        for value, name in (
            (lh, "low_res_height"),
            (lw, "low_res_width"),
            (hh, "high_res_height"),
            (hw, "high_res_width"),
        ):
            if not isinstance(value, int) or value <= 0: raise ValueError(f"{name} must be a positive integer, got {value}")

    def reset_parameters(self) -> None:
        """Initialize low-resolution learnable map according to init_mode."""
        with torch.no_grad():
            if self.init_mode == "zeros": self.low_res_param.zero_()
            elif self.init_mode == "constant": self.low_res_param.fill_(self.init_value)
            elif self.init_mode == "uniform":
                self.low_res_param.uniform_(self.init_value - self.init_scale, self.init_value + self.init_scale)
            elif self.init_mode == "normal": self.low_res_param.normal_(mean=self.init_value, std=self.init_scale)
            else: raise RuntimeError(f"Unsupported init_mode: {self.init_mode}")

    def _interpolate(self, x: Tensor) -> Tensor:
        size = (self.high_res_height, self.high_res_width)
        if self.interp_mode in {"bilinear", "bicubic"}:
            return F.interpolate(x, size=size, mode=self.interp_mode, align_corners=self.align_corners)
        return F.interpolate(x, size=size, mode=self.interp_mode)

    def forward(self) -> Tensor:
        """Generate high-resolution continuous control map: [1, 1, H, W]."""
        return self._interpolate(self.low_res_param)

    def get_low_res_map(self) -> Tensor:
        """Return learnable low-resolution control map parameter tensor."""
        return self.low_res_param

    def first_order_loss(self) -> Tensor:
        """L1 regularization of first-order finite differences on low-res parameter map."""
        x = self.low_res_param
        diff_h, diff_v = x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]
        loss_h = diff_h.abs().mean() if diff_h.numel() > 0 else x.new_tensor(0.0)
        loss_v = diff_v.abs().mean() if diff_v.numel() > 0 else x.new_tensor(0.0)
        return loss_h + loss_v

    def second_order_loss(self) -> Tensor:
        """L1 regularization of second-order finite differences on low-res parameter map."""
        x = self.low_res_param
        diff2_h, diff2_v = x[:, :, :, 2:] - 2.0 * x[:, :, :, 1:-1] + x[:, :, :, :-2], x[:, :, 2:, :] - 2.0 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
        loss_h = diff2_h.abs().mean() if diff2_h.numel() > 0 else x.new_tensor(0.0)
        loss_v = diff2_v.abs().mean() if diff2_v.numel() > 0 else x.new_tensor(0.0)
        return loss_h + loss_v

    def regularization_loss(self) -> Tensor:
        """Weighted sum of first- and second-order regularization losses."""
        return self.lambda_first_order * self.first_order_loss() + self.lambda_second_order * self.second_order_loss()

    def regularization_items(self) -> dict[str, Tensor]:
        """Return individual and total regularization items as tensors."""
        loss_first, loss_second = self.first_order_loss(), self.second_order_loss()
        loss_total = self.lambda_first_order * loss_first + self.lambda_second_order * loss_second
        return {"loss_first_order": loss_first, "loss_second_order": loss_second, "loss_total": loss_total}


class control_map_router(nn.Module):
    """Route N input images with N learnable control logits maps.

    The module internally keeps ``num_maps`` instances of ``control_map``.
    Each map produces one logit field with shape [1, 1, H, W]. Logits are
    stacked to [1, N, 1, H, W], temperature-scaled, optionally top-k masked,
    then normalized with softmax on the N dimension.

    API:
    - ``forward`` only returns fused image tensor [B, C, H, W].
    - ``regularization_items`` / ``regularization_loss`` return router losses.
    - ``routing_maps`` returns weight maps (and optional logits) for analysis.
    """

    def __init__(self, config: ControlMapRouterConfig) -> None:
        super().__init__()
        if not isinstance(config, ControlMapRouterConfig):
            raise ValueError(f"config must be ControlMapRouterConfig, got {type(config)!r}")
        if not isinstance(config.map_config, ControlMapConfig):
            raise ValueError(f"config.map_config must be ControlMapConfig, got {type(config.map_config)!r}")

        num_maps = config.num_maps
        temperature = config.temperature
        map_config = config.map_config

        if not isinstance(num_maps, int) or num_maps <= 0:
            raise ValueError(f"num_maps must be a positive integer, got {num_maps}")
        if not isinstance(temperature, (int, float)): raise ValueError(f"temperature must be numeric, got {type(temperature)!r}")
        if temperature <= 0: raise ValueError(f"temperature must be > 0, got {temperature}")
        align_corners = map_config.align_corners
        if align_corners is not None and not isinstance(align_corners, bool): raise ValueError(f"align_corners must be bool or None, got {type(align_corners)!r}")

        self.num_maps = int(num_maps)
        self.low_res_height = int(map_config.low_res_height)
        self.low_res_width = int(map_config.low_res_width)
        self.high_res_height = int(map_config.high_res_height)
        self.high_res_width = int(map_config.high_res_width)
        self.temperature = float(temperature)

        map_align_corners = False if align_corners is None else align_corners
        self.maps = nn.ModuleList(
            [
                control_map(
                    ControlMapConfig(
                        low_res_height=self.low_res_height,
                        low_res_width=self.low_res_width,
                        high_res_height=self.high_res_height,
                        high_res_width=self.high_res_width,
                        interp_mode=map_config.interp_mode,
                        align_corners=map_align_corners,
                        lambda_first_order=map_config.lambda_first_order,
                        lambda_second_order=map_config.lambda_second_order,
                        init_mode=map_config.init_mode,
                        init_value=map_config.init_value,
                        init_scale=map_config.init_scale,
                    )
                )
                for _ in range(self.num_maps)
            ]
        )

    def _stack_logits(self) -> Tensor:
        """Stack per-map logits to shape [1, N, 1, H, W]."""
        return torch.stack([module() for module in self.maps], dim=1)

    def _normalize_top_k(self, top_k: int | None) -> int | None:
        if top_k is None: return None
        if not isinstance(top_k, int): raise ValueError(f"top_k must be None or int, got {type(top_k)!r}")
        if top_k <= 0: raise ValueError(f"top_k must be positive when provided, got {top_k}")
        return None if top_k >= self.num_maps else top_k

    @staticmethod
    def _apply_topk_mask(scaled_logits: Tensor, top_k: int | None) -> Tensor:
        """Mask non-topk logits on N dim with -inf before softmax."""
        if top_k is None: return scaled_logits

        _, n, _, _, _ = scaled_logits.shape
        if top_k >= n: return scaled_logits

        topk_indices = torch.topk(scaled_logits, k=top_k, dim=1).indices
        keep_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
        keep_mask.scatter_(1, topk_indices, True)
        return torch.where(keep_mask, scaled_logits, torch.full_like(scaled_logits, float("-inf")))

    def regularization_items(self) -> dict[str, Tensor]:
        """Return summed first-order/second-order/total regularization losses."""
        loss_first = torch.stack([module.first_order_loss() for module in self.maps], dim=0).sum()
        loss_second = torch.stack([module.second_order_loss() for module in self.maps], dim=0).sum()
        loss_total = torch.stack([module.regularization_loss() for module in self.maps], dim=0).sum()
        return {"loss_first_order": loss_first, "loss_second_order": loss_second, "loss_total": loss_total}

    def regularization_loss(self) -> Tensor:
        """Return total regularization loss summed over all internal maps."""
        return torch.stack([module.regularization_loss() for module in self.maps], dim=0).sum()

    def routing_maps(self, top_k: int | None = None, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Return weight maps [1, N, 1, H, W], optionally with raw logits."""
        top_k = self._normalize_top_k(top_k)
        logits = self._stack_logits()
        masked_logits = self._apply_topk_mask(logits / self.temperature, top_k)
        weight_maps = torch.softmax(masked_logits, dim=1)
        return (weight_maps, logits) if return_logits else weight_maps

    def forward(
        self,
        inputs: Tensor,
        top_k: int | None = None,
    ) -> Tensor:
        """Fuse inputs with per-pixel softmax routing and return output image.

        Args:
            inputs: Tensor with shape [B, N, C, H, W].
            top_k: If set and smaller than N, only top-k logits at each pixel
                (on N dim) participate in softmax.

        Returns:
            Output tensor with shape [B, C, H, W].
        """
        if not isinstance(inputs, torch.Tensor): raise ValueError(f"inputs must be a torch.Tensor, got {type(inputs)!r}")
        if inputs.ndim != 5: raise ValueError(f"inputs must have shape (B, N, C, H, W), got {tuple(inputs.shape)}")
        _, n, _, input_h, input_w = inputs.shape
        if n != self.num_maps: raise ValueError(f"inputs.shape[1] must equal num_maps={self.num_maps}, got {n}")
        if input_h != self.high_res_height or input_w != self.high_res_width:
            raise ValueError(
                "inputs spatial size must be "
                f"({self.high_res_height}, {self.high_res_width}), got ({input_h}, {input_w})"
            )

        top_k = self._normalize_top_k(top_k)

        logits = self._stack_logits()
        masked_logits = self._apply_topk_mask(logits / self.temperature, top_k)
        weight_maps = torch.softmax(masked_logits, dim=1)
        return (inputs * weight_maps).sum(dim=1)

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import torch.nn.functional as F


allowed_interp_modes = {"nearest", "bilinear", "bicubic", "area"}
allowed_value_bound_modes = {"none", "sigmoid", "tanh", "clamp"}
allowed_init_modes = {"zeros", "constant", "uniform", "normal"}


class control_map(nn.Module):
    """Generic continuous control-map module for restoration degradation fields."""

    def __init__(
        self,
        low_res_height: int,
        low_res_width: int,
        high_res_height: int,
        high_res_width: int,
        *,
        interp_mode: str = "bicubic",
        align_corners: bool = False,
        lambda_first_order: float = 1e-2,
        lambda_second_order: float = 1e-3,
        value_bound_mode: str = "none",
        value_min: float = 0.0,
        value_max: float = 1.0,
        mu: float | None = None,
        init_mode: str = "zeros",
        init_value: float = 0.0,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self._validate_spatial(low_res_height, low_res_width, high_res_height, high_res_width)
        if interp_mode not in allowed_interp_modes:
            raise ValueError(f"interp_mode must be one of {allowed_interp_modes}, got {interp_mode!r}")
        if value_bound_mode not in allowed_value_bound_modes:
            raise ValueError(
                f"value_bound_mode must be one of {allowed_value_bound_modes}, got {value_bound_mode!r}"
            )
        if init_mode not in allowed_init_modes:
            raise ValueError(f"init_mode must be one of {allowed_init_modes}, got {init_mode!r}")
        if not isinstance(lambda_first_order, (int, float)):
            raise ValueError(
                f"lambda_first_order must be numeric, got {type(lambda_first_order)!r}"
            )
        if not isinstance(lambda_second_order, (int, float)):
            raise ValueError(
                f"lambda_second_order must be numeric, got {type(lambda_second_order)!r}"
            )
        if value_min >= value_max:
            raise ValueError(f"value_min must be < value_max, got ({value_min}, {value_max})")
        if mu is not None and not isinstance(mu, (int, float)):
            raise ValueError(f"mu must be numeric, got {type(mu)!r}")

        if init_scale < 0:
            raise ValueError(f"init_scale must be non-negative, got {init_scale}")

        self.low_res_height = low_res_height
        self.low_res_width = low_res_width
        self.high_res_height = high_res_height
        self.high_res_width = high_res_width

        self.interp_mode = interp_mode
        self.align_corners = align_corners

        self.lambda_first_order = float(lambda_first_order)
        self.lambda_second_order = float(lambda_second_order)

        self.value_bound_mode = value_bound_mode
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.mu = float(mu) if mu is not None else None

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
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")

    def reset_parameters(self) -> None:
        """Initialize low-resolution learnable map according to init_mode."""
        with torch.no_grad():
            if self.init_mode == "zeros":
                self.low_res_param.zero_()
            elif self.init_mode == "constant":
                self.low_res_param.fill_(self.init_value)
            elif self.init_mode == "uniform":
                left = self.init_value - self.init_scale
                right = self.init_value + self.init_scale
                self.low_res_param.uniform_(left, right)
            elif self.init_mode == "normal":
                self.low_res_param.normal_(mean=self.init_value, std=self.init_scale)
            else:
                raise RuntimeError(f"Unsupported init_mode: {self.init_mode}")

    def _apply_value_bound(self, x: Tensor) -> Tensor:
        if self.value_bound_mode == "none":
            return x
        if self.value_bound_mode == "sigmoid":
            bounded = torch.sigmoid(x)
            return self.value_min + (self.value_max - self.value_min) * bounded
        if self.value_bound_mode == "tanh":
            bounded = 0.5 * (torch.tanh(x) + 1.0)
            return self.value_min + (self.value_max - self.value_min) * bounded
        if self.value_bound_mode == "clamp":
            return torch.clamp(x, min=self.value_min, max=self.value_max)
        raise RuntimeError(f"Unsupported value_bound_mode: {self.value_bound_mode}")

    def _interpolate(self, x: Tensor) -> Tensor:
        size = (self.high_res_height, self.high_res_width)
        if self.interp_mode in {"bilinear", "bicubic"}:
            return F.interpolate(x, size=size, mode=self.interp_mode, align_corners=self.align_corners)
        return F.interpolate(x, size=size, mode=self.interp_mode)

    def forward(self) -> Tensor:
        """Generate high-resolution continuous control map: [1, 1, H, W].

        If self.mu is not None, shift the whole map so its mean equals self.mu.
        """
        high_res = self._interpolate(self.low_res_param)
        high_res = self._apply_value_bound(high_res)
        if self.mu is not None:
            high_res = high_res - high_res.mean() + self.mu
        return high_res

    def get_low_res_map(self) -> Tensor:
        """Return learnable low-resolution control map parameter tensor."""
        return self.low_res_param

    def first_order_loss(self) -> Tensor:
        """L1 regularization of first-order finite differences on low-res parameter map."""
        x = self.low_res_param
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]

        loss_h = diff_h.abs().mean() if diff_h.numel() > 0 else x.new_tensor(0.0)
        loss_v = diff_v.abs().mean() if diff_v.numel() > 0 else x.new_tensor(0.0)
        return loss_h + loss_v

    def second_order_loss(self) -> Tensor:
        """L1 regularization of second-order finite differences on low-res parameter map."""
        x = self.low_res_param
        diff2_h = x[:, :, :, 2:] - 2.0 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
        diff2_v = x[:, :, 2:, :] - 2.0 * x[:, :, 1:-1, :] + x[:, :, :-2, :]

        loss_h = diff2_h.abs().mean() if diff2_h.numel() > 0 else x.new_tensor(0.0)
        loss_v = diff2_v.abs().mean() if diff2_v.numel() > 0 else x.new_tensor(0.0)
        return loss_h + loss_v

    def regularization_loss(self) -> Tensor:
        """Weighted sum of first- and second-order regularization losses."""
        loss_first = self.first_order_loss()
        loss_second = self.second_order_loss()
        return self.lambda_first_order * loss_first + self.lambda_second_order * loss_second

    def regularization_items(self) -> dict[str, Tensor]:
        """Return individual and total regularization items as tensors."""
        loss_first = self.first_order_loss()
        loss_second = self.second_order_loss()
        loss_total = self.lambda_first_order * loss_first + self.lambda_second_order * loss_second
        return {
            "loss_first_order": loss_first,
            "loss_second_order": loss_second,
            "loss_total": loss_total,
        }


class control_map_router(nn.Module):
    """Route N input images with N learnable control logits maps.

    The module internally keeps ``num_maps`` instances of ``control_map``.
    Each map produces one logit field with shape [1, 1, H, W]. Logits are
    stacked to [1, N, 1, H, W], temperature-scaled, optionally top-k masked,
    then normalized with softmax on the N dimension.

    Notes:
    - For logits usage, ``value_bound_mode='none'`` is usually preferred.
    - Strong value bounds (for example sigmoid/tanh) may shrink logit range and
      weaken routing sharpness.
    - Regularization items are aggregated by SUM across all internal maps.
    """

    def __init__(
        self,
        num_maps: int,
        low_res_height: int,
        low_res_width: int,
        high_res_height: int,
        high_res_width: int,
        interp_mode: str = "bilinear",
        align_corners: bool | None = False,
        lambda_first_order: float = 0.0,
        lambda_second_order: float = 0.0,
        value_bound_mode: str = "none",
        value_min: float = 0.0,
        value_max: float = 1.0,
        mu: float | None = None,
        init_mode: str = "zeros",
        init_value: float = 0.0,
        init_scale: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(num_maps, int) or num_maps <= 0:
            raise ValueError(f"num_maps must be a positive integer, got {num_maps}")
        if not isinstance(temperature, (int, float)):
            raise ValueError(f"temperature must be numeric, got {type(temperature)!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if align_corners is not None and not isinstance(align_corners, bool):
            raise ValueError(f"align_corners must be bool or None, got {type(align_corners)!r}")

        self.num_maps = int(num_maps)
        self.low_res_height = int(low_res_height)
        self.low_res_width = int(low_res_width)
        self.high_res_height = int(high_res_height)
        self.high_res_width = int(high_res_width)
        self.temperature = float(temperature)

        map_align_corners = False if align_corners is None else align_corners
        self.maps = nn.ModuleList(
            [
                control_map(
                    low_res_height=self.low_res_height,
                    low_res_width=self.low_res_width,
                    high_res_height=self.high_res_height,
                    high_res_width=self.high_res_width,
                    interp_mode=interp_mode,
                    align_corners=map_align_corners,
                    lambda_first_order=lambda_first_order,
                    lambda_second_order=lambda_second_order,
                    value_bound_mode=value_bound_mode,
                    value_min=value_min,
                    value_max=value_max,
                    mu=mu,
                    init_mode=init_mode,
                    init_value=init_value,
                    init_scale=init_scale,
                )
                for _ in range(self.num_maps)
            ]
        )

    def _stack_logits(self) -> Tensor:
        """Stack per-map logits to shape [1, N, 1, H, W]."""
        logits = [module() for module in self.maps]
        return torch.stack(logits, dim=1)

    @staticmethod
    def _apply_topk_mask(scaled_logits: Tensor, top_k: int | None) -> Tensor:
        """Mask non-topk logits on N dim with -inf before softmax."""
        if top_k is None:
            return scaled_logits

        _, n, _, _, _ = scaled_logits.shape
        if top_k >= n:
            return scaled_logits

        topk_indices = torch.topk(scaled_logits, k=top_k, dim=1).indices
        keep_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
        keep_mask.scatter_(1, topk_indices, True)
        neg_inf = torch.full_like(scaled_logits, float("-inf"))
        return torch.where(keep_mask, scaled_logits, neg_inf)

    def _aggregate_regularization(self) -> dict[str, Tensor]:
        """Aggregate regularization tensors by summing over internal maps."""
        first_items = [module.first_order_loss() for module in self.maps]
        second_items = [module.second_order_loss() for module in self.maps]
        total_items = [module.regularization_loss() for module in self.maps]

        loss_first = torch.stack(first_items, dim=0).sum()
        loss_second = torch.stack(second_items, dim=0).sum()
        loss_total = torch.stack(total_items, dim=0).sum()
        return {
            "loss_first_order": loss_first,
            "loss_second_order": loss_second,
            "loss_total": loss_total,
        }

    def forward(
        self,
        inputs: Tensor,
        top_k: int | None = None,
        return_weight_maps: bool = True,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Fuse inputs with per-pixel softmax routing.

        Args:
            inputs: Tensor with shape [B, N, C, H, W].
            top_k: If set and smaller than N, only top-k logits at each pixel
                (on N dim) participate in softmax.
            return_weight_maps: Whether to include "weight_maps" in output.
            return_logits: Whether to include raw "logits" in output.

        Returns:
            A dict containing at least:
            - "output": [B, C, H, W]
            - "reg_items": dict of regularization losses
            Optional:
            - "weight_maps": [1, N, 1, H, W]
            - "logits": [1, N, 1, H, W]
        """
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"inputs must be a torch.Tensor, got {type(inputs)!r}")
        if inputs.ndim != 5:
            raise ValueError(f"inputs must have shape (B, N, C, H, W), got {tuple(inputs.shape)}")
        _, n, _, input_h, input_w = inputs.shape
        if n != self.num_maps:
            raise ValueError(f"inputs.shape[1] must equal num_maps={self.num_maps}, got {n}")
        if input_h != self.high_res_height or input_w != self.high_res_width:
            raise ValueError(
                "inputs spatial size must be "
                f"({self.high_res_height}, {self.high_res_width}), got ({input_h}, {input_w})"
            )

        if top_k is not None:
            if not isinstance(top_k, int):
                raise ValueError(f"top_k must be None or int, got {type(top_k)!r}")
            if top_k <= 0:
                raise ValueError(f"top_k must be positive when provided, got {top_k}")
            if top_k >= self.num_maps:
                top_k = None

        # logits: [1, N, 1, H, W]
        logits = self._stack_logits()
        scaled_logits = logits / self.temperature
        masked_logits = self._apply_topk_mask(scaled_logits, top_k)

        # softmax on candidate map dimension N
        weight_maps = torch.softmax(masked_logits, dim=1)

        # inputs: [B, N, C, H, W], weight_maps broadcasts on B and C
        weighted = inputs * weight_maps
        output = weighted.sum(dim=1)

        result: dict[str, Any] = {
            "output": output,
            "reg_items": self._aggregate_regularization(),
        }
        if return_weight_maps:
            result["weight_maps"] = weight_maps
        if return_logits:
            result["logits"] = logits
        return result


def _save_map_png(array: Tensor, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(array.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def _save_map_with_point_png(array: Tensor, y: int, x: int, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np_array = array.detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(np_array, cmap="magma")
    plt.scatter([x], [y], c="cyan", s=36)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def _prepare_demo_output_dir() -> Path:
    """Reset /tmp_demo before each demo run as required by project rules."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "tmp_demo"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_control_map_demo() -> None:
    """Run a single-file demo with random high-res loss map and gradient descent."""
    module = control_map(
        low_res_height=16,
        low_res_width=16,
        high_res_height=256,
        high_res_width=256,
        interp_mode="bicubic",
        align_corners=False,
        lambda_first_order=1e-2,
        lambda_second_order=2e-2,
        value_bound_mode="sigmoid",
        value_min=0.0,
        value_max=1.0,
        init_mode="normal",
        init_value=0.0,
        init_scale=0.1,
    )

    output_dir = _prepare_demo_output_dir()

    random_loss_map = torch.rand((1, 1, module.high_res_height, module.high_res_width))
    optimizer = torch.optim.Adam(module.parameters(), lr=0.05)

    best_total_loss = None
    best_high_res_map = None
    best_per_pixel_loss = None
    best_step = -1

    for step_idx in range(200):
        optimizer.zero_grad()
        high_res_map = module()
        per_pixel_loss = high_res_map * random_loss_map
        task_loss = per_pixel_loss.mean()
        reg_loss = module.regularization_loss()
        total_loss = task_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        if best_total_loss is None or total_loss.item() < best_total_loss:
            best_total_loss = float(total_loss.item())
            best_high_res_map = high_res_map.detach().clone()
            best_per_pixel_loss = per_pixel_loss.detach().clone()
            best_step = step_idx

    assert best_high_res_map is not None
    assert best_per_pixel_loss is not None

    pixel_loss_2d = best_per_pixel_loss.squeeze(0).squeeze(0)
    min_flat_idx = int(torch.argmin(pixel_loss_2d).item())
    min_y = min_flat_idx // module.high_res_width
    min_x = min_flat_idx % module.high_res_width

    low_res_map = module.get_low_res_map().detach().squeeze(0).squeeze(0)
    high_res_map = best_high_res_map.squeeze(0).squeeze(0)
    loss_map_2d = random_loss_map.squeeze(0).squeeze(0)

    low_png = output_dir / "control_map_low_res.png"
    high_png = output_dir / "control_map_high_res_optimized.png"
    random_loss_png = output_dir / "random_high_res_loss_map.png"
    pixel_loss_png = output_dir / "pixel_loss_map_with_min_point.png"
    stats_txt = output_dir / "control_map_demo_stats.txt"

    _save_map_png(low_res_map, low_png, "low_res_parameter_map")
    _save_map_png(high_res_map, high_png, "optimized_high_res_control_map")
    _save_map_png(loss_map_2d, random_loss_png, "random_high_res_loss_map")
    _save_map_with_point_png(pixel_loss_2d, min_y, min_x, pixel_loss_png, "pixel_loss_map_min_point")

    items = module.regularization_items()
    stats_lines = [
        "control_map_demo_stats",
        f"low_res_shape={tuple(module.get_low_res_map().shape)}",
        f"high_res_shape={tuple(module().shape)}",
        f"best_step={best_step}",
        f"best_total_loss={best_total_loss:.10f}",
        f"first_order_loss={items['loss_first_order'].item():.10f}",
        f"second_order_loss={items['loss_second_order'].item():.10f}",
        f"regularization_loss={items['loss_total'].item():.10f}",
        f"high_res_min={high_res_map.min().item():.10f}",
        f"high_res_max={high_res_map.max().item():.10f}",
        f"high_res_mean={high_res_map.mean().item():.10f}",
        f"min_loss_point_y={min_y}",
        f"min_loss_point_x={min_x}",
        f"low_res_png={low_png}",
        f"high_res_png={high_png}",
        f"random_loss_png={random_loss_png}",
        f"pixel_loss_png={pixel_loss_png}",
    ]
    stats_txt.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    print(f"low_res_shape={tuple(module.get_low_res_map().shape)}")
    print(f"high_res_shape={tuple(module().shape)}")
    print(f"best_step={best_step}")
    print(f"best_total_loss={best_total_loss:.10f}")
    print(f"min_loss_point=(y={min_y}, x={min_x})")
    print(f"output_dir={output_dir}")
    print(f"generated_files={[str(stats_txt), str(low_png), str(high_png), str(random_loss_png), str(pixel_loss_png)]}")


def _build_router_demo_inputs(height: int, width: int) -> Tensor:
    """Create demo inputs with clear spatial semantics: zero, x-gradient, center blob."""
    zeros = torch.zeros((1, 1, height, width), dtype=torch.float32)

    x_grad = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
    x_grad = x_grad.expand(1, 1, height, width)

    ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius2 = xx.pow(2) + yy.pow(2)
    center_blob = torch.exp(-radius2 / 0.08).unsqueeze(0).unsqueeze(0)

    # [B=1, N=3, C=1, H, W]
    return torch.stack([zeros, x_grad, center_blob], dim=1)


def _init_router_logits_for_demo(router: control_map_router) -> None:
    """Write low-res logits patterns so left/right/center prefer different branches."""
    lh, lw = router.low_res_height, router.low_res_width
    ys = torch.linspace(-1.0, 1.0, lh, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, lw, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    left_favor = -3.0 * xx
    right_favor = 3.0 * xx
    center_favor = 3.5 * torch.exp(-(xx.pow(2) + yy.pow(2)) / 0.2) - 0.6

    patterns = [left_favor, right_favor, center_favor]
    with torch.no_grad():
        for idx, pattern in enumerate(patterns):
            router.maps[idx].get_low_res_map().copy_(pattern.unsqueeze(0).unsqueeze(0))


def run_control_map_router_demo() -> None:
    """Run a router demo to validate softmax routing, top-k behavior, and regularization."""
    output_dir = _prepare_demo_output_dir()

    h, w = 128, 128
    inputs = _build_router_demo_inputs(height=h, width=w)
    router = control_map_router(
        num_maps=3,
        low_res_height=16,
        low_res_width=16,
        high_res_height=h,
        high_res_width=w,
        interp_mode="bicubic",
        align_corners=False,
        lambda_first_order=1e-2,
        lambda_second_order=2e-2,
        value_bound_mode="none",
        init_mode="zeros",
        init_value=0.0,
        init_scale=1.0,
        temperature=0.5,
    )

    _init_router_logits_for_demo(router)

    result_all = router(inputs, top_k=None, return_weight_maps=True, return_logits=True)
    result_top2 = router(inputs, top_k=2, return_weight_maps=True, return_logits=False)
    result_top1 = router(inputs, top_k=1, return_weight_maps=True, return_logits=False)

    weight_all = result_all["weight_maps"]
    assert isinstance(weight_all, torch.Tensor)

    output_all = result_all["output"]
    output_top2 = result_top2["output"]
    output_top1 = result_top1["output"]
    logits = result_all["logits"]
    reg_items = result_all["reg_items"]

    output_all_png = output_dir / "router_output_topk_none.png"
    output_top2_png = output_dir / "router_output_topk_2.png"
    output_top1_png = output_dir / "router_output_topk_1.png"
    weight0_png = output_dir / "router_weight_map_branch0.png"
    weight1_png = output_dir / "router_weight_map_branch1.png"
    weight2_png = output_dir / "router_weight_map_branch2.png"
    stats_txt = output_dir / "control_map_router_demo_stats.txt"

    _save_map_png(output_all.squeeze(0).squeeze(0), output_all_png, "router_output_topk_none")
    _save_map_png(output_top2.squeeze(0).squeeze(0), output_top2_png, "router_output_topk_2")
    _save_map_png(output_top1.squeeze(0).squeeze(0), output_top1_png, "router_output_topk_1")
    _save_map_png(weight_all[0, 0, 0], weight0_png, "router_weight_branch0")
    _save_map_png(weight_all[0, 1, 0], weight1_png, "router_weight_branch1")
    _save_map_png(weight_all[0, 2, 0], weight2_png, "router_weight_branch2")

    branch_stats = []
    for idx in range(weight_all.shape[1]):
        w_map = weight_all[0, idx, 0]
        branch_stats.append(
            f"branch_{idx}_weight_min={w_map.min().item():.10f},"
            f"max={w_map.max().item():.10f},mean={w_map.mean().item():.10f}"
        )

    stats_lines = [
        "control_map_router_demo_stats",
        f"inputs_shape={tuple(inputs.shape)}",
        f"output_shape_topk_none={tuple(output_all.shape)}",
        f"output_shape_topk_2={tuple(output_top2.shape)}",
        f"output_shape_topk_1={tuple(output_top1.shape)}",
        f"weight_maps_shape={tuple(weight_all.shape)}",
        f"logits_shape={tuple(logits.shape)}",
        f"reg_loss_first_order={reg_items['loss_first_order'].item():.10f}",
        f"reg_loss_second_order={reg_items['loss_second_order'].item():.10f}",
        f"reg_loss_total={reg_items['loss_total'].item():.10f}",
        *branch_stats,
        f"output_all_png={output_all_png}",
        f"output_top2_png={output_top2_png}",
        f"output_top1_png={output_top1_png}",
        f"weight0_png={weight0_png}",
        f"weight1_png={weight1_png}",
        f"weight2_png={weight2_png}",
    ]
    stats_txt.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    print(f"inputs_shape={tuple(inputs.shape)}")
    print(f"output_shape_topk_none={tuple(output_all.shape)}")
    print(f"output_shape_topk_2={tuple(output_top2.shape)}")
    print(f"output_shape_topk_1={tuple(output_top1.shape)}")
    print(f"weight_maps_shape={tuple(weight_all.shape)}")
    for line in branch_stats:
        print(line)
    print(
        "reg_items="
        f"{{'loss_first_order': {reg_items['loss_first_order'].item():.10f}, "
        f"'loss_second_order': {reg_items['loss_second_order'].item():.10f}, "
        f"'loss_total': {reg_items['loss_total'].item():.10f}}}"
    )
    print(f"output_dir={output_dir}")
    print(
        "generated_files="
        f"{[str(stats_txt), str(output_all_png), str(output_top2_png), str(output_top1_png), str(weight0_png), str(weight1_png), str(weight2_png)]}"
    )


if __name__ == "__main__":
    run_control_map_router_demo()

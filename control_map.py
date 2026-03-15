from __future__ import annotations

from pathlib import Path

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
        self._validate_interp_mode(interp_mode)
        self._validate_value_bound_mode(value_bound_mode)
        self._validate_init_mode(init_mode)
        self._validate_numeric(lambda_first_order, "lambda_first_order")
        self._validate_numeric(lambda_second_order, "lambda_second_order")
        self._validate_range(value_min, value_max)
        if mu is not None:
            self._validate_numeric(mu, "mu")

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

    @staticmethod
    def _validate_interp_mode(mode: str) -> None:
        if mode not in allowed_interp_modes:
            raise ValueError(f"interp_mode must be one of {allowed_interp_modes}, got {mode!r}")

    @staticmethod
    def _validate_value_bound_mode(mode: str) -> None:
        if mode not in allowed_value_bound_modes:
            raise ValueError(f"value_bound_mode must be one of {allowed_value_bound_modes}, got {mode!r}")

    @staticmethod
    def _validate_init_mode(mode: str) -> None:
        if mode not in allowed_init_modes:
            raise ValueError(f"init_mode must be one of {allowed_init_modes}, got {mode!r}")

    @staticmethod
    def _validate_numeric(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)!r}")

    @staticmethod
    def _validate_range(value_min: float, value_max: float) -> None:
        if value_min >= value_max:
            raise ValueError(f"value_min must be < value_max, got ({value_min}, {value_max})")

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

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "tmp_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    run_control_map_demo()

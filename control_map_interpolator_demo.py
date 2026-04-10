from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from control_map_interpolator import InterpolationFieldConfig, control_map_interpolator
except ModuleNotFoundError:
    from .control_map_interpolator import InterpolationFieldConfig, control_map_interpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp_demo" / "control_map_interpolator_demo"


def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape={tuple(x.shape)}")
    arr = x.detach().cpu().float().numpy()
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_map_png(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_to_uint8_image(x), mode="L")
    img.save(path)


def _build_demo_input(low_h: int, low_w: int, device: torch.device) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=low_h, device=device).view(low_h, 1)
    xx = torch.linspace(-1.0, 1.0, steps=low_w, device=device).view(1, low_w)
    base = torch.exp(-((xx * xx + yy * yy) / 0.6))
    ramp = 0.35 * xx + 0.20 * yy
    signal = base + ramp
    return signal.unsqueeze(0).unsqueeze(0).contiguous()


def run_demo(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    low_h, low_w = 9, 13
    high_h, high_w = 96, 144
    lowres = _build_demo_input(low_h=low_h, low_w=low_w, device=device)

    bilinear_cfg = InterpolationFieldConfig(
        low_res_height=low_h,
        low_res_width=low_w,
        high_res_height=high_h,
        high_res_width=high_w,
        mode="bilinear",
        align_corners=False,
    )
    bicubic_cfg = InterpolationFieldConfig(
        low_res_height=low_h,
        low_res_width=low_w,
        high_res_height=high_h,
        high_res_width=high_w,
        mode="bicubic",
        align_corners=False,
    )
    gaussian_default_cfg = InterpolationFieldConfig(
        low_res_height=low_h,
        low_res_width=low_w,
        high_res_height=high_h,
        high_res_width=high_w,
        mode="gaussian",
        gaussian_radius=3,
        gaussian_sigma=1.25,
        gaussian_extra_cells=1,
    )
    gaussian_no_extra_cfg = InterpolationFieldConfig(
        low_res_height=low_h,
        low_res_width=low_w,
        high_res_height=high_h,
        high_res_width=high_w,
        mode="gaussian",
        gaussian_radius=3,
        gaussian_sigma=1.25,
        gaussian_extra_cells=0,
    )

    bilinear_out = control_map_interpolator(bilinear_cfg)(lowres)
    bicubic_out = control_map_interpolator(bicubic_cfg)(lowres)
    gaussian_default_out = control_map_interpolator(gaussian_default_cfg)(lowres)
    gaussian_no_extra_out = control_map_interpolator(gaussian_no_extra_cfg)(lowres)

    bilinear_ref = F.interpolate(lowres, size=(high_h, high_w), mode="bilinear", align_corners=False)
    bicubic_ref = F.interpolate(lowres, size=(high_h, high_w), mode="bicubic", align_corners=False)

    bilinear_max_abs_diff = float((bilinear_out - bilinear_ref).abs().max().item())
    bicubic_max_abs_diff = float((bicubic_out - bicubic_ref).abs().max().item())

    edge_default = gaussian_default_out[:, :, 0, :]
    edge_no_extra = gaussian_no_extra_out[:, :, 0, :]
    gaussian_edge_mean_abs_diff = float((edge_default - edge_no_extra).abs().mean().item())

    _save_map_png(lowres[0, 0], output_root / "lowres_input.png")
    _save_map_png(bilinear_out[0, 0], output_root / "highres_bilinear.png")
    _save_map_png(bicubic_out[0, 0], output_root / "highres_bicubic.png")
    _save_map_png(gaussian_default_out[0, 0], output_root / "highres_gaussian_default.png")
    _save_map_png(gaussian_no_extra_out[0, 0], output_root / "highres_gaussian_no_extra.png")

    summary = {
        "demo_name": "control_map_interpolator_demo",
        "low_res_shape": [1, 1, low_h, low_w],
        "high_res_shape": [1, 1, high_h, high_w],
        "default_behavior_check": {
            "bilinear_align_corners": False,
            "bicubic_align_corners": False,
            "bilinear_vs_torch_max_abs_diff": bilinear_max_abs_diff,
            "bicubic_vs_torch_max_abs_diff": bicubic_max_abs_diff,
        },
        "gaussian_config": {
            "default": {
                "gaussian_radius": gaussian_default_cfg.gaussian_radius,
                "gaussian_sigma": gaussian_default_cfg.gaussian_sigma,
                "gaussian_extra_cells": gaussian_default_cfg.gaussian_extra_cells,
            },
            "ablation_no_extra": {
                "gaussian_radius": gaussian_no_extra_cfg.gaussian_radius,
                "gaussian_sigma": gaussian_no_extra_cfg.gaussian_sigma,
                "gaussian_extra_cells": gaussian_no_extra_cfg.gaussian_extra_cells,
            },
            "edge_mean_abs_diff_default_vs_no_extra": gaussian_edge_mean_abs_diff,
        },
        "outputs": {
            "lowres_input": str(output_root / "lowres_input.png"),
            "highres_bilinear": str(output_root / "highres_bilinear.png"),
            "highres_bicubic": str(output_root / "highres_bicubic.png"),
            "highres_gaussian_default": str(output_root / "highres_gaussian_default.png"),
            "highres_gaussian_no_extra": str(output_root / "highres_gaussian_no_extra.png"),
        },
    }

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo for unified control-map interpolation module")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_demo(output_root=args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

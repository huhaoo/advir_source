from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from control_map_interpolator import InterpolationFieldConfig, control_map_interpolator
    from haze_dataset_builder import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_TRAIN_MANIFEST,
        FIXED_GRAD_TARGET_STD,
        PROJECT_ROOT,
        _collect_train_clear_images,
        _ensure_bchw,
        _load_rgb_tensor,
        _save_density_tensor,
        _save_rgb_tensor,
        deg,
    )
    from util.runtime import seed_everything as _seed_everything
except ModuleNotFoundError:
    from .control_map_interpolator import InterpolationFieldConfig, control_map_interpolator
    from .haze_dataset_builder import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_TRAIN_MANIFEST,
        FIXED_GRAD_TARGET_STD,
        PROJECT_ROOT,
        _collect_train_clear_images,
        _ensure_bchw,
        _load_rgb_tensor,
        _save_density_tensor,
        _save_rgb_tensor,
        deg,
    )
    from .util.runtime import seed_everything as _seed_everything


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp_demo" / "random_adv_ots_s1_64_1024"


def _first_order_regularization(low_res_map: Tensor) -> Tensor:
    diff_h = low_res_map[:, :, :, 1:] - low_res_map[:, :, :, :-1]
    diff_v = low_res_map[:, :, 1:, :] - low_res_map[:, :, :-1, :]
    loss_h = diff_h.abs().mean() if diff_h.numel() > 0 else low_res_map.new_tensor(0.0)
    loss_v = diff_v.abs().mean() if diff_v.numel() > 0 else low_res_map.new_tensor(0.0)
    return loss_h + loss_v


def _second_order_regularization(low_res_map: Tensor) -> Tensor:
    diff2_h = low_res_map[:, :, :, 2:] - 2.0 * low_res_map[:, :, :, 1:-1] + low_res_map[:, :, :, :-2]
    diff2_v = low_res_map[:, :, 2:, :] - 2.0 * low_res_map[:, :, 1:-1, :] + low_res_map[:, :, :-2, :]
    loss2_h = diff2_h.abs().mean() if diff2_h.numel() > 0 else low_res_map.new_tensor(0.0)
    loss2_v = diff2_v.abs().mean() if diff2_v.numel() > 0 else low_res_map.new_tensor(0.0)
    return loss2_h + loss2_v


def _build_interpolator(
    low_res_height: int,
    low_res_width: int,
    high_res_height: int,
    high_res_width: int,
    mode: str,
    gaussian_radius: int,
    gaussian_sigma: float,
    gaussian_extra_cells: int,
    device: torch.device,
) -> torch.nn.Module:
    return control_map_interpolator(
        InterpolationFieldConfig(
            low_res_height=low_res_height,
            low_res_width=low_res_width,
            high_res_height=high_res_height,
            high_res_width=high_res_width,
            mode=mode,
            align_corners=False,
            gaussian_radius=int(gaussian_radius),
            gaussian_sigma=float(gaussian_sigma),
            gaussian_extra_cells=int(gaussian_extra_cells),
        )
    ).to(device=device)


def _adv_from_random_gradient(
    grad: Tensor,
    lam1: float,
    lam2: float,
    fixed_output_mean: float,
    steps1: int,
    steps2: int,
    step_size: float,
    highres_interpolator: torch.nn.Module,
) -> Tensor:
    if steps1 != 1:
        raise ValueError(f"this generator fixes outer adversarial steps1 to 1, got {steps1}")
    if steps2 <= 0:
        raise ValueError(f"steps2 must be > 0, got {steps2}")

    fixed_grad, _ = _ensure_bchw(grad)
    if fixed_grad.shape[1] != 1:
        fixed_grad = fixed_grad.mean(dim=1, keepdim=True)
    fixed_grad = fixed_grad.detach()
    fixed_grad = fixed_grad - fixed_grad.mean(dim=(2, 3), keepdim=True)
    grad_std = fixed_grad.std(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    fixed_grad = fixed_grad * (float(FIXED_GRAD_TARGET_STD) / grad_std)

    b, _, h, w = fixed_grad.shape
    if not hasattr(highres_interpolator, "low_res_height") or not hasattr(highres_interpolator, "low_res_width"):
        raise ValueError("highres_interpolator must expose low_res_height/low_res_width attributes")
    low_res_height = int(highres_interpolator.low_res_height)
    low_res_width = int(highres_interpolator.low_res_width)
    high_res_size = float(h * w)
    low_res_size = float(low_res_height * low_res_width)
    if low_res_size <= 0:
        raise ValueError(f"low_res_size must be > 0, got {low_res_size}")
    lambda_scale_factor = float(high_res_size / low_res_size)
    effective_lam1 = float(lam1) / lambda_scale_factor
    effective_lam2 = float(lam2) / lambda_scale_factor

    low_res_param = torch.nn.Parameter(
        torch.zeros((1, 1, int(low_res_height), int(low_res_width)), dtype=fixed_grad.dtype, device=fixed_grad.device)
    )
    optimizer = torch.optim.Adam([low_res_param], lr=float(step_size))
    target_mean = float(fixed_output_mean)

    for _ in range(int(steps2)):
        optimizer.zero_grad(set_to_none=True)
        rho_high = highres_interpolator(low_res_param).expand(b, 1, h, w)
        current_mean = rho_high.mean(dim=(2, 3), keepdim=True)
        rho_high = rho_high + (target_mean - current_mean)

        surrogate_task = (rho_high * fixed_grad).mean()
        reg_first = _first_order_regularization(low_res_param)
        reg_second = _second_order_regularization(low_res_param)
        surrogate_obj = surrogate_task - effective_lam1 * reg_first - effective_lam2 * reg_second
        (-surrogate_obj).backward()
        optimizer.step()

    with torch.no_grad():
        rho_high = highres_interpolator(low_res_param).expand(b, 1, h, w)
        current_mean = rho_high.mean(dim=(2, 3), keepdim=True)
        rho_high = rho_high + (target_mean - current_mean)
    return rho_high.relu(), lambda_scale_factor, effective_lam1, effective_lam2


def _resize_bchw_to_hw(image_bchw: Tensor, out_hw: tuple[int, int]) -> Tensor:
    _, _, h, w = image_bchw.shape
    target_h, target_w = int(out_hw[0]), int(out_hw[1])
    if (h, w) == (target_h, target_w):
        return image_bchw
    return F.interpolate(image_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {device_name}")


def generate_random_adversarial_dataset(
    n: int,
    output_root: Path,
    manifest_path: Path,
    dataset_root: Path,
    seed: int,
    low_res_side_min: int,
    low_res_side_max: int,
    steps1: int,
    steps2: int,
    step_size: float,
    lam1: float,
    lam2: float,
    gaussian_radius_min: int,
    gaussian_radius_max: int,
    gaussian_sigma_min: float,
    gaussian_sigma_max: float,
    gaussian_extra_cells_min: int,
    gaussian_extra_cells_max: int,
    fixed_output_mean_min: float,
    fixed_output_mean_max: float,
    density_plot_count: int,
    device_name: str,
) -> dict[str, object]:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if low_res_side_min <= 0 or low_res_side_max <= 0:
        raise ValueError(f"low_res_side range must be > 0, got {(low_res_side_min, low_res_side_max)}")
    if low_res_side_min > low_res_side_max:
        raise ValueError(f"low_res_side_min must be <= low_res_side_max, got {(low_res_side_min, low_res_side_max)}")
    if gaussian_radius_min <= 0 or gaussian_radius_max <= 0:
        raise ValueError(f"gaussian_radius range must be > 0, got {(gaussian_radius_min, gaussian_radius_max)}")
    if gaussian_radius_min > gaussian_radius_max:
        raise ValueError(
            f"gaussian_radius_min must be <= gaussian_radius_max, got {(gaussian_radius_min, gaussian_radius_max)}"
        )
    if gaussian_sigma_min <= 0 or gaussian_sigma_max <= 0:
        raise ValueError(f"gaussian_sigma range must be > 0, got {(gaussian_sigma_min, gaussian_sigma_max)}")
    if gaussian_sigma_min > gaussian_sigma_max:
        raise ValueError(
            f"gaussian_sigma_min must be <= gaussian_sigma_max, got {(gaussian_sigma_min, gaussian_sigma_max)}"
        )
    if gaussian_extra_cells_min < 0 or gaussian_extra_cells_max < 0:
        raise ValueError(
            f"gaussian_extra_cells range must be >= 0, got {(gaussian_extra_cells_min, gaussian_extra_cells_max)}"
        )
    if gaussian_extra_cells_min > gaussian_extra_cells_max:
        raise ValueError(
            "gaussian_extra_cells_min must be <= gaussian_extra_cells_max, "
            f"got {(gaussian_extra_cells_min, gaussian_extra_cells_max)}"
        )
    if fixed_output_mean_min <= 0 or fixed_output_mean_max <= 0:
        raise ValueError("fixed_output_mean range must be > 0")
    if fixed_output_mean_min > fixed_output_mean_max:
        raise ValueError(
            f"fixed_output_mean_min must be <= fixed_output_mean_max, got {(fixed_output_mean_min, fixed_output_mean_max)}"
        )

    _seed_everything(seed)
    rng = random.Random(seed)
    device = _resolve_device(device_name)

    clear_paths = _collect_train_clear_images(manifest_path=manifest_path, dataset_root=dataset_root)
    size_ref_path = rng.choice(clear_paths)
    size_ref_image = _load_rgb_tensor(size_ref_path)
    _, _, ref_h, ref_w = size_ref_image.shape
    out_hw = (int(ref_h), int(ref_w))

    output_root.mkdir(parents=True, exist_ok=True)
    input_dir = output_root / "input"
    target_dir = output_root / "target"
    density_dir = output_root / "density"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)

    density_plot_count = max(0, min(int(density_plot_count), int(n)))
    records: list[dict[str, object]] = []

    for idx in range(int(n)):
        source_clear = rng.choice(clear_paths)
        i_gt = _load_rgb_tensor(source_clear)
        i_gt = _resize_bchw_to_hw(i_gt, out_hw=out_hw).to(device)

        low_res_side = int(rng.randint(int(low_res_side_min), int(low_res_side_max)))
        gaussian_radius = int(rng.randint(int(gaussian_radius_min), int(gaussian_radius_max)))
        gaussian_sigma = float(rng.uniform(float(gaussian_sigma_min), float(gaussian_sigma_max)))
        gaussian_extra_cells = int(rng.randint(int(gaussian_extra_cells_min), int(gaussian_extra_cells_max)))

        highres_interpolator = _build_interpolator(
            low_res_height=low_res_side,
            low_res_width=low_res_side,
            high_res_height=out_hw[0],
            high_res_width=out_hw[1],
            mode="gaussian",
            gaussian_radius=gaussian_radius,
            gaussian_sigma=gaussian_sigma,
            gaussian_extra_cells=gaussian_extra_cells,
            device=device,
        )

        random_grad = torch.randn((1, 1, out_hw[0], out_hw[1]), dtype=i_gt.dtype, device=device)
        fixed_output_mean = rng.uniform(float(fixed_output_mean_min), float(fixed_output_mean_max))
        rho, lambda_scale_factor, effective_lam1, effective_lam2 = _adv_from_random_gradient(
            grad=random_grad,
            lam1=float(lam1),
            lam2=float(lam2),
            fixed_output_mean=float(fixed_output_mean),
            steps1=int(steps1),
            steps2=int(steps2),
            step_size=float(step_size),
            highres_interpolator=highres_interpolator,
        )
        i_deg = deg(i_gt=i_gt, rho=rho)

        image_id = f"{idx:06d}"
        input_path = input_dir / f"{image_id}.png"
        target_path = target_dir / f"{image_id}.png"
        _save_rgb_tensor(i_deg.detach().cpu(), input_path)
        _save_rgb_tensor(i_gt.detach().cpu(), target_path)

        density_path: Path | None = None
        if idx < density_plot_count:
            density_path = density_dir / f"{image_id}.png"
            _save_density_tensor(rho.detach().cpu(), density_path)

        records.append(
            {
                "id": image_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "source_clear_path": str(source_clear),
                "size_reference_path": str(size_ref_path),
                "output_hw": [int(out_hw[0]), int(out_hw[1])],
                "low_res_height": int(low_res_side),
                "low_res_width": int(low_res_side),
                "steps1": int(steps1),
                "steps2": int(steps2),
                "lambda_scale_factor": float(lambda_scale_factor),
                "lam1_base": float(lam1),
                "lam2_base": float(lam2),
                "lam1_effective": float(effective_lam1),
                "lam2_effective": float(effective_lam2),
                "fixed_output_mean": float(fixed_output_mean),
                "interp_mode": "gaussian",
                "gaussian_radius": int(gaussian_radius),
                "gaussian_sigma": float(gaussian_sigma),
                "gaussian_extra_cells": int(gaussian_extra_cells),
                "density_path": None if density_path is None else str(density_path),
                "gradient_source": "random_normal",
            }
        )

        if (idx + 1) % 64 == 0 or (idx + 1) == n:
            print(f"[random_adv_ots_dataset_builder] generated {idx + 1}/{n}")

    manifest = {
        "count": len(records),
        "seed": int(seed),
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "size_reference_path": str(size_ref_path),
        "size_reference_hw": [int(out_hw[0]), int(out_hw[1])],
        "records": records,
    }
    summary = {
        "count": len(records),
        "output_root": str(output_root),
        "input_dir": str(input_dir),
        "target_dir": str(target_dir),
        "density_dir": str(density_dir),
        "size_reference_path": str(size_ref_path),
        "size_reference_hw": [int(out_hw[0]), int(out_hw[1])],
        "interp_mode": "gaussian",
        "low_res_side_min": int(low_res_side_min),
        "low_res_side_max": int(low_res_side_max),
        "gaussian_radius_min": int(gaussian_radius_min),
        "gaussian_radius_max": int(gaussian_radius_max),
        "gaussian_sigma_min": float(gaussian_sigma_min),
        "gaussian_sigma_max": float(gaussian_sigma_max),
        "gaussian_extra_cells_min": int(gaussian_extra_cells_min),
        "gaussian_extra_cells_max": int(gaussian_extra_cells_max),
        "steps1": int(steps1),
        "steps2": int(steps2),
        "step_size": float(step_size),
        "lam1_base": float(lam1),
        "lam2_base": float(lam2),
        "lambda_scaling_rule": "effective_lambda = base_lambda / ((high_res_height*high_res_width)/(low_res_height*low_res_width))",
        "fixed_output_mean_min": float(fixed_output_mean_min),
        "fixed_output_mean_max": float(fixed_output_mean_max),
        "fixed_grad_target_std": float(FIXED_GRAD_TARGET_STD),
        "density_plot_count": int(density_plot_count),
        "device": str(device),
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random adversarial haze dataset by: "
            "sampling one random OTS image size, creating random gradient maps, "
            "then running gaussian-interpolated adversarial surrogate optimization "
            "with steps1=1 and steps2=64."
        )
    )
    parser.add_argument("--n", type=int, default=1024, help="Number of generated pairs.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output root path.")
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_TRAIN_MANIFEST), help="OTS train manifest path.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Dataset root path.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--low_res_side_min", type=int, default=16, help="Per-sample random low-res map side minimum.")
    parser.add_argument("--low_res_side_max", type=int, default=32, help="Per-sample random low-res map side maximum.")
    parser.add_argument("--steps1", type=int, default=1, help="Outer adversarial step count (fixed to 1).")
    parser.add_argument("--steps2", type=int, default=64, help="Inner adversarial step count.")
    parser.add_argument("--step_size", type=float, default=1e-3, help="Inner optimizer learning rate.")
    parser.add_argument("--lam1", type=float, default=1e-4, help="First-order regularization weight.")
    parser.add_argument("--lam2", type=float, default=2e-4, help="Second-order regularization weight.")
    parser.add_argument("--gaussian_radius_min", type=int, default=3, help="Per-sample random gaussian radius minimum.")
    parser.add_argument("--gaussian_radius_max", type=int, default=6, help="Per-sample random gaussian radius maximum.")
    parser.add_argument("--gaussian_sigma_min", type=float, default=0.9, help="Per-sample random gaussian sigma minimum.")
    parser.add_argument("--gaussian_sigma_max", type=float, default=1.6, help="Per-sample random gaussian sigma maximum.")
    parser.add_argument(
        "--gaussian_extra_cells_min",
        type=int,
        default=1,
        help="Per-sample random gaussian extra-cells minimum.",
    )
    parser.add_argument(
        "--gaussian_extra_cells_max",
        type=int,
        default=4,
        help="Per-sample random gaussian extra-cells maximum.",
    )
    parser.add_argument("--fixed_output_mean_min", type=float, default=0.1, help="Min fixed output mean.")
    parser.add_argument("--fixed_output_mean_max", type=float, default=0.5, help="Max fixed output mean.")
    parser.add_argument("--density_plot_count", type=int, default=32, help="Number of density visualizations to save.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_random_adversarial_dataset(
        n=int(args.n),
        output_root=Path(args.output_root),
        manifest_path=Path(args.manifest_path),
        dataset_root=Path(args.dataset_root),
        seed=int(args.seed),
        low_res_side_min=int(args.low_res_side_min),
        low_res_side_max=int(args.low_res_side_max),
        steps1=int(args.steps1),
        steps2=int(args.steps2),
        step_size=float(args.step_size),
        lam1=float(args.lam1),
        lam2=float(args.lam2),
        gaussian_radius_min=int(args.gaussian_radius_min),
        gaussian_radius_max=int(args.gaussian_radius_max),
        gaussian_sigma_min=float(args.gaussian_sigma_min),
        gaussian_sigma_max=float(args.gaussian_sigma_max),
        gaussian_extra_cells_min=int(args.gaussian_extra_cells_min),
        gaussian_extra_cells_max=int(args.gaussian_extra_cells_max),
        fixed_output_mean_min=float(args.fixed_output_mean_min),
        fixed_output_mean_max=float(args.fixed_output_mean_max),
        density_plot_count=int(args.density_plot_count),
        device_name=str(args.device),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

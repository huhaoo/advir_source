from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "PromptIR" / "data_dir" / "hazy" / "hazy_outside_train.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp_demo" / "haze_dataset_builder"
FIXED_GRAD_TARGET_STD = 9.64294260850096e-05


def _ensure_bchw(image: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"image must be torch.Tensor, got {type(image)!r}")
    if image.ndim == 3:
        if image.shape[0] not in {1, 3}:
            raise ValueError(f"3D image must be (C,H,W) with C in {{1,3}}, got {tuple(image.shape)}")
        return image.unsqueeze(0), True
    if image.ndim == 4:
        return image, False
    raise ValueError(f"image must be (C,H,W) or (B,C,H,W), got {tuple(image.shape)}")


def _load_rgb_tensor(path: Path) -> Tensor:
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()


def _save_rgb_tensor(image: Tensor, path: Path) -> None:
    image_bchw, _ = _ensure_bchw(image)
    image_np = image_bchw[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image_np * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_density_tensor(density: Tensor, path: Path) -> None:
    density_bchw, _ = _ensure_bchw(density)
    if density_bchw.shape[1] != 1:
        density_bchw = density_bchw.mean(dim=1, keepdim=True)

    rho = density_bchw[0, 0].detach().cpu().float().numpy()
    rho_min = float(rho.min())
    rho_max = float(rho.max())
    rho_mean = float(rho.mean())
    if rho_max > rho_min:
        rho_norm = (rho - rho_min) / (rho_max - rho_min)
    else:
        rho_norm = np.zeros_like(rho, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    density_img = Image.fromarray((rho_norm * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(density_img)
    font = ImageFont.load_default()
    text = f"min={rho_min:.4f} max={rho_max:.4f} mean={rho_mean:.4f}"

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 3
    rect = (0, 0, text_w + 2 * pad, text_h + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)

    density_img.save(path)


def _collect_train_clear_images(manifest_path: Path, dataset_root: Path) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    clear_paths: list[Path] = []
    seen: set[str] = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel == "":
                continue
            haze_path = (dataset_root / rel).resolve()
            if not haze_path.exists():
                continue
            haze_key = "/haze/reside_ots/haze/"
            haze_str = str(haze_path)
            if haze_key not in haze_str:
                continue

            clear_str = haze_str.replace(haze_key, "/haze/reside_ots/clear/")
            clear_path = Path(clear_str)
            stem = clear_path.stem.split("_")[0]
            clear_path = clear_path.with_name(stem + clear_path.suffix)
            if not clear_path.exists():
                continue
            key = str(clear_path)
            if key in seen:
                continue
            seen.add(key)
            clear_paths.append(clear_path)

    if len(clear_paths) == 0:
        raise RuntimeError("no valid clear images were resolved from train manifest")
    return clear_paths


def _estimate_distance_from_image(image_bchw: Tensor) -> Tensor:
    # Distance proxy is computed from luminance only, keeping API deg(Igt)->Ideg independent from depth files.
    if image_bchw.shape[1] == 3:
        gray = 0.299 * image_bchw[:, 0:1] + 0.587 * image_bchw[:, 1:2] + 0.114 * image_bchw[:, 2:3]
    else:
        gray = image_bchw[:, 0:1]

    gmin = gray.amin(dim=(2, 3), keepdim=True)
    gmax = gray.amax(dim=(2, 3), keepdim=True)
    norm = (gray - gmin) / (gmax - gmin + 1e-6)
    return 0.1 + 2.8 * (1.0 - norm)


def _random_border_crop(
    image_bchw: Tensor,
    rng: random.Random,
    max_border_crop: int = 16,
) -> tuple[Tensor, dict[str, int]]:
    if max_border_crop < 0:
        raise ValueError(f"max_border_crop must be >= 0, got {max_border_crop}")

    _, _, h, w = image_bchw.shape
    if h <= 1 or w <= 1 or max_border_crop == 0:
        return image_bchw, {"top": 0, "bottom": 0, "left": 0, "right": 0}

    top = rng.randint(0, min(max_border_crop, h - 1))
    bottom = rng.randint(0, min(max_border_crop, h - top - 1))
    left = rng.randint(0, min(max_border_crop, w - 1))
    right = rng.randint(0, min(max_border_crop, w - left - 1))

    h_end = h - bottom if bottom > 0 else h
    w_end = w - right if right > 0 else w
    cropped = image_bchw[:, :, top:h_end, left:w_end].contiguous()
    return cropped, {"top": top, "bottom": bottom, "left": left, "right": right}


def _first_order_regularization(low_res_map: Tensor) -> Tensor:
    if low_res_map.ndim != 4:
        raise ValueError(f"low_res_map must be 4D, got {tuple(low_res_map.shape)}")

    diff_h = low_res_map[:, :, :, 1:] - low_res_map[:, :, :, :-1]
    diff_v = low_res_map[:, :, 1:, :] - low_res_map[:, :, :-1, :]
    loss_h = diff_h.abs().mean() if diff_h.numel() > 0 else low_res_map.new_tensor(0.0)
    loss_v = diff_v.abs().mean() if diff_v.numel() > 0 else low_res_map.new_tensor(0.0)
    return loss_h + loss_v


def _second_order_regularization(low_res_map: Tensor) -> Tensor:
    if low_res_map.ndim != 4:
        raise ValueError(f"low_res_map must be 4D, got {tuple(low_res_map.shape)}")

    diff2_h = low_res_map[:, :, :, 2:] - 2.0 * low_res_map[:, :, :, 1:-1] + low_res_map[:, :, :, :-2]
    diff2_v = low_res_map[:, :, 2:, :] - 2.0 * low_res_map[:, :, 1:-1, :] + low_res_map[:, :, :-2, :]
    loss2_h = diff2_h.abs().mean() if diff2_h.numel() > 0 else low_res_map.new_tensor(0.0)
    loss2_v = diff2_v.abs().mean() if diff2_v.numel() > 0 else low_res_map.new_tensor(0.0)
    return loss2_h + loss2_v


def adv(
    grad: Tensor | tuple[Tensor, ...] | list[Tensor],
    lam1: float,
    lam2: float,
    size: tuple[int, int],
    fixed_output_mean: float = 0.2,
) -> Tensor:
    """Helper for future density implementation.

    Current behavior: build an adversarial-like density map from gradient map
    using the first-order surrogate style aligned with
    run_single_image_adversarial_degradation_search:
    - steps1 is fixed to 1
    - fixed gradient uses the first input grad (former/first)
    - fixed_grad is rescaled to static gradient scale (std≈9.64e-05 from doc/static.txt)
    - inner optimization updates low-res density parameters

    - lam1: first-order regularization strength
    - lam2: second-order regularization strength
    - fixed_output_mean: enforce output matrix mean to this constant value
    This helper is not used by the default generation path.
    """
    if not isinstance(lam1, (int, float)):
        raise ValueError(f"lam1 must be numeric, got {type(lam1)!r}")
    if not isinstance(lam2, (int, float)):
        raise ValueError(f"lam2 must be numeric, got {type(lam2)!r}")
    if float(lam1) < 0:
        raise ValueError(f"lam1 must be >= 0, got {lam1}")
    if float(lam2) < 0:
        raise ValueError(f"lam2 must be >= 0, got {lam2}")
    if not isinstance(fixed_output_mean, (int, float)):
        raise ValueError(f"fixed_output_mean must be numeric, got {type(fixed_output_mean)!r}")
    if float(fixed_output_mean) <= 0:
        raise ValueError(f"fixed_output_mean must be > 0, got {fixed_output_mean}")

    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError(f"size must be a tuple(height, width), got {size!r}")
    low_h, low_w = int(size[0]), int(size[1])
    if low_h <= 0 or low_w <= 0:
        raise ValueError(f"size values must be positive, got {(low_h, low_w)}")

    grad_first = grad
    if isinstance(grad, (tuple, list)):
        if len(grad) == 0:
            raise ValueError("grad tuple/list must be non-empty")
        grad_first = grad[0]
    if not isinstance(grad_first, torch.Tensor):
        raise ValueError(f"grad first item must be torch.Tensor, got {type(grad_first)!r}")

    fixed_grad, _ = _ensure_bchw(grad_first)
    if fixed_grad.shape[1] != 1:
        fixed_grad = fixed_grad.mean(dim=1, keepdim=True)
    fixed_grad = fixed_grad.detach()
    fixed_grad = fixed_grad - fixed_grad.mean(dim=(2, 3), keepdim=True)
    grad_std = fixed_grad.std(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    fixed_grad = fixed_grad * (float(FIXED_GRAD_TARGET_STD) / grad_std)

    b, _, h, w = fixed_grad.shape
    low_res_param = torch.nn.Parameter(torch.zeros((1, 1, low_h, low_w), dtype=fixed_grad.dtype, device=fixed_grad.device))
    optimizer = torch.optim.Adam([low_res_param], lr=1e-1)
    target_mean = float(fixed_output_mean)

    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        rho_low = low_res_param
        rho_high = F.interpolate(rho_low, size=(h, w), mode="bicubic", align_corners=False).expand(b, 1, h, w)
        # Keep optimization under fixed-mean constraint by additive projection.
        current_mean = rho_high.mean(dim=(2, 3), keepdim=True)
        rho_high = rho_high + (target_mean - current_mean)

        surrogate_task = (rho_high * fixed_grad).mean()
        reg_first = _first_order_regularization(rho_low)
        reg_second = _second_order_regularization(rho_low)
        surrogate_obj = surrogate_task - float(lam1) * reg_first - float(lam2) * reg_second
        (-surrogate_obj).backward()
        optimizer.step()

    with torch.no_grad():
        rho_high = F.interpolate(low_res_param, size=(h, w), mode="bicubic", align_corners=False).expand(b, 1, h, w)
        # Fix output matrix mean exactly by additive projection.
        current_mean = rho_high.mean(dim=(2, 3), keepdim=True)
        rho_high = rho_high + (target_mean - current_mean)
    return rho_high.relu()


def density(i_gt: Tensor) -> Tensor:
    """Implementation interface: return fog density map rho with shape (B,1,H,W).

    Current default uses a deterministic 2D function with:
    - amplitude upper-bounded by a random amp_max in [0.3, 0.5] via sigmoid scaling
    - higher density near borders
    - lower density near image center
    - adaptive ellipse-like contour, with inward notches at long-side centers
    - monotonic fade from outer side to inner side along the deformed ellipse normal
    - one adaptive contour level is computed to touch all four borders
    - profile parameters are sampled randomly within predefined ranges per call
    """
    i_gt_bchw, _ = _ensure_bchw(i_gt)
    b, _, h, w = i_gt_bchw.shape

    yy = torch.linspace(-1.0, 1.0, steps=h, dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, steps=w, dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).view(1, 1, 1, w)
    xx = xx.expand(b, 1, h, w)
    yy = yy.expand(b, 1, h, w)

    # Base adaptive ellipse metric: it naturally adapts to width/height.
    q_base = torch.sqrt(xx * xx + yy * yy + 1e-12)

    # Inward notches at long-side centers:
    # - if width >= height: long sides are top/bottom
    # - else: long sides are left/right
    # Random profile parameters (sampled once per density call).
    # 1) edge_inset: inward offset from the edge-touch contour (controls distance to border).
    # 2) q_temp: sigmoid temperature (controls fade speed; smaller = steeper).
    # 3) notch_strength: inward notch magnitude at long-side centers.
    # 4) amp_max: random upper-bound amplitude for rho.
    edge_inset = float(torch.empty((), dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).uniform_(0.02, 0.14).item())
    q_temp = float(torch.empty((), dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).uniform_(0.2, 0.8).item())
    notch_strength = float(torch.empty((), dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).uniform_(0.10, 0.32).item())
    amp_max = float(torch.empty((), dtype=i_gt_bchw.dtype, device=i_gt_bchw.device).uniform_(0.15, 0.3).item())

    if w >= h:
        notch_sigma_tangent = 0.22
        notch_sigma_normal = 0.14
        notch_top = torch.exp(-(xx * xx) / (2.0 * notch_sigma_tangent * notch_sigma_tangent)) * torch.exp(
            -((yy + 1.0) ** 2) / (2.0 * notch_sigma_normal * notch_sigma_normal)
        )
        notch_bottom = torch.exp(-(xx * xx) / (2.0 * notch_sigma_tangent * notch_sigma_tangent)) * torch.exp(
            -((yy - 1.0) ** 2) / (2.0 * notch_sigma_normal * notch_sigma_normal)
        )
        notch = notch_strength * (notch_top + notch_bottom)
    else:
        notch_sigma_tangent = 0.22
        notch_sigma_normal = 0.14
        notch_left = torch.exp(-(yy * yy) / (2.0 * notch_sigma_tangent * notch_sigma_tangent)) * torch.exp(
            -((xx + 1.0) ** 2) / (2.0 * notch_sigma_normal * notch_sigma_normal)
        )
        notch_right = torch.exp(-(yy * yy) / (2.0 * notch_sigma_tangent * notch_sigma_tangent)) * torch.exp(
            -((xx - 1.0) ** 2) / (2.0 * notch_sigma_normal * notch_sigma_normal)
        )
        notch = notch_strength * (notch_left + notch_right)
    q_deformed = q_base + notch

    # Compute an adaptive edge-touch level so one contour line touches all four borders.
    top_min = q_deformed[:, :, 0, :].amin(dim=-1, keepdim=True)
    bottom_min = q_deformed[:, :, -1, :].amin(dim=-1, keepdim=True)
    left_min = q_deformed[:, :, :, 0].amin(dim=-1, keepdim=True)
    right_min = q_deformed[:, :, :, -1].amin(dim=-1, keepdim=True)
    q_touch = torch.maximum(torch.maximum(top_min, bottom_min), torch.maximum(left_min, right_min)).unsqueeze(-1)

    # Monotonic outer->inner fade:
    # larger q_deformed (outer side) -> larger density, smaller q_deformed (inner side) -> smaller density.
    q_mid = q_touch - edge_inset
    # Use sigmoid profile and scale it to a random max in [0.3, 0.5], without clamp.
    rho_sigmoid = torch.sigmoid((q_deformed - q_mid) / q_temp)
    rho_sigmoid_max = rho_sigmoid.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    rho = amp_max * (rho_sigmoid / rho_sigmoid_max)
    rho=adv(rho, 1e-5, 2e-5,rho.shape[-2:],amp_max)
    return rho


def deg(
    i_gt: Tensor,
    rho: Tensor | None = None,
) -> Tensor:
    """External interface: deg(Igt)->Ideg, implemented via density map."""
    i_gt_bchw, was_3d = _ensure_bchw(i_gt)
    i_gt_bchw = i_gt_bchw.clamp(0.0, 1.0)

    if rho is None:
        rho = density(i_gt=i_gt_bchw)
    rho_bchw, _ = _ensure_bchw(rho)

    if rho_bchw.shape[0] != i_gt_bchw.shape[0]:
        raise ValueError(f"density batch mismatch: rho={rho_bchw.shape[0]}, i_gt={i_gt_bchw.shape[0]}")
    if rho_bchw.shape[2:] != i_gt_bchw.shape[2:]:
        rho_bchw = F.interpolate(rho_bchw, size=i_gt_bchw.shape[2:], mode="bilinear", align_corners=False)
    if rho_bchw.shape[1] not in {1, i_gt_bchw.shape[1]}:
        raise ValueError(
            f"density channel must be 1 or C={i_gt_bchw.shape[1]}, got {rho_bchw.shape[1]}"
        )
    if rho_bchw.shape[1] == 1 and i_gt_bchw.shape[1] > 1:
        rho_bchw = rho_bchw.expand(-1, i_gt_bchw.shape[1], -1, -1)

    distance_map = _estimate_distance_from_image(i_gt_bchw).to(i_gt_bchw.device)
    transmission = torch.exp(-rho_bchw.clamp_min(0.0) * distance_map)
    airlight = torch.ones_like(i_gt_bchw)
    i_deg = i_gt_bchw * transmission + airlight * (1.0 - transmission)
    i_deg = i_deg.clamp(0.0, 1.0)
    if was_3d:
        return i_deg[0]
    return i_deg


def generate_haze_dataset_from_train(
    n: int,
    output_root: Path,
    manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    seed: int = 123,
    max_border_crop: int = 16,
    density_plot_count: int | None = None,
) -> dict[str, object]:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    rng = random.Random(seed)
    clear_paths = _collect_train_clear_images(manifest_path=manifest_path, dataset_root=dataset_root)

    if len(clear_paths) >= n:
        selected = rng.sample(clear_paths, k=n)
    else:
        selected = [rng.choice(clear_paths) for _ in range(n)]

    if density_plot_count is None:
        density_plot_count = max(1, n // 10)
    density_plot_count = max(0, min(int(density_plot_count), n))

    input_dir = output_root / "input"
    target_dir = output_root / "target"
    density_dir = output_root / "density"
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for idx, clear_path in enumerate(selected):
        i_gt = _load_rgb_tensor(clear_path)
        i_gt, crop_info = _random_border_crop(i_gt, rng=rng, max_border_crop=max_border_crop)
        # Use exactly one density() call for consistency when density is stochastic.
        rho = density(i_gt=i_gt).detach().clone()
        i_deg = deg(i_gt=i_gt, rho=rho)

        i_deg_bchw, _ = _ensure_bchw(i_deg)
        if i_deg_bchw.shape != i_gt.shape:
            raise RuntimeError(
                f"deg output shape mismatch for {clear_path.name}: expected {tuple(i_gt.shape)}, got {tuple(i_deg_bchw.shape)}"
            )

        image_id = f"{idx:06d}"
        input_path = input_dir / f"{image_id}.png"
        target_path = target_dir / f"{image_id}.png"
        _save_rgb_tensor(i_deg_bchw, input_path)
        _save_rgb_tensor(i_gt, target_path)

        density_path: Path | None = None
        if idx < density_plot_count:
            density_path = density_dir / f"{image_id}.png"
            _save_density_tensor(rho, density_path)

        records.append(
            {
                "id": image_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "source_clear_path": str(clear_path),
                "crop": crop_info,
                "density_path": None if density_path is None else str(density_path),
                "density_sample_calls": 1,
            }
        )

    manifest = {
        "count": len(records),
        "seed": seed,
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "records": records,
    }
    summary = {
        "count": len(records),
        "unique_source_clears": len({r["source_clear_path"] for r in records}),
        "output_root": str(output_root),
        "interface": "deg(Igt)->Ideg",
        "implementation_interface": "density(Igt)->rho",
        "max_border_crop": int(max_border_crop),
        "density_plot_count": int(density_plot_count),
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build haze dataset from train Igt using deg(Igt)->Ideg.")
    parser.add_argument("--n", type=int, default=100, help="Number of generated pairs.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output root path.")
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_TRAIN_MANIFEST), help="Train manifest path.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Dataset root path.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for source sampling.")
    parser.add_argument(
        "--max_border_crop",
        type=int,
        default=16,
        help="Auto crop each border by random pixels in [0, max_border_crop] before generation.",
    )
    parser.add_argument(
        "--density_plot_count",
        type=int,
        default=-1,
        help="Number of density maps to save. Default uses max(1, n//10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = generate_haze_dataset_from_train(
        n=int(args.n),
        output_root=Path(args.output_root),
        manifest_path=Path(args.manifest_path),
        dataset_root=Path(args.dataset_root),
        seed=int(args.seed),
        max_border_crop=int(args.max_border_crop),
        density_plot_count=None if int(args.density_plot_count) < 0 else int(args.density_plot_count),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from control_map_interpolator import InterpolationFieldConfig, control_map_interpolator
    from util.image_io import ensure_bchw as _ensure_bchw
    from util.image_io import load_rgb_tensor as _load_rgb_tensor
    from util.image_io import save_rgb_tensor as _save_rgb_tensor
    from util.runtime import resolve_runtime_device as _resolve_runtime_device
except ModuleNotFoundError:
    from .control_map_interpolator import InterpolationFieldConfig, control_map_interpolator
    from .util.image_io import ensure_bchw as _ensure_bchw
    from .util.image_io import load_rgb_tensor as _load_rgb_tensor
    from .util.image_io import save_rgb_tensor as _save_rgb_tensor
    from .util.runtime import resolve_runtime_device as _resolve_runtime_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "PromptIR" / "data_dir" / "hazy" / "hazy_outside_test.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp_demo" / "haze_dataset_builder"
FIXED_GRAD_TARGET_STD = 9.64294260850096e-05


def _save_scalar_tensor(tensor: Tensor, path: Path, title: str) -> None:
    tensor_bchw, _ = _ensure_bchw(tensor)
    if tensor_bchw.shape[1] != 1:
        tensor_bchw = tensor_bchw.mean(dim=1, keepdim=True)

    scalar_map = tensor_bchw[0, 0].detach().cpu().float().numpy()
    v_min = float(scalar_map.min())
    v_max = float(scalar_map.max())
    v_mean = float(scalar_map.mean())
    if v_max > v_min:
        scalar_norm = (scalar_map - v_min) / (v_max - v_min)
    else:
        scalar_norm = np.zeros_like(scalar_map, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_img = Image.fromarray((scalar_norm * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(scalar_img)
    font = ImageFont.load_default()
    text = f"{title} min={v_min:.4f} max={v_max:.4f} mean={v_mean:.4f}"

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 3
    rect = (0, 0, text_w + 2 * pad, text_h + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)

    scalar_img.save(path)


def _save_density_tensor(density: Tensor, path: Path) -> None:
    _save_scalar_tensor(density, path, title="rho")


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


def _load_distance_from_mat(depth_path: Path, image_hw: tuple[int, int]) -> Tensor:
    with h5py.File(depth_path, "r") as mat:
        if "depth" not in mat:
            raise KeyError(f"key 'depth' not found in {depth_path}")
        depth_np = mat["depth"][()]

    depth = torch.from_numpy(depth_np.copy()).float().unsqueeze(0).unsqueeze(0)
    raw_h, raw_w = int(depth.shape[-2]), int(depth.shape[-1])
    h, w = int(image_hw[0]), int(image_hw[1])

    if (raw_h, raw_w) == (w, h):
        depth = depth.transpose(-2, -1).contiguous()
    if depth.shape[-2:] != (h, w):
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
    return depth


def _apply_crop_from_info(image_bchw: Tensor, crop_info: dict[str, int]) -> Tensor:
    image_bchw, _ = _ensure_bchw(image_bchw)
    _, _, h, w = image_bchw.shape
    top = int(crop_info.get("top", 0))
    bottom = int(crop_info.get("bottom", 0))
    left = int(crop_info.get("left", 0))
    right = int(crop_info.get("right", 0))

    if top < 0 or bottom < 0 or left < 0 or right < 0:
        raise ValueError(f"crop values must be >= 0, got {crop_info}")
    if top + bottom >= h or left + right >= w:
        raise ValueError(f"invalid crop_info for shape {(h, w)}: {crop_info}")

    h_end = h - bottom if bottom > 0 else h
    w_end = w - right if right > 0 else w
    return image_bchw[:, :, top:h_end, left:w_end].contiguous()


def _estimate_distance_from_image(
    image_bchw: Tensor,
    depth_map: Tensor | None = None,
) -> Tensor:
    image_bchw, _ = _ensure_bchw(image_bchw)

    if depth_map is not None:
        depth_bchw, _ = _ensure_bchw(depth_map)
        if depth_bchw.shape[0] != image_bchw.shape[0]:
            raise ValueError(
                f"depth batch mismatch: depth={depth_bchw.shape[0]}, image={image_bchw.shape[0]}"
            )
        if depth_bchw.shape[2:] != image_bchw.shape[2:]:
            depth_bchw = F.interpolate(
                depth_bchw,
                size=(int(image_bchw.shape[2]), int(image_bchw.shape[3])),
                mode="bilinear",
                align_corners=False,
            )
        if depth_bchw.shape[1] != 1:
            depth_bchw = depth_bchw.mean(dim=1, keepdim=True)

        depth_bchw = depth_bchw.clamp_min(0.0)
        dmin = depth_bchw.amin(dim=(2, 3), keepdim=True)
        dmax = depth_bchw.amax(dim=(2, 3), keepdim=True)
        norm = (depth_bchw - dmin) / (dmax - dmin + 1e-6)
        return 0.1 + 2.8 * norm

    # Fallback to luminance proxy when depth is unavailable.
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
    optimizer = torch.optim.Adam([low_res_param], lr=1e-3)
    target_mean = float(fixed_output_mean)
    highres_interpolator = control_map_interpolator(
        InterpolationFieldConfig(
            low_res_height=low_h,
            low_res_width=low_w,
            high_res_height=h,
            high_res_width=w,
            mode="gaussian",
            align_corners=False,
        )
    ).to(device=fixed_grad.device)

    for _ in range(256):
        optimizer.zero_grad(set_to_none=True)
        rho_low = low_res_param
        rho_high = highres_interpolator(rho_low).expand(b, 1, h, w)
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
        rho_high = highres_interpolator(low_res_param).expand(b, 1, h, w)
        # Fix output matrix mean exactly by additive projection.
        current_mean = rho_high.mean(dim=(2, 3), keepdim=True)
        rho_high = rho_high + (target_mean - current_mean)
    return rho_high.relu()


def proj(
    target: Tensor | tuple[Tensor, ...] | list[Tensor],
    lam1: float,
    lam2: float,
    size: tuple[int, int],
    offset: bool = False,
    offset_max: float = 0.5,
    offset_lam1: float = 0.01,
    offset_lam2: float = 0.04,
) -> Tensor:
    """Project a target map to a smooth output map.

    Objective:
        minimize |out - target|^2 + lam1 * R1(out) + lam2 * R2(out)
                + offset_lam1 * R1(offset) + offset_lam2 * R2(offset)
    where R1/R2 are first/second-order smoothness regularizers.

    The output mean is constrained to match target mean by additive projection.
    When ``offset=True``, gaussian interpolation enables learnable x/y offsets.
    """
    if not isinstance(lam1, (int, float)):
        raise ValueError(f"lam1 must be numeric, got {type(lam1)!r}")
    if not isinstance(lam2, (int, float)):
        raise ValueError(f"lam2 must be numeric, got {type(lam2)!r}")
    if float(lam1) < 0:
        raise ValueError(f"lam1 must be >= 0, got {lam1}")
    if float(lam2) < 0:
        raise ValueError(f"lam2 must be >= 0, got {lam2}")
    if not isinstance(offset, bool):
        raise ValueError(f"offset must be bool, got {type(offset)!r}")
    if not isinstance(offset_max, (int, float)):
        raise ValueError(f"offset_max must be numeric, got {type(offset_max)!r}")
    if float(offset_max) < 0:
        raise ValueError(f"offset_max must be >= 0, got {offset_max}")
    if not isinstance(offset_lam1, (int, float)):
        raise ValueError(f"offset_lam1 must be numeric, got {type(offset_lam1)!r}")
    if not isinstance(offset_lam2, (int, float)):
        raise ValueError(f"offset_lam2 must be numeric, got {type(offset_lam2)!r}")
    if float(offset_lam1) < 0:
        raise ValueError(f"offset_lam1 must be >= 0, got {offset_lam1}")
    if float(offset_lam2) < 0:
        raise ValueError(f"offset_lam2 must be >= 0, got {offset_lam2}")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError(f"size must be a tuple(height, width), got {size!r}")
    low_h, low_w = int(size[0]), int(size[1])
    if low_h <= 0 or low_w <= 0:
        raise ValueError(f"size values must be positive, got {(low_h, low_w)}")

    target_first = target
    if isinstance(target, (tuple, list)):
        if len(target) == 0:
            raise ValueError("target tuple/list must be non-empty")
        target_first = target[0]
    if not isinstance(target_first, torch.Tensor):
        raise ValueError(f"target first item must be torch.Tensor, got {type(target_first)!r}")

    fixed_target, _ = _ensure_bchw(target_first)
    if fixed_target.shape[1] != 1:
        fixed_target = fixed_target.mean(dim=1, keepdim=True)
    fixed_target = fixed_target.detach()

    b, _, h, w = fixed_target.shape
    low_res_param = torch.nn.Parameter(torch.zeros((1, 1, low_h, low_w), dtype=fixed_target.dtype, device=fixed_target.device))
    low_res_offset_xy_param: torch.nn.Parameter | None = None
    optimizer_params: list[torch.nn.Parameter] = [low_res_param]
    if offset:
        low_res_offset_xy_param = torch.nn.Parameter(
            torch.zeros((1, 2, low_h, low_w), dtype=fixed_target.dtype, device=fixed_target.device)
        )
        optimizer_params.append(low_res_offset_xy_param)

    optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
    target_mean = fixed_target.mean(dim=(2, 3), keepdim=True)
    highres_interpolator = control_map_interpolator(
        InterpolationFieldConfig(
            low_res_height=low_h,
            low_res_width=low_w,
            high_res_height=h,
            high_res_width=w,
            mode="gaussian",
            align_corners=False,
            gaussian_enable_offset=offset,
            gaussian_offset_max=float(offset_max),
        )
    ).to(device=fixed_target.device)

    for _ in range(256):
        optimizer.zero_grad(set_to_none=True)
        if low_res_offset_xy_param is None:
            out_high = highres_interpolator(low_res_param).expand(b, 1, h, w)
        else:
            out_high = highres_interpolator(low_res_param, gaussian_offset_xy=low_res_offset_xy_param).expand(b, 1, h, w)
        current_mean = out_high.mean(dim=(2, 3), keepdim=True)
        out_high = out_high + (target_mean - current_mean)

        data_loss = (out_high - fixed_target).pow(2).mean()
        reg_first = _first_order_regularization(out_high)
        reg_second = _second_order_regularization(out_high)
        reg_offset = out_high.new_tensor(0.0)
        if low_res_offset_xy_param is not None:
            reg_offset = (
                float(offset_lam1) * _first_order_regularization(low_res_offset_xy_param)
                + float(offset_lam2) * _second_order_regularization(low_res_offset_xy_param)
            )
        proj_obj = data_loss + float(lam1) * reg_first + float(lam2) * reg_second + reg_offset
        proj_obj.backward()
        optimizer.step()
        if low_res_offset_xy_param is not None:
            with torch.no_grad():
                low_res_offset_xy_param.clamp_(-float(offset_max), float(offset_max))

    with torch.no_grad():
        if low_res_offset_xy_param is None:
            out_high = highres_interpolator(low_res_param).expand(b, 1, h, w)
        else:
            out_high = highres_interpolator(low_res_param, gaussian_offset_xy=low_res_offset_xy_param).expand(b, 1, h, w)
        current_mean = out_high.mean(dim=(2, 3), keepdim=True)
        out_high = out_high + (target_mean - current_mean)
    return out_high


def density(i_gt: Tensor) -> Tensor:
    """Implementation interface: return fog density map rho with shape (B,1,H,W).

    Current default uses a deterministic 2D function with:
    - edge-heavy base map from deformed ellipse geometry
    - linear normalization (no sigmoid/proj)
    - random range remap with a+-b where a in [0.2, 0.4], b in [0.025, 0.1]
    - higher density near borders
    - lower density near image center
    - adaptive ellipse-like contour, with inward notches at long-side centers
    - monotonic fade from outer side to inner side along the deformed ellipse normal
    - one adaptive contour level is computed to touch all four borders
    - profile parameters are sampled randomly within predefined ranges per call
    """
    i_gt_bchw, _ = _ensure_bchw(i_gt)
    b, _, h, w = i_gt_bchw.shape

    dtype, device = i_gt_bchw.dtype, i_gt_bchw.device
    yy = torch.linspace(-1.0, 1.0, steps=h, dtype=dtype, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, steps=w, dtype=dtype, device=device).view(1, 1, 1, w).expand(b, 1, h, w)

    edge_inset = float(torch.empty((), dtype=dtype, device=device).uniform_(0.02, 0.14).item())
    notch_strength = float(torch.empty((), dtype=dtype, device=device).uniform_(0.10, 0.32).item())
    sigma_t, sigma_n = 0.22, 0.14
    tangent, normal = (xx, yy) if w >= h else (yy, xx)

    notch = torch.exp(-(tangent * tangent) / (2.0 * sigma_t * sigma_t))
    notch = notch * (
        torch.exp(-((normal + 1.0) ** 2) / (2.0 * sigma_n * sigma_n))
        + torch.exp(-((normal - 1.0) ** 2) / (2.0 * sigma_n * sigma_n))
    )
    q_deformed = torch.sqrt(xx * xx + yy * yy + 1e-12) + notch_strength * notch

    top_min = q_deformed[:, :, 0, :].amin(dim=-1, keepdim=True)
    bottom_min = q_deformed[:, :, -1, :].amin(dim=-1, keepdim=True)
    left_min = q_deformed[:, :, :, 0].amin(dim=-1, keepdim=True)
    right_min = q_deformed[:, :, :, -1].amin(dim=-1, keepdim=True)
    rho_score = q_deformed - (
        torch.maximum(torch.maximum(top_min, bottom_min), torch.maximum(left_min, right_min)).unsqueeze(-1) - edge_inset
    )

    score_min = rho_score.amin(dim=(2, 3), keepdim=True)
    score_delta = rho_score.amax(dim=(2, 3), keepdim=True) - score_min
    rho_norm = (rho_score - score_min) / score_delta.clamp_min(1e-12)
    rho_norm = torch.where(score_delta > 1e-12, rho_norm, torch.full_like(rho_norm, 0.5))

    a = torch.empty((b, 1, 1, 1), dtype=dtype, device=device).uniform_(0.2, 0.4)
    b_span = torch.empty((b, 1, 1, 1), dtype=dtype, device=device).uniform_(0.05, 0.15)
    res = (a - b_span) + (2.0 * b_span) * rho_norm
    # res = proj(res, 1e-3, 5e-3, size=(32, 32), offset=True)
    return res


def deg(
    i_gt: Tensor,
    rho: Tensor | None = None,
    distance_map: Tensor | None = None,
) -> Tensor:
    """External interface: deg(Igt)->Ideg, implemented via density map."""
    i_gt_bchw, was_3d = _ensure_bchw(i_gt)
    i_gt_bchw = i_gt_bchw.clamp(0.0, 1.0)

    if rho is None:
        rho = density(i_gt=i_gt_bchw)
    _, _, transmission = _prepare_degradation_components(
        i_gt_bchw=i_gt_bchw,
        rho=rho,
        distance_map=distance_map,
    )
    airlight = torch.ones_like(i_gt_bchw)
    i_deg = i_gt_bchw * transmission + airlight * (1.0 - transmission)
    i_deg = i_deg.clamp(0.0, 1.0)
    if was_3d:
        return i_deg[0]
    return i_deg


def _prepare_degradation_components(
    i_gt_bchw: Tensor,
    rho: Tensor,
    distance_map: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare rho, distance, and transmission maps used by deg."""
    i_gt_bchw, _ = _ensure_bchw(i_gt_bchw)
    rho_bchw, _ = _ensure_bchw(rho)

    if rho_bchw.shape[0] != i_gt_bchw.shape[0]:
        raise ValueError(f"density batch mismatch: rho={rho_bchw.shape[0]}, i_gt={i_gt_bchw.shape[0]}")
    if rho_bchw.shape[2:] != i_gt_bchw.shape[2:]:
        rho_interpolator = control_map_interpolator(
            InterpolationFieldConfig(
                low_res_height=int(rho_bchw.shape[2]),
                low_res_width=int(rho_bchw.shape[3]),
                high_res_height=int(i_gt_bchw.shape[2]),
                high_res_width=int(i_gt_bchw.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        ).to(device=rho_bchw.device)
        rho_bchw = rho_interpolator(rho_bchw)
    if rho_bchw.shape[1] not in {1, i_gt_bchw.shape[1]}:
        raise ValueError(
            f"density channel must be 1 or C={i_gt_bchw.shape[1]}, got {rho_bchw.shape[1]}"
        )
    if rho_bchw.shape[1] == 1 and i_gt_bchw.shape[1] > 1:
        rho_bchw = rho_bchw.expand(-1, i_gt_bchw.shape[1], -1, -1)

    distance_used = _estimate_distance_from_image(i_gt_bchw, depth_map=distance_map).to(i_gt_bchw.device)
    transmission = torch.exp(-rho_bchw.clamp_min(0.0) * distance_used)
    return rho_bchw, distance_used, transmission


def generate_haze_dataset_from_train(
    n: int,
    output_root: Path,
    manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    device: str = "cpu",
    seed: int = 123,
    max_border_crop: int = 16,
    density_plot_count: int | None = None,
) -> dict[str, object]:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    rng = random.Random(seed)
    runtime_device = _resolve_runtime_device(device)
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
    transmission_dir = output_root / "transmission"
    distance_dir = output_root / "distance"
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)
    transmission_dir.mkdir(parents=True, exist_ok=True)
    distance_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    depth_root = dataset_root / "haze" / "reside_ots" / "depth"
    for idx, clear_path in enumerate(selected):
        i_gt_full = _load_rgb_tensor(clear_path).to(device=runtime_device)
        distance_full: Tensor | None = None
        distance_source = "luminance_proxy"
        depth_path = depth_root / f"{clear_path.stem}.mat"
        if depth_path.exists():
            distance_full = _load_distance_from_mat(
                depth_path=depth_path,
                image_hw=(int(i_gt_full.shape[-2]), int(i_gt_full.shape[-1])),
            ).to(device=runtime_device)
            distance_source = "depth_mat"
        i_gt, crop_info = _random_border_crop(i_gt_full, rng=rng, max_border_crop=max_border_crop)
        distance_map: Tensor | None = None
        if distance_full is not None:
            distance_map = _apply_crop_from_info(distance_full, crop_info=crop_info)

        # Use exactly one density() call for consistency when density is stochastic.
        rho = density(i_gt=i_gt).detach().clone()
        _, distance_used, transmission = _prepare_degradation_components(
            i_gt_bchw=i_gt,
            rho=rho,
            distance_map=distance_map,
        )
        airlight = torch.ones_like(i_gt)
        i_deg = (i_gt * transmission + airlight * (1.0 - transmission)).clamp(0.0, 1.0)

        i_deg_bchw, _ = _ensure_bchw(i_deg)
        if i_deg_bchw.shape != i_gt.shape:
            raise RuntimeError(
                f"deg output shape mismatch for {clear_path.name}: expected {tuple(i_gt.shape)}, got {tuple(i_deg_bchw.shape)}"
            )

        image_id = f"{idx:06d}"
        input_path = input_dir / f"{image_id}.png"
        target_path = target_dir / f"{image_id}.png"
        transmission_path = transmission_dir / f"{image_id}.png"
        distance_path = distance_dir / f"{image_id}.png"
        _save_rgb_tensor(i_deg_bchw, input_path)
        _save_rgb_tensor(i_gt, target_path)
        _save_scalar_tensor(transmission, transmission_path, title="transmission")
        _save_scalar_tensor(distance_used, distance_path, title="distance")

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
                "depth_path": str(depth_path) if depth_path.exists() else None,
                "distance_source": distance_source,
                "crop": crop_info,
                "transmission_path": str(transmission_path),
                "distance_path": str(distance_path),
                "density_path": None if density_path is None else str(density_path),
                "density_sample_calls": 1,
            }
        )

    manifest = {
        "count": len(records),
        "seed": seed,
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "requested_device": str(device),
        "resolved_device": str(runtime_device),
        "records": records,
    }
    summary = {
        "count": len(records),
        "unique_source_clears": len({r["source_clear_path"] for r in records}),
        "depth_distance_count": int(sum(1 for r in records if r["distance_source"] == "depth_mat")),
        "luminance_distance_count": int(sum(1 for r in records if r["distance_source"] != "depth_mat")),
        "transmission_count": len(records),
        "distance_map_count": len(records),
        "output_root": str(output_root),
        "interface": "deg(Igt)->Ideg",
        "implementation_interface": "density(Igt)->rho",
        "max_border_crop": int(max_border_crop),
        "density_plot_count": int(density_plot_count),
        "requested_device": str(device),
        "resolved_device": str(runtime_device),
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="compute device: cpu, cuda, cuda:N, or auto",
    )
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
        device=str(args.device),
        seed=int(args.seed),
        max_border_crop=int(args.max_border_crop),
        density_plot_count=None if int(args.density_plot_count) < 0 else int(args.density_plot_count),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

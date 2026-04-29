from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import Tensor
from PIL import Image, ImageDraw

try:
    from control_map import ControlMapConfig
    from degradation import MotionBlurDegradationConfig, motion_blur_degradation
    from util.image_io import load_rgb_tensor as _load_rgb_tensor
    from util.image_io import save_rgb_tensor as _save_rgb_tensor
    from util.runtime import resolve_runtime_device as _resolve_runtime_device
    from util.runtime import seed_everything as _seed_everything
except ModuleNotFoundError:
    from .control_map import ControlMapConfig
    from .degradation import MotionBlurDegradationConfig, motion_blur_degradation
    from .util.image_io import load_rgb_tensor as _load_rgb_tensor
    from .util.image_io import save_rgb_tensor as _save_rgb_tensor
    from .util.runtime import resolve_runtime_device as _resolve_runtime_device
    from .util.runtime import seed_everything as _seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_JSON = PROJECT_ROOT / "dataset_path" / "promptir_clear_depth_sets_only.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset_ours" / "motion_blur_rotation0.02"


def _probe_device_available(device: torch.device) -> tuple[bool, str | None]:
    try:
        _ = torch.zeros((1,), device=device)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _load_split_clear_paths(splits_json: Path, split: str) -> list[Path]:
    if not splits_json.exists():
        raise FileNotFoundError(f"splits_json not found: {splits_json}")
    obj = json.loads(splits_json.read_text(encoding="utf-8"))
    if "splits" not in obj or split not in obj["splits"]:
        raise KeyError(f"split not found in splits_json: {split}")

    paths = obj["splits"][split].get("clear_paths", [])
    resolved: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.exists():
            resolved.append(pp)
    if len(resolved) == 0:
        raise RuntimeError(f"no valid clear paths for split={split}")
    return resolved


def _select_source_paths(clear_paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if len(clear_paths) == 0:
        raise ValueError("clear_paths is empty")

    if count <= len(clear_paths):
        return rng.sample(clear_paths, k=count)

    shuffled = list(clear_paths)
    rng.shuffle(shuffled)
    extra = [rng.choice(clear_paths) for _ in range(count - len(clear_paths))]
    return shuffled + extra


def _sample_random_alpha(alpha_min: float, alpha_max: float, rng: random.Random) -> float:
    lo = float(alpha_min)
    hi = float(alpha_max)
    if lo > hi:
        raise ValueError(f"alpha_min must be <= alpha_max, got {(alpha_min, alpha_max)}")
    return float(rng.uniform(lo, hi))


def _sample_origin_xy(origin_mode: str, h: int, w: int, rng: random.Random) -> tuple[float, float]:
    h_i = int(h)
    w_i = int(w)
    if h_i <= 0 or w_i <= 0:
        raise ValueError(f"invalid image shape for origin sampling: {(h, w)}")
    mode = str(origin_mode).strip().lower()
    if mode == "center":
        return (float(w_i) - 1.0) / 2.0, (float(h_i) - 1.0) / 2.0
    if mode == "uniform_random":
        x_hi = float(max(0, w_i - 1))
        y_hi = float(max(0, h_i - 1))
        return float(rng.uniform(0.0, x_hi)), float(rng.uniform(0.0, y_hi))
    raise ValueError(f"unsupported origin_mode: {origin_mode}")


def _build_radial_dx_map_from_alpha(
    h: int,
    w: int,
    alpha: float,
    origin_x: float,
    origin_y: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    h_i = int(h)
    w_i = int(w)
    diagonal = math.sqrt(float(h_i * h_i + w_i * w_i))
    if diagonal <= 0.0:
        raise ValueError(f"invalid image shape for diagonal: {(h, w)}")

    ox = float(origin_x)
    oy = float(origin_y)

    ys = torch.arange(h_i, dtype=dtype, device=device)
    xs = torch.arange(w_i, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x_centered = xx - ox
    y_centered = yy - oy

    # In the normalized polar space: origin is image center and diagonal length is 1.
    x_norm = x_centered / float(diagonal)
    y_norm = y_centered / float(diagonal)

    # Tangential (theta-direction) vector field:
    # v(r,theta) = alpha * r * e_theta(theta), where e_theta = [-sin(theta), cos(theta)].
    # Using x_norm=r*cos(theta), y_norm=r*sin(theta):
    # v_norm = alpha * [-y_norm, x_norm].
    dx_norm_x = -float(alpha) * y_norm
    dx_norm_y = float(alpha) * x_norm

    # Convert normalized displacement to pixel displacement used by grid_sample path integration.
    dx_x = dx_norm_x * float(diagonal)
    dx_y = dx_norm_y * float(diagonal)

    dx = torch.stack((dx_x, dx_y), dim=0).unsqueeze(0).contiguous()

    r_norm = torch.sqrt(x_norm * x_norm + y_norm * y_norm)
    dx_mag = torch.sqrt(dx_x * dx_x + dx_y * dx_y)
    stats = {
        "alpha": float(alpha),
        "origin_x": float(ox),
        "origin_y": float(oy),
        "radius_norm_max": float(r_norm.max().item()),
        "radius_norm_mean": float(r_norm.mean().item()),
        "dx_magnitude_min": float(dx_mag.min().item()),
        "dx_magnitude_max": float(dx_mag.max().item()),
        "dx_magnitude_mean": float(dx_mag.mean().item()),
    }
    return dx, stats


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: int | tuple[int, int, int],
    width: int,
    head_len: float = 4.0,
) -> None:
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = math.sqrt(dx * dx + dy * dy)
    if norm <= 1e-6:
        return
    ux = dx / norm
    uy = dy / norm
    left_x = x1 - head_len * (0.866 * ux + 0.5 * uy)
    left_y = y1 - head_len * (0.866 * uy - 0.5 * ux)
    right_x = x1 - head_len * (0.866 * ux - 0.5 * uy)
    right_y = y1 - head_len * (0.866 * uy + 0.5 * ux)
    draw.line((x1, y1, left_x, left_y), fill=color, width=width)
    draw.line((x1, y1, right_x, right_y), fill=color, width=width)


def _save_vector_field_sampled_viz(
    target_image: Tensor,
    dx: Tensor,
    out_path: Path,
    stride: int,
    max_points: int,
    arrow_scale: float,
    arrow_color: int = 255,
) -> dict[str, float]:
    _ = target_image

    dx_cpu = dx.detach().cpu()[0]
    h = int(dx_cpu.shape[1])
    w = int(dx_cpu.shape[2])
    canvas = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(canvas)
    stride_i = max(1, int(stride))
    half = stride_i // 2
    ys = list(range(half, h, stride_i))
    xs = list(range(half, w, stride_i))
    if len(ys) == 0:
        ys = [h // 2]
    if len(xs) == 0:
        xs = [w // 2]
    points = [(x, y) for y in ys for x in xs]

    max_points_i = max(1, int(max_points))
    if len(points) > max_points_i:
        keep_step = max(1, math.ceil(len(points) / float(max_points_i)))
        points = points[::keep_step]

    for x, y in points:
        vx = float(dx_cpu[0, y, x]) * float(arrow_scale)
        vy = float(dx_cpu[1, y, x]) * float(arrow_scale)
        _draw_arrow(
            draw=draw,
            x0=float(x),
            y0=float(y),
            x1=float(x) + vx,
            y1=float(y) + vy,
            color=arrow_color,
            width=2,
            head_len=4.0,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "sample_stride": float(stride_i),
        "sample_count": float(len(points)),
        "arrow_scale": float(arrow_scale),
        "viz_style": "black_background_white_arrows",
    }


def generate_motion_blur_split_dataset_radial_alpha(
    split: str,
    count: int,
    splits_json: Path,
    output_root: Path,
    seed: int,
    alpha_min: float,
    alpha_max: float,
    num_steps: int,
    batchify_steps: bool,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    device_name: str,
    progress_interval: int,
    save_viz: bool = False,
    viz_count: int = 24,
    viz_stride: int = 48,
    viz_max_points: int = 2048,
    viz_arrow_scale: float = 5.0,
    origin_mode: str = "center",
    flat_output: bool = False,
    index_offset: int = 0,
    artifact_suffix: str = "",
) -> dict[str, object]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split}")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if progress_interval <= 0:
        raise ValueError(f"progress_interval must be > 0, got {progress_interval}")
    if int(index_offset) < 0:
        raise ValueError(f"index_offset must be >= 0, got {index_offset}")
    if float(alpha_min) > float(alpha_max):
        raise ValueError(f"alpha_min must be <= alpha_max, got {(alpha_min, alpha_max)}")
    if int(viz_count) < 0:
        raise ValueError(f"viz_count must be >= 0, got {viz_count}")
    if int(viz_stride) <= 0:
        raise ValueError(f"viz_stride must be > 0, got {viz_stride}")
    if int(viz_max_points) <= 0:
        raise ValueError(f"viz_max_points must be > 0, got {viz_max_points}")
    if float(viz_arrow_scale) <= 0:
        raise ValueError(f"viz_arrow_scale must be > 0, got {viz_arrow_scale}")
    origin_mode_norm = str(origin_mode).strip().lower()
    if origin_mode_norm not in {"center", "uniform_random"}:
        raise ValueError(f"origin_mode must be center or uniform_random, got {origin_mode}")

    _seed_everything(seed)
    rng = random.Random(int(seed) + {"train": 0, "val": 1, "test": 2}[split])

    requested_device = str(device_name)
    device = _resolve_runtime_device(device_name)
    probe_ok, probe_error = _probe_device_available(device=device)
    fallback_reason: str | None = None
    if not probe_ok:
        if str(device_name).strip().lower() == "auto":
            fallback_reason = probe_error
            device = torch.device("cpu")
        else:
            raise RuntimeError(f"requested device {device} is unavailable: {probe_error}")

    clear_paths = _load_split_clear_paths(splits_json=splits_json, split=split)
    selected_paths = _select_source_paths(clear_paths=clear_paths, count=int(count), rng=rng)

    split_root = output_root if bool(flat_output) else (output_root / split)
    input_dir = split_root / "input"
    target_dir = split_root / "target"
    viz_dir = split_root / "v_field_sampled"
    split_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    if bool(save_viz) and int(viz_count) > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    blur = motion_blur_degradation(
        MotionBlurDegradationConfig(
            map_config=ControlMapConfig(1, 1, 1, 1),
            num_steps=int(num_steps),
            mode=str(mode),
            padding_mode=str(padding_mode),
            align_corners=bool(align_corners),
            batchify_steps=bool(batchify_steps),
            dmax=None,
            dlambda=0.0,
        )
    ).to(device=device)
    blur.eval()

    records: list[dict[str, object]] = []
    for idx, source_clear in enumerate(selected_paths):
        i_gt = _load_rgb_tensor(source_clear).to(device=device, dtype=torch.float32)
        _, _, h, w = i_gt.shape

        alpha_i = _sample_random_alpha(alpha_min=float(alpha_min), alpha_max=float(alpha_max), rng=rng)
        origin_x_i, origin_y_i = _sample_origin_xy(origin_mode=origin_mode_norm, h=int(h), w=int(w), rng=rng)
        dx, dx_stats = _build_radial_dx_map_from_alpha(
            h=h,
            w=w,
            alpha=float(alpha_i),
            origin_x=float(origin_x_i),
            origin_y=float(origin_y_i),
            dtype=i_gt.dtype,
            device=device,
        )

        with torch.no_grad():
            i_deg = blur(image=i_gt, dx=dx)

        image_id = f"{int(index_offset) + idx:06d}"
        input_path = input_dir / f"{image_id}.png"
        target_path = target_dir / f"{image_id}.png"
        _save_rgb_tensor(i_deg.detach().cpu(), input_path)
        _save_rgb_tensor(i_gt.detach().cpu(), target_path)

        diag = math.sqrt(float(h * h + w * w))
        viz_path: str | None = None
        viz_meta: dict[str, float] | None = None
        if bool(save_viz) and idx < int(viz_count):
            out_viz_path = viz_dir / f"{image_id}.png"
            viz_meta = _save_vector_field_sampled_viz(
                target_image=i_gt,
                dx=dx,
                out_path=out_viz_path,
                stride=int(viz_stride),
                max_points=int(viz_max_points),
                arrow_scale=float(viz_arrow_scale),
            )
            viz_path = str(out_viz_path)

        records.append(
            {
                "id": image_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "source_clear_path": str(source_clear),
                "source_hw": [int(h), int(w)],
                "source_diagonal": float(diag),
                "sampling_model": "theta_tangential_alpha_r",
                "formula": "v(r,theta)=alpha*r*e_theta(theta), alpha~U[alpha_min,alpha_max]",
                "coordinate_system": "origin=per-image-sampled, diagonal_length=1",
                "origin_mode": str(origin_mode_norm),
                "alpha": float(dx_stats["alpha"]),
                "origin_x": float(dx_stats["origin_x"]),
                "origin_y": float(dx_stats["origin_y"]),
                "radius_norm_max": float(dx_stats["radius_norm_max"]),
                "radius_norm_mean": float(dx_stats["radius_norm_mean"]),
                "dx_magnitude_min": float(dx_stats["dx_magnitude_min"]),
                "dx_magnitude_max": float(dx_stats["dx_magnitude_max"]),
                "dx_magnitude_mean": float(dx_stats["dx_magnitude_mean"]),
                "v_field_sampled_viz_path": viz_path,
                "v_field_sampled_viz_meta": viz_meta,
            }
        )

        if (idx + 1) % int(progress_interval) == 0 or (idx + 1) == int(count):
            print(f"[motion_blur_dataset_builder_radial_alpha] split={split} generated {idx + 1}/{count}")

    manifest = {
        "split": split,
        "count": int(count),
        "seed": int(seed),
        "splits_json": str(splits_json),
        "per_image_dx_sampling": {
            "sampling_model": "theta_tangential_alpha_r",
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "coordinate_system": {
                "origin": "image_center" if origin_mode_norm == "center" else "per_image_uniform_random_in_image",
                "origin_mode": str(origin_mode_norm),
                "diagonal_length_normalized": 1.0,
            },
            "formula": "v(r,theta)=alpha*r*e_theta(theta)",
            "alpha_sampling": "fixed" if float(alpha_min) == float(alpha_max) else "uniform",
            "pixel_space_conversion": "dx_pixel = alpha * [-(y-origin_y), (x-origin_x)]",
        },
        "vector_field_visualization": {
            "enabled": bool(save_viz),
            "viz_count": int(viz_count),
            "viz_stride": int(viz_stride),
            "viz_max_points": int(viz_max_points),
            "viz_arrow_scale": float(viz_arrow_scale),
            "viz_dir": str(viz_dir),
        },
        "output_layout": {
            "flat_output": bool(flat_output),
            "split_root": str(split_root),
        },
        "motion_blur": {
            "num_steps": int(num_steps),
            "batchify_steps": bool(batchify_steps),
            "mode": str(mode),
            "padding_mode": str(padding_mode),
            "align_corners": bool(align_corners),
        },
        "requested_device": str(requested_device),
        "resolved_device": str(device),
        "device_fallback_reason": fallback_reason,
        "records": records,
    }

    summary = {
        "split": split,
        "count": int(count),
        "output_root": str(split_root),
        "input_dir": str(input_dir),
        "target_dir": str(target_dir),
        "seed": int(seed),
        "requested_device": str(requested_device),
        "resolved_device": str(device),
        "device_fallback_reason": fallback_reason,
        "sampling_model": "theta_tangential_alpha_r",
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "origin_mode": str(origin_mode_norm),
        "save_viz": bool(save_viz),
        "viz_count": int(viz_count),
        "viz_stride": int(viz_stride),
        "viz_max_points": int(viz_max_points),
        "viz_arrow_scale": float(viz_arrow_scale),
        "v_field_sampled_viz_dir": str(viz_dir),
        "flat_output": bool(flat_output),
        "num_steps": int(num_steps),
        "batchify_steps": bool(batchify_steps),
        "mode": str(mode),
        "padding_mode": str(padding_mode),
        "align_corners": bool(align_corners),
        "source_pool_size": len(clear_paths),
        "sample_with_replacement": bool(count > len(clear_paths)),
    }

    suffix = str(artifact_suffix).strip()
    manifest_name = "manifest.json" if suffix == "" else f"manifest_{suffix}.json"
    summary_name = "summary.json" if suffix == "" else f"summary_{suffix}.json"
    (split_root / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (split_root / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one split of motion-blur paired dataset with radial per-pixel displacement field "
            "v(r,theta)=alpha*r*e_theta(theta) in center-origin normalized polar coordinates (diagonal=1)."
        )
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--count", type=int, required=True, help="Number of samples to generate for this split.")
    parser.add_argument("--splits_json", type=str, default=str(DEFAULT_SPLITS_JSON), help="Path to split json.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for source-path sampling.")

    parser.add_argument("--alpha_min", type=float, default=0.01, help="Per-image alpha lower bound.")
    parser.add_argument("--alpha_max", type=float, default=0.02, help="Per-image alpha upper bound.")

    parser.add_argument("--num_steps", type=int, default=16, help="Motion blur integration step count K.")
    parser.add_argument(
        "--batchify_steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to batchify K sampling steps (default: true).",
    )
    parser.add_argument("--mode", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic"])
    parser.add_argument("--padding_mode", type=str, default="border", choices=["zeros", "border", "reflection"])
    parser.add_argument(
        "--align_corners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="grid_sample align_corners flag.",
    )

    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/cuda:N/auto")
    parser.add_argument("--progress_interval", type=int, default=128, help="Progress print interval.")
    parser.add_argument(
        "--save_viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save sampled vector-field visualization images under <split>/v_field_sampled.",
    )
    parser.add_argument("--viz_count", type=int, default=24, help="Only save vector-field viz for first N samples.")
    parser.add_argument("--viz_stride", type=int, default=48, help="Sample stride for vector-field arrows.")
    parser.add_argument("--viz_max_points", type=int, default=2048, help="Max sampled points in each vector-field viz.")
    parser.add_argument("--viz_arrow_scale", type=float, default=5.0, help="Arrow length scale in vector-field viz.")
    parser.add_argument(
        "--origin_mode",
        type=str,
        default="center",
        choices=["center", "uniform_random"],
        help="Polar-origin mode: center or per-image uniform-random origin in image.",
    )
    parser.add_argument(
        "--flat_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, write directly under output_root without creating split subdirectory.",
    )
    parser.add_argument("--index_offset", type=int, default=0, help="Starting id offset for output file naming.")
    parser.add_argument(
        "--artifact_suffix",
        type=str,
        default="",
        help="Optional suffix for manifest/summary filenames to avoid overwrite in parallel shard runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_motion_blur_split_dataset_radial_alpha(
        split=str(args.split),
        count=int(args.count),
        splits_json=Path(args.splits_json),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        num_steps=int(args.num_steps),
        batchify_steps=bool(args.batchify_steps),
        mode=str(args.mode),
        padding_mode=str(args.padding_mode),
        align_corners=bool(args.align_corners),
        device_name=str(args.device),
        progress_interval=int(args.progress_interval),
        save_viz=bool(args.save_viz),
        viz_count=int(args.viz_count),
        viz_stride=int(args.viz_stride),
        viz_max_points=int(args.viz_max_points),
        viz_arrow_scale=float(args.viz_arrow_scale),
        origin_mode=str(args.origin_mode),
        flat_output=bool(args.flat_output),
        index_offset=int(args.index_offset),
        artifact_suffix=str(args.artifact_suffix),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

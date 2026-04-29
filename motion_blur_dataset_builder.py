from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import Tensor

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "motion_sim"


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


def _effective_magnitude_for_hw(dmax: float, h: int, w: int) -> float:
    del h, w
    dmax_f = float(dmax)
    return abs(dmax_f)


def _build_dx_from_params(
    h: int,
    w: int,
    dmax: float,
    theta_rad: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, float, float, float]:
    magnitude = _effective_magnitude_for_hw(dmax=dmax, h=h, w=w)
    vx = magnitude * math.cos(float(theta_rad))
    vy = magnitude * math.sin(float(theta_rad))

    dx = torch.empty((1, 2, int(h), int(w)), dtype=dtype, device=device)
    dx[:, 0:1, :, :] = float(vx)
    dx[:, 1:2, :, :] = float(vy)
    return dx, float(magnitude), float(vx), float(vy)


def _sample_random_dmax_theta(
    dmax_min: float,
    dmax_max: float,
    theta_min: float,
    theta_max: float,
    rng: random.Random,
) -> tuple[float, float]:
    lo = float(dmax_min)
    hi = float(dmax_max)
    if lo > hi:
        raise ValueError(f"dmax_min must be <= dmax_max, got {(dmax_min, dmax_max)}")
    t_lo = float(theta_min)
    t_hi = float(theta_max)
    if t_lo > t_hi:
        raise ValueError(f"theta_min must be <= theta_max, got {(theta_min, theta_max)}")
    dmax = float(rng.uniform(lo, hi))
    theta = float(rng.uniform(t_lo, t_hi))
    return dmax, theta


def generate_motion_blur_split_dataset(
    split: str,
    count: int,
    splits_json: Path,
    output_root: Path,
    seed: int,
    dmax_min: float,
    dmax_max: float,
    theta_min: float,
    theta_max: float,
    num_steps: int,
    batchify_steps: bool,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    device_name: str,
    progress_interval: int,
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

    split_root = output_root / split
    input_dir = split_root / "input"
    target_dir = split_root / "target"
    split_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

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
        dmax_i, theta_i = _sample_random_dmax_theta(
            dmax_min=float(dmax_min),
            dmax_max=float(dmax_max),
            theta_min=float(theta_min),
            theta_max=float(theta_max),
            rng=rng,
        )
        dx, dx_mag, vx, vy = _build_dx_from_params(
            h=h,
            w=w,
            dmax=float(dmax_i),
            theta_rad=float(theta_i),
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
        records.append(
            {
                "id": image_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "source_clear_path": str(source_clear),
                "source_hw": [int(h), int(w)],
                "source_diagonal": float(diag),
                "dmax": float(dmax_i),
                "theta_rad": float(theta_i),
                "theta_deg": float(theta_i * 180.0 / math.pi),
                "dx_magnitude": float(dx_mag),
                "dx_value_x": float(vx),
                "dx_value_y": float(vy),
                "dx_magnitude_diagonal_unit": float(dx_mag),
                "dx_value_x_diagonal_unit": float(vx),
                "dx_value_y_diagonal_unit": float(vy),
            }
        )

        if (idx + 1) % int(progress_interval) == 0 or (idx + 1) == int(count):
            print(f"[motion_blur_dataset_builder] split={split} generated {idx + 1}/{count}")

    manifest = {
        "split": split,
        "count": int(count),
        "seed": int(seed),
        "splits_json": str(splits_json),
        "per_image_dx_sampling": {
            "dmax_min": float(dmax_min),
            "dmax_max": float(dmax_max),
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
            "dmax_rule": "|dx| = abs(dmax)",
            "dx_unit": "diagonal_length=1",
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
        "dmax_min": float(dmax_min),
        "dmax_max": float(dmax_max),
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
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
            "Generate one split of motion-blur paired dataset from promptir_clear_depth_sets_only.json. "
            "Per-image dx is independently sampled by dmax/theta ranges."
        )
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--count", type=int, required=True, help="Number of samples to generate for this split.")
    parser.add_argument("--splits_json", type=str, default=str(DEFAULT_SPLITS_JSON), help="Path to split json.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for source-path sampling.")

    parser.add_argument("--dmax_min", type=float, default=-0.02, help="Per-image dmax lower bound.")
    parser.add_argument("--dmax_max", type=float, default=-0.0, help="Per-image dmax upper bound.")
    parser.add_argument("--theta_min", type=float, default=0.0, help="Per-image theta lower bound (radians).")
    parser.add_argument("--theta_max", type=float, default=6.283185307179586, help="Per-image theta upper bound (radians).")

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

    summary = generate_motion_blur_split_dataset(
        split=str(args.split),
        count=int(args.count),
        splits_json=Path(args.splits_json),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        dmax_min=float(args.dmax_min),
        dmax_max=float(args.dmax_max),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
        num_steps=int(args.num_steps),
        batchify_steps=bool(args.batchify_steps),
        mode=str(args.mode),
        padding_mode=str(args.padding_mode),
        align_corners=bool(args.align_corners),
        device_name=str(args.device),
        progress_interval=int(args.progress_interval),
        index_offset=int(args.index_offset),
        artifact_suffix=str(args.artifact_suffix),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

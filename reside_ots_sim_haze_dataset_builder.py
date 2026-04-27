from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

try:
    from util.image_io import load_rgb_tensor as _load_rgb_tensor
    from util.image_io import save_rgb_tensor as _save_rgb_tensor
    from util.runtime import resolve_runtime_device as _resolve_runtime_device
    from util.runtime import seed_everything as _seed_everything
except ModuleNotFoundError:
    from .util.image_io import load_rgb_tensor as _load_rgb_tensor
    from .util.image_io import save_rgb_tensor as _save_rgb_tensor
    from .util.runtime import resolve_runtime_device as _resolve_runtime_device
    from .util.runtime import seed_everything as _seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_JSON = PROJECT_ROOT / "dataset_path" / "promptir_clear_depth_sets_only.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "haze" / "reside_ots_sim"


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

    split_obj = obj["splits"][split]
    clear_paths = split_obj.get("clear_paths", [])
    resolved: list[Path] = []
    for p in clear_paths:
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


def _sample_exp_uniform(value_min: float, value_max: float, rng: random.Random, name: str) -> float:
    lo = float(value_min)
    hi = float(value_max)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(f"{name} exp-uniform range must be > 0, got {(value_min, value_max)}")
    if lo > hi:
        raise ValueError(f"{name}_min must be <= {name}_max, got {(value_min, value_max)}")
    lo_log = math.log(lo)
    hi_log = math.log(hi)
    return float(math.exp(rng.uniform(lo_log, hi_log)))


def _apply_haze_with_scalar_params(
    clear_image: torch.Tensor,
    fog_density: float,
    global_light_intensity: float,
) -> tuple[torch.Tensor, float]:
    if clear_image.ndim != 4:
        raise ValueError(f"clear_image must be BCHW, got shape={tuple(clear_image.shape)}")

    transmission = torch.exp(
        torch.full((1, 1, 1, 1), fill_value=-float(fog_density), dtype=clear_image.dtype, device=clear_image.device)
    )
    airlight = torch.full_like(clear_image, fill_value=float(global_light_intensity))
    hazy_image = clear_image * transmission + airlight * (1.0 - transmission)
    return hazy_image.clamp(0.0, 1.0), float(transmission.item())


def generate_haze_split_dataset(
    split: str,
    count: int,
    splits_json: Path,
    output_root: Path,
    seed: int,
    global_light_min: float,
    global_light_max: float,
    fog_density_min: float,
    fog_density_max: float,
    device_name: str,
    progress_interval: int,
    index_offset: int = 0,
    artifact_suffix: str = "",
) -> dict[str, object]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split}")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
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

    records: list[dict[str, object]] = []
    for idx, source_clear in enumerate(selected_paths):
        i_gt = _load_rgb_tensor(source_clear).to(device=device, dtype=torch.float32)
        global_light = _sample_exp_uniform(global_light_min, global_light_max, rng=rng, name="global_light")
        fog_density = _sample_exp_uniform(fog_density_min, fog_density_max, rng=rng, name="fog_density")

        with torch.no_grad():
            i_deg, transmission_scalar = _apply_haze_with_scalar_params(
                clear_image=i_gt,
                fog_density=float(fog_density),
                global_light_intensity=float(global_light),
            )

        image_id = f"{int(index_offset) + idx:06d}"
        input_path = input_dir / f"{image_id}.png"
        target_path = target_dir / f"{image_id}.png"
        _save_rgb_tensor(i_deg.detach().cpu(), input_path)
        _save_rgb_tensor(i_gt.detach().cpu(), target_path)

        _, _, h, w = i_gt.shape
        records.append(
            {
                "id": image_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "source_clear_path": str(source_clear),
                "source_hw": [int(h), int(w)],
                "global_light_intensity": float(global_light),
                "fog_density": float(fog_density),
                "transmission_scalar": float(transmission_scalar),
                "sampling_mode": "exp_uniform",
            }
        )

        if (idx + 1) % int(progress_interval) == 0 or (idx + 1) == int(count):
            print(f"[reside_ots_sim_haze_dataset_builder] split={split} generated {idx + 1}/{count}")

    manifest = {
        "split": split,
        "count": int(count),
        "seed": int(seed),
        "splits_json": str(splits_json),
        "sampling": {
            "mode": "exp_uniform",
            "global_light_min": float(global_light_min),
            "global_light_max": float(global_light_max),
            "fog_density_min": float(fog_density_min),
            "fog_density_max": float(fog_density_max),
            "formula": "x = exp(uniform(log(min), log(max)))",
        },
        "haze_model": "I = J * exp(-beta) + A * (1 - exp(-beta)), scalar beta and scalar A per image",
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
        "global_light_min": float(global_light_min),
        "global_light_max": float(global_light_max),
        "fog_density_min": float(fog_density_min),
        "fog_density_max": float(fog_density_max),
        "sampling_mode": "exp_uniform",
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
            "Generate one split of RESIDE-OTS-like synthetic haze paired dataset from "
            "promptir_clear_depth_sets_only.json using scalar global-light and fog-density "
            "sampled by exp-uniform (log-uniform)."
        )
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--count", type=int, required=True, help="Number of samples to generate for this split.")
    parser.add_argument("--splits_json", type=str, default=str(DEFAULT_SPLITS_JSON), help="Path to split json.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for source-path and parameter sampling.")
    parser.add_argument("--global_light_min", type=float, default=0.5, help="Global light lower bound (A_min).")
    parser.add_argument("--global_light_max", type=float, default=1.0, help="Global light upper bound (A_max).")
    parser.add_argument("--fog_density_min", type=float, default=0.1, help="Fog density lower bound (beta_min).")
    parser.add_argument("--fog_density_max", type=float, default=1.0, help="Fog density upper bound (beta_max).")
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/cuda:N/auto")
    parser.add_argument("--progress_interval", type=int, default=256, help="Progress print interval.")
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
    summary = generate_haze_split_dataset(
        split=str(args.split),
        count=int(args.count),
        splits_json=Path(args.splits_json),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        global_light_min=float(args.global_light_min),
        global_light_max=float(args.global_light_max),
        fog_density_min=float(args.fog_density_min),
        fog_density_max=float(args.fog_density_max),
        device_name=str(args.device),
        progress_interval=int(args.progress_interval),
        index_offset=int(args.index_offset),
        artifact_suffix=str(args.artifact_suffix),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

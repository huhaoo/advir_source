from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch import Tensor

from promptir_attack import (
    _forward_promptir_with_padding,
    build_motion_blur_controller_from_image,
    run_single_image_adversarial_degradation_search,
)
from promptir_paired_dataset_test import load_promptir_model

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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "exp" / "motion_blur" / "train_ckpt_nafnet" / "epoch=128-step=87297.ckpt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset_ours" / "motion_blur_adv16_nafnet_epoch128_step87297"


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


def _build_model_args_namespace() -> SimpleNamespace:
    # Match paired-eval defaults used in this repository for NAFNet checkpoints.
    return SimpleNamespace(
        model_arch="nafnet",
        naf_width=32,
        naf_middle_blk_num=1,
        naf_enc_blk_nums=[1, 1, 1, 28],
        naf_dec_blk_nums=[1, 1, 1, 1],
        naf_dw_expand=2,
        naf_ffn_expand=2,
        naf_dropout=0.0,
    )


def _save_strength_map(strength: Tensor, path: Path, title: str = "|dx|") -> dict[str, float]:
    if strength.ndim != 2:
        raise ValueError(f"strength map must be 2D, got shape={tuple(strength.shape)}")

    arr = strength.detach().cpu().float().numpy()
    v_min = float(arr.min())
    v_max = float(arr.max())
    v_mean = float(arr.mean())
    if v_max > v_min:
        norm = (arr - v_min) / (v_max - v_min)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((norm * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{title} min={v_min:.4f} max={v_max:.4f} mean={v_mean:.4f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 3
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    img.save(path)

    return {
        "min": v_min,
        "max": v_max,
        "mean": v_mean,
    }


def _save_signed_map(value_map: Tensor, path: Path, title: str) -> dict[str, float]:
    if value_map.ndim != 2:
        raise ValueError(f"signed map must be 2D, got shape={tuple(value_map.shape)}")

    arr = value_map.detach().cpu().float().numpy()
    v_min = float(arr.min())
    v_max = float(arr.max())
    v_mean = float(arr.mean())
    scale = max(abs(v_min), abs(v_max), 1e-12)
    # Map [-scale, scale] -> [0, 1], so zero is mid-gray.
    norm = (arr / (2.0 * scale)) + 0.5
    norm = np.clip(norm, 0.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((norm * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{title} min={v_min:.4f} max={v_max:.4f} mean={v_mean:.4f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 3
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    img.save(path)

    return {
        "min": v_min,
        "max": v_max,
        "mean": v_mean,
        "scale_abs_max": float(scale),
    }


def generate_motion_blur_adv_samples(
    model_path: Path,
    split: str,
    count: int,
    splits_json: Path,
    output_root: Path,
    seed: int,
    device_name: str,
    steps1: int,
    steps2: int,
    step_size: float,
    lambda_reg: float,
    promptir_patch_size: int,
    promptir_patch_overlap: int,
    motion_num_steps: int,
    motion_dmax: float | None,
    motion_dlambda: float,
    motion_interp_mode: str,
    motion_low_res_height: int | None,
    motion_low_res_width: int | None,
    map_lambda_first_order: float,
    map_lambda_second_order: float,
    save_npy: bool,
    progress_interval: int,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split}")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if progress_interval <= 0:
        raise ValueError(f"progress_interval must be > 0, got {progress_interval}")

    _seed_everything(int(seed))
    rng = random.Random(int(seed) + {"train": 0, "val": 1, "test": 2}[split])

    device = _resolve_runtime_device(device_name)
    model = load_promptir_model(
        checkpoint_path=model_path,
        device=device,
        args=_build_model_args_namespace(),
    )

    clear_paths = _load_split_clear_paths(splits_json=splits_json, split=split)
    selected_paths = _select_source_paths(clear_paths=clear_paths, count=int(count), rng=rng)

    input_dir = output_root / "input"
    target_dir = output_root / "target"
    restored_dir = output_root / "restored"
    motion_strength_dir = output_root / "motion_strength"
    motion_strength_npy_dir = output_root / "motion_strength_npy"
    dx_npy_dir = output_root / "dx_npy"
    low_res_dx_map_npy_dir = output_root / "low_res_dx_map_npy"
    low_res_dy_map_npy_dir = output_root / "low_res_dy_map_npy"
    low_res_dx_map_png_dir = output_root / "low_res_dx_map_png"
    low_res_dy_map_png_dir = output_root / "low_res_dy_map_png"
    attack_log_dir = output_root / "attack_log"

    output_dirs = [
        input_dir,
        target_dir,
        restored_dir,
        motion_strength_dir,
        low_res_dx_map_png_dir,
        low_res_dy_map_png_dir,
        attack_log_dir,
    ]
    if save_npy:
        output_dirs.extend(
            [
                motion_strength_npy_dir,
                dx_npy_dir,
                low_res_dx_map_npy_dir,
                low_res_dy_map_npy_dir,
            ]
        )

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for idx, source_clear in enumerate(selected_paths):
        image_id = f"{idx:06d}"
        clean = _load_rgb_tensor(source_clear).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

        controller = build_motion_blur_controller_from_image(
            image=clean,
            num_steps=int(motion_num_steps),
            dmax=motion_dmax,
            dlambda=float(motion_dlambda),
            interp_mode=str(motion_interp_mode),
            map_low_res_height=motion_low_res_height,
            map_low_res_width=motion_low_res_width,
            map_lambda_first_order=float(map_lambda_first_order),
            map_lambda_second_order=float(map_lambda_second_order),
        ).to(device=device)

        result = run_single_image_adversarial_degradation_search(
            image=clean,
            target=clean,
            promptir_model=model,
            degradation_controller=controller,
            distance_map=None,
            steps1=int(steps1),
            steps2=int(steps2),
            step_size=float(step_size),
            lambda_reg=float(lambda_reg),
            save_dir=None,
            record_history=True,
            allow_promptir_trainable_params=False,
            promptir_patch_size=int(promptir_patch_size),
            promptir_patch_overlap=int(promptir_patch_overlap),
        )

        with torch.no_grad():
            adv = controller(clean).detach().cpu()
            restored = _forward_promptir_with_padding(model, controller(clean)).detach().cpu().clamp(0.0, 1.0)
            dx_map = controller.get_dx_map().detach().cpu()[0]  # [2,H,W], diagonal-normalized unit
            low_res_dx_map = controller.dx_map_module.get_low_res_map().detach().cpu()[0, 0]  # [h_low,w_low]
            low_res_dy_map = controller.dy_map_module.get_low_res_map().detach().cpu()[0, 0]  # [h_low,w_low]

        strength = torch.sqrt(dx_map[0] ** 2 + dx_map[1] ** 2)
        strength_stats = _save_strength_map(strength=strength, path=motion_strength_dir / f"{image_id}.png")

        _save_rgb_tensor(adv, input_dir / f"{image_id}.png")
        _save_rgb_tensor(clean.detach().cpu(), target_dir / f"{image_id}.png")
        _save_rgb_tensor(restored, restored_dir / f"{image_id}.png")

        if save_npy:
            np.save(motion_strength_npy_dir / f"{image_id}.npy", strength.numpy().astype(np.float32))
            np.save(dx_npy_dir / f"{image_id}.npy", dx_map.numpy().astype(np.float32))
            np.save(low_res_dx_map_npy_dir / f"{image_id}.npy", low_res_dx_map.numpy().astype(np.float32))
            np.save(low_res_dy_map_npy_dir / f"{image_id}.npy", low_res_dy_map.numpy().astype(np.float32))
        low_res_dx_stats = _save_signed_map(
            value_map=low_res_dx_map,
            path=low_res_dx_map_png_dir / f"{image_id}.png",
            title="low_res_dx",
        )
        low_res_dy_stats = _save_signed_map(
            value_map=low_res_dy_map,
            path=low_res_dy_map_png_dir / f"{image_id}.png",
            title="low_res_dy",
        )

        attack_log = {
            "image_id": image_id,
            "initial_task_loss": float(result["initial_task_loss"]),
            "initial_reg_loss": float(result["initial_reg_loss"]),
            "initial_attack_obj": float(result["initial_attack_obj"]),
            "final_task_loss": float(result["final_task_loss"]),
            "final_reg_loss": float(result["final_reg_loss"]),
            "best_attack_obj": float(result["best_attack_obj"]),
            "best_task_loss": float(result["best_task_loss"]),
            "best_attack_step": result["best_attack_step"],
            "best_task_step": result["best_task_step"],
            "promptir_calls": int(result["promptir_calls"]),
            "history": result.get("history", []),
        }
        (attack_log_dir / f"{image_id}.json").write_text(json.dumps(attack_log, indent=2), encoding="utf-8")

        h = int(clean.shape[-2])
        w = int(clean.shape[-1])
        diag = math.sqrt(float(h * h + w * w))
        record = {
            "id": image_id,
            "source_clear_path": str(source_clear),
            "source_hw": [h, w],
            "source_diagonal": float(diag),
            "input_path": str(input_dir / f"{image_id}.png"),
            "target_path": str(target_dir / f"{image_id}.png"),
            "restored_path": str(restored_dir / f"{image_id}.png"),
            "motion_strength_png_path": str(motion_strength_dir / f"{image_id}.png"),
            "low_res_dx_map_png_path": str(low_res_dx_map_png_dir / f"{image_id}.png"),
            "low_res_dy_map_png_path": str(low_res_dy_map_png_dir / f"{image_id}.png"),
            "attack_log_path": str(attack_log_dir / f"{image_id}.json"),
            "motion_strength_min": float(strength_stats["min"]),
            "motion_strength_max": float(strength_stats["max"]),
            "motion_strength_mean": float(strength_stats["mean"]),
            "low_res_dx_min": float(low_res_dx_stats["min"]),
            "low_res_dx_max": float(low_res_dx_stats["max"]),
            "low_res_dx_mean": float(low_res_dx_stats["mean"]),
            "low_res_dy_min": float(low_res_dy_stats["min"]),
            "low_res_dy_max": float(low_res_dy_stats["max"]),
            "low_res_dy_mean": float(low_res_dy_stats["mean"]),
            "attack_final_task_loss": float(result["final_task_loss"]),
            "attack_final_reg_loss": float(result["final_reg_loss"]),
        }
        if save_npy:
            record["motion_strength_npy_path"] = str(motion_strength_npy_dir / f"{image_id}.npy")
            record["dx_npy_path"] = str(dx_npy_dir / f"{image_id}.npy")
            record["low_res_dx_map_npy_path"] = str(low_res_dx_map_npy_dir / f"{image_id}.npy")
            record["low_res_dy_map_npy_path"] = str(low_res_dy_map_npy_dir / f"{image_id}.npy")
        records.append(record)
        if (idx + 1) % int(progress_interval) == 0 or (idx + 1) == int(count):
            print(f"[generate_motion_blur_adv_samples] generated {idx + 1}/{count}")

    summary = {
        "model_path": str(model_path),
        "split": split,
        "count": int(count),
        "output_root": str(output_root),
        "seed": int(seed),
        "device": str(device),
        "steps1": int(steps1),
        "steps2": int(steps2),
        "step_size": float(step_size),
        "lambda_reg": float(lambda_reg),
        "promptir_patch_size": int(promptir_patch_size),
        "promptir_patch_overlap": int(promptir_patch_overlap),
        "motion_blur": {
            "num_steps": int(motion_num_steps),
            "dmax": None if motion_dmax is None else float(motion_dmax),
            "dlambda": float(motion_dlambda),
            "interp_mode": str(motion_interp_mode),
            "map_low_res_height": None if motion_low_res_height is None else int(motion_low_res_height),
            "map_low_res_width": None if motion_low_res_width is None else int(motion_low_res_width),
            "map_lambda_first_order": float(map_lambda_first_order),
            "map_lambda_second_order": float(map_lambda_second_order),
            "dx_unit": "diagonal_length=1",
            "strength_field": "|dx| from final optimized controller (diagonal-normalized unit)",
            "save_npy": bool(save_npy),
            "low_res_control_maps": {
                "dx_npy_dir": str(low_res_dx_map_npy_dir) if save_npy else None,
                "dy_npy_dir": str(low_res_dy_map_npy_dir) if save_npy else None,
                "dx_png_dir": str(low_res_dx_map_png_dir),
                "dy_png_dir": str(low_res_dy_map_png_dir),
                "unit": "diagonal_length=1",
            },
        },
        "source_pool_size": len(clear_paths),
        "sample_with_replacement": bool(count > len(clear_paths)),
    }

    manifest = {
        "summary": summary,
        "records": records,
    }

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adversarial motion-blur samples against a PromptIR/NAFNet checkpoint, "
            "and keep corresponding motion strength fields."
        )
    )
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--splits_json", type=str, default=str(DEFAULT_SPLITS_JSON))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu/cuda/cuda:N/auto")

    parser.add_argument("--steps1", type=int, default=2)
    parser.add_argument("--steps2", type=int, default=2)
    parser.add_argument("--step_size", type=float, default=3e-2)
    parser.add_argument("--lambda_reg", type=float, default=2.0)
    parser.add_argument("--promptir_patch_size", type=int, default=128)
    parser.add_argument("--promptir_patch_overlap", type=int, default=32)

    parser.add_argument("--motion_num_steps", type=int, default=16)
    parser.add_argument("--motion_dmax", type=float, default=-0.02)
    parser.add_argument("--motion_dlambda", type=float, default=0.0)
    parser.add_argument(
        "--motion_interp_mode",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "area", "gaussian"],
    )
    parser.add_argument("--motion_low_res_height", type=int, default=None)
    parser.add_argument("--motion_low_res_width", type=int, default=None)
    parser.add_argument("--map_lambda_first_order", type=float, default=0.1)
    parser.add_argument("--map_lambda_second_order", type=float, default=0.5)
    parser.add_argument(
        "--save_npy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save motion_strength/dx/lowres maps as .npy files.",
    )

    parser.add_argument("--progress_interval", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motion_dmax_value: float | None = None if args.motion_dmax is None else float(args.motion_dmax)
    summary = generate_motion_blur_adv_samples(
        model_path=Path(args.model_path),
        split=str(args.split),
        count=int(args.count),
        splits_json=Path(args.splits_json),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        device_name=str(args.device),
        steps1=int(args.steps1),
        steps2=int(args.steps2),
        step_size=float(args.step_size),
        lambda_reg=float(args.lambda_reg),
        promptir_patch_size=int(args.promptir_patch_size),
        promptir_patch_overlap=int(args.promptir_patch_overlap),
        motion_num_steps=int(args.motion_num_steps),
        motion_dmax=motion_dmax_value,
        motion_dlambda=float(args.motion_dlambda),
        motion_interp_mode=str(args.motion_interp_mode),
        motion_low_res_height=None if args.motion_low_res_height is None else int(args.motion_low_res_height),
        motion_low_res_width=None if args.motion_low_res_width is None else int(args.motion_low_res_width),
        map_lambda_first_order=float(args.map_lambda_first_order),
        map_lambda_second_order=float(args.map_lambda_second_order),
        save_npy=bool(args.save_npy),
        progress_interval=int(args.progress_interval),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"

if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from net.model import build_promptir_model  # noqa: E402
from utils.pytorch_ssim import ssim as pytorch_ssim  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PromptIR checkpoint on a paired input/target dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/huhao/adv_ir/PromptIR/train_ckpt_dehaze/epoch=127-step=87424.ckpt",
        help="PromptIR Lightning checkpoint path (expects net.* keys).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/huhao/adv_ir/dataset_ours/adv_haze_100_beta04",
        help="Dataset root containing input/target paired subfolders.",
    )
    parser.add_argument(
        "--input_subdir",
        type=str,
        default="input",
        help="Subdirectory under dataset_root for degraded inputs.",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        default="target",
        help="Subdirectory under dataset_root for clean targets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/huhao/adv_ir/tmp_demo/promptir_paired_dataset_test",
        help="Output directory for metrics and optional restored images.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=">0 to only evaluate first N matched pairs.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="0 means full-image inference, >0 enables tiled inference.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=64,
        help="Tile overlap when tile inference is enabled.",
    )
    parser.add_argument(
        "--save_restored",
        action="store_true",
        help="Save restored images under output_dir/restored.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input/target directories.",
    )
    parser.add_argument("--model_arch", type=str, default="promptir", choices=["nafnet", "promptir"])
    parser.add_argument("--naf_width", type=int, default=32)
    parser.add_argument("--naf_middle_blk_num", type=int, default=1)
    parser.add_argument("--naf_enc_blk_nums", type=int, nargs="+", default=[1, 1, 1, 28])
    parser.add_argument("--naf_dec_blk_nums", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--naf_dw_expand", type=int, default=2)
    parser.add_argument("--naf_ffn_expand", type=int, default=2)
    parser.add_argument("--naf_dropout", type=float, default=0.0)
    return parser.parse_args()


def _valid_image_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _scan_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [path for path in iterator if path.is_file() and _valid_image_suffix(path)]
    return sorted(files)


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()


def _pad_to_multiple_of(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    pad_mode = "reflect"
    if h <= 1 or w <= 1:
        pad_mode = "replicate"

    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x_pad, (pad_h, pad_w)


def _restore_full(model: torch.nn.Module, haze: torch.Tensor) -> torch.Tensor:
    _, _, h, w = haze.shape
    haze_pad, _ = _pad_to_multiple_of(haze, multiple=8)
    restored_pad = model(haze_pad)
    return restored_pad[:, :, :h, :w]


def _build_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]

    positions: list[int] = []
    start = 0
    while start + tile_size < length:
        positions.append(start)
        start += step

    last_start = length - tile_size
    if len(positions) == 0 or positions[-1] != last_start:
        positions.append(last_start)
    return positions


def _restore_with_tiling(model: torch.nn.Module, haze: torch.Tensor, tile_size: int, tile_overlap: int) -> torch.Tensor:
    if tile_size <= 0:
        return _restore_full(model, haze)

    _, _, h, w = haze.shape
    if tile_size >= h and tile_size >= w:
        return _restore_full(model, haze)

    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError(
            f"invalid tile settings: tile_size={tile_size}, tile_overlap={tile_overlap}, step={step}"
        )

    y_positions = _build_positions(h, tile_size, step)
    x_positions = _build_positions(w, tile_size, step)

    output = torch.zeros_like(haze)
    weight = torch.zeros_like(haze)

    for y0 in y_positions:
        for x0 in x_positions:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = haze[:, :, y0:y1, x0:x1]
            restored_tile = _restore_full(model, tile)
            output[:, :, y0:y1, x0:x1] += restored_tile
            weight[:, :, y0:y1, x0:x1] += 1.0

    return output / torch.clamp(weight, min=1e-6)


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def _distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_promptir_model(checkpoint_path: Path, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    try:
        checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except (TypeError, ValueError, RuntimeError, pickle.UnpicklingError):
        checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")

    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a valid state_dict")

    promptir_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            promptir_state[key[4:]] = value

    if len(promptir_state) == 0:
        raise ValueError("no 'net.' prefixed keys found in checkpoint state_dict")

    model = build_promptir_model(
        model_arch=args.model_arch,
        decoder=True,
        inp_channels=3,
        out_channels=3,
        naf_width=int(args.naf_width),
        naf_middle_blk_num=int(args.naf_middle_blk_num),
        naf_enc_blk_nums=list(args.naf_enc_blk_nums),
        naf_dec_blk_nums=list(args.naf_dec_blk_nums),
        naf_dw_expand=int(args.naf_dw_expand),
        naf_ffn_expand=int(args.naf_ffn_expand),
        naf_dropout=float(args.naf_dropout),
    )
    missing_keys, unexpected_keys = model.load_state_dict(promptir_state, strict=False)
    if len(unexpected_keys) > 0:
        raise RuntimeError(f"unexpected model keys while loading checkpoint: {unexpected_keys[:10]}")
    if len(missing_keys) > 0:
        raise RuntimeError(f"missing model keys while loading checkpoint: {missing_keys[:10]}")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def collect_pairs(
    input_dir: Path,
    target_dir: Path,
    recursive: bool,
) -> tuple[list[tuple[Path, Path, Path]], list[dict[str, str]]]:
    input_files = _scan_images(root=input_dir, recursive=recursive)
    pairs: list[tuple[Path, Path, Path]] = []
    skipped: list[dict[str, str]] = []

    for input_path in input_files:
        rel_path = input_path.relative_to(input_dir)
        target_path_exact = target_dir / rel_path
        if target_path_exact.exists() and target_path_exact.is_file():
            pairs.append((rel_path, input_path, target_path_exact))
            continue

        candidates = [
            p
            for p in (target_dir / rel_path.parent).glob(f"{input_path.stem}.*")
            if p.is_file() and _valid_image_suffix(p)
        ]
        if len(candidates) == 1:
            pairs.append((rel_path, input_path, candidates[0]))
            continue

        skipped.append(
            {
                "input_path": str(input_path),
                "reason": "target_not_found_or_ambiguous",
            }
        )

    pairs = sorted(pairs, key=lambda x: str(x[0]))
    return pairs, skipped


def save_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "model_path",
                "dataset_root",
                "input_path",
                "target_path",
                "height",
                "width",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    dataset_root = Path(args.dataset_root)
    input_dir = dataset_root / args.input_subdir
    target_dir = dataset_root / args.target_subdir
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir not found: {target_dir}")
    if args.max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {args.max_samples}")
    if args.tile_size < 0:
        raise ValueError(f"tile_size must be >= 0, got {args.tile_size}")
    if args.tile_overlap < 0:
        raise ValueError(f"tile_overlap must be >= 0, got {args.tile_overlap}")

    pairs, skipped = collect_pairs(input_dir=input_dir, target_dir=target_dir, recursive=bool(args.recursive))
    if len(pairs) == 0:
        raise RuntimeError(
            f"no matched pairs found in input/target: input_dir={input_dir}, target_dir={target_dir}"
        )

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Warn] CUDA requested but unavailable, fallback to CPU")

    output_dir.mkdir(parents=True, exist_ok=True)
    restored_dir = output_dir / "restored"
    if args.save_restored:
        restored_dir.mkdir(parents=True, exist_ok=True)

    model = load_promptir_model(checkpoint_path=model_path, device=device, args=args)
    model_path_text = str(model_path.resolve())

    print(f"[Model] {model_path_text}")
    print(f"[Dataset] {dataset_root}")
    print(f"[Input] {input_dir}")
    print(f"[Target] {target_dir}")
    print(f"[Pairs] matched={len(pairs)}, skipped={len(skipped)}")

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for rel_path, input_path, target_path in tqdm(pairs, desc="PromptIR paired-dataset eval"):
            degraded = _load_rgb_tensor(input_path).to(device)
            target = _load_rgb_tensor(target_path).to(device)

            if degraded.shape != target.shape:
                skipped.append(
                    {
                        "input_path": str(input_path),
                        "reason": f"shape_mismatch_{tuple(degraded.shape)}_{tuple(target.shape)}",
                    }
                )
                continue

            restored = _restore_with_tiling(
                model=model,
                haze=degraded,
                tile_size=int(args.tile_size),
                tile_overlap=int(args.tile_overlap),
            )
            restored = torch.clamp(restored, 0.0, 1.0)

            mse = torch.mean((restored - target) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            ssim_val = float(
                pytorch_ssim(
                    torch.clamp(restored, 0.0, 1.0),
                    torch.clamp(target, 0.0, 1.0),
                    size_average=True,
                ).item()
            )

            records.append(
                {
                    "sample_id": str(rel_path.with_suffix("")).replace("\\\\", "/"),
                    "model_path": model_path_text,
                    "dataset_root": str(dataset_root),
                    "input_path": str(input_path),
                    "target_path": str(target_path),
                    "height": int(degraded.shape[-2]),
                    "width": int(degraded.shape[-1]),
                    "psnr": float(psnr),
                    "ssim": float(ssim_val),
                }
            )

            if args.save_restored:
                save_path = restored_dir / rel_path.with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_tensor_to_rgb_uint8(restored), mode="RGB").save(save_path)

    if len(records) == 0:
        raise RuntimeError("all matched pairs were skipped during evaluation")

    psnr_values = [float(x["psnr"]) for x in records]
    ssim_values = [float(x["ssim"]) for x in records]

    csv_path = output_dir / "metrics_per_sample.csv"
    summary_path = output_dir / "metrics_summary.json"
    skipped_path = output_dir / "skipped_samples.json"

    save_csv(records=records, csv_path=csv_path)
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    summary = {
        "model_path": model_path_text,
        "dataset_root": str(dataset_root.resolve()),
        "input_dir": str(input_dir.resolve()),
        "target_dir": str(target_dir.resolve()),
        "num_samples": len(records),
        "num_skipped": len(skipped),
        "device": str(device),
        "tile_size": int(args.tile_size),
        "tile_overlap": int(args.tile_overlap),
        "psnr": _distribution(psnr_values),
        "ssim": _distribution(ssim_values),
        "metrics_csv": str(csv_path),
        "skipped_samples": str(skipped_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"[Done] Evaluated {len(records)} pairs")
    print(f"[PSNR] mean={summary['psnr']['mean']:.4f}, std={summary['psnr']['std']:.4f}")
    print(f"[SSIM] mean={summary['ssim']['mean']:.4f}, std={summary['ssim']['std']:.4f}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {skipped_path}")
    if args.save_restored:
        print(f"[Saved] {restored_dir}")


if __name__ == "__main__":
    main()

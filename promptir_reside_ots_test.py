from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"

if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from net.model import build_promptir_model  # noqa: E402
from utils.pytorch_ssim import ssim as pytorch_ssim  # noqa: E402


SCOPE_TO_MANIFEST = {
    "train": "hazy_outside_train.txt",
    "val": "hazy_outside_val.txt",
    "test": "hazy_outside_test.txt",
    "all": "hazy_outside.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PromptIR on RESIDE-OTS split manifest and report PSNR/SSIM.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/huhao/adv_ir/PromptIR/train_ckpt_adv_haze50/epoch=127-step=524288.ckpt",
        help="PromptIR lightning checkpoint path.",
    )
    parser.add_argument(
        "--manifest_root",
        type=str,
        default="/home/huhao/adv_ir/PromptIR/data_dir/hazy",
        help="Directory containing hazy_outside_{train,val,test}.txt manifests.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/huhao/adv_ir/dataset",
        help="Dataset root for resolving relative haze paths in manifest.",
    )
    parser.add_argument(
        "--test_scope",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which RESIDE-OTS split manifest to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/huhao/adv_ir/tmp_demo/promptir_reside_ots_test",
        help="Output directory for summary/csv and optional restored images.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=">0 for smoke test on first N samples.",
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
        help="Tile overlap in pixels when tile inference is enabled.",
    )
    parser.add_argument(
        "--save_restored",
        action="store_true",
        help="Save restored images to output_dir/restored.",
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


def _resolve_manifest_path(manifest_root: Path, test_scope: str) -> Path:
    return manifest_root / SCOPE_TO_MANIFEST[test_scope]


def _resolve_clear_path(haze_path_abs: Path) -> Path | None:
    haze_key = "/haze/reside_ots/haze/"
    haze_str = str(haze_path_abs)
    if haze_key not in haze_str:
        return None

    clear_str = haze_str.replace(haze_key, "/haze/reside_ots/clear/")
    clear_path = Path(clear_str)
    stem = clear_path.stem.split("_")[0]
    candidate = clear_path.with_name(stem + clear_path.suffix)
    if candidate.exists():
        return candidate
    return None


def collect_pairs(manifest_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    records: list[dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel == "":
                continue

            haze_path = (dataset_root / rel).resolve()
            if not haze_path.exists():
                continue
            clear_path = _resolve_clear_path(haze_path)
            if clear_path is None or not clear_path.exists():
                continue

            sample_id = haze_path.stem
            records.append(
                {
                    "sample_id": sample_id,
                    "haze_path": str(haze_path),
                    "clear_path": str(clear_path),
                }
            )
    return records


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous()


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


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


def _restore_with_tiling(
    model: torch.nn.Module,
    haze: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
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


def compute_psnr_ssim(restored: torch.Tensor, clean: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((restored - clean) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
    ssim_val = pytorch_ssim(
        torch.clamp(restored, 0.0, 1.0),
        torch.clamp(clean, 0.0, 1.0),
        size_average=True,
    )
    return float(psnr.mean().item()), float(ssim_val.item())


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


def save_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "scope",
                "model_path",
                "haze_path",
                "clear_path",
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
    manifest_root = Path(args.manifest_root)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not manifest_root.exists():
        raise FileNotFoundError(f"manifest_root not found: {manifest_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if args.max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {args.max_samples}")
    if args.tile_size < 0:
        raise ValueError(f"tile_size must be >= 0, got {args.tile_size}")
    if args.tile_overlap < 0:
        raise ValueError(f"tile_overlap must be >= 0, got {args.tile_overlap}")

    manifest_path = _resolve_manifest_path(manifest_root=manifest_root, test_scope=args.test_scope)
    pairs = collect_pairs(manifest_path=manifest_path, dataset_root=dataset_root)
    if len(pairs) == 0:
        raise RuntimeError(f"no valid pairs resolved from manifest: {manifest_path}")

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
    print(f"[Scope] {args.test_scope}")
    print(f"[Manifest] {manifest_path}")
    print(f"[Pairs] {len(pairs)}")

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in tqdm(pairs, desc=f"PromptIR RESIDE-OTS {args.test_scope}"):
            haze_path = Path(item["haze_path"])
            clear_path = Path(item["clear_path"])
            sample_id = item["sample_id"]

            haze = _load_rgb_tensor(haze_path).to(device)
            clean = _load_rgb_tensor(clear_path).to(device)
            restored = _restore_with_tiling(
                model=model,
                haze=haze,
                tile_size=int(args.tile_size),
                tile_overlap=int(args.tile_overlap),
            )
            restored = torch.clamp(restored, 0.0, 1.0)

            psnr, ssim_val = compute_psnr_ssim(restored=restored, clean=clean)
            h = int(haze.shape[-2])
            w = int(haze.shape[-1])

            records.append(
                {
                    "sample_id": sample_id,
                    "scope": args.test_scope,
                    "model_path": model_path_text,
                    "haze_path": str(haze_path),
                    "clear_path": str(clear_path),
                    "height": h,
                    "width": w,
                    "psnr": psnr,
                    "ssim": ssim_val,
                }
            )

            if args.save_restored:
                save_path = restored_dir / f"{sample_id}.png"
                Image.fromarray(_tensor_to_rgb_uint8(restored), mode="RGB").save(save_path)

    psnr_values = [float(x["psnr"]) for x in records]
    ssim_values = [float(x["ssim"]) for x in records]
    summary = {
        "model_path": model_path_text,
        "manifest": str(manifest_path),
        "scope": args.test_scope,
        "num_samples": len(records),
        "device": str(device),
        "tile_size": int(args.tile_size),
        "tile_overlap": int(args.tile_overlap),
        "psnr": _distribution(psnr_values),
        "ssim": _distribution(ssim_values),
        "output_dir": str(output_dir),
    }

    csv_path = output_dir / "metrics_per_sample.csv"
    summary_path = output_dir / "metrics_summary.json"
    save_csv(records, csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"[Done] Evaluated {len(records)} samples")
    print(f"[PSNR] mean={summary['psnr']['mean']:.4f}, std={summary['psnr']['std']:.4f}")
    print(f"[SSIM] mean={summary['ssim']['mean']:.4f}, std={summary['ssim']['std']:.4f}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")
    if args.save_restored:
        print(f"[Saved] {restored_dir}")


if __name__ == "__main__":
    main()

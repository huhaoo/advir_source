from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from utils.image_utils import crop_img, random_augmentation  # noqa: E402

from promptir_attack import (  # noqa: E402
    _load_distance_from_mat,
    _load_image_rgb,
    build_haze_controller_from_image,
    run_single_image_adversarial_degradation_search,
)


class promptir_adv_mix_dataset(Dataset):
    def __init__(self, base_dataset: Dataset, args: Any):
        super().__init__()
        self.base_dataset = base_dataset
        self.args = args

        self.adv_ratio = float(getattr(args, "adv_ratio", 0.5))
        if not (0.0 < self.adv_ratio < 1.0):
            raise ValueError(f"adv_ratio must be in (0, 1), got {self.adv_ratio}")

        self.adv_resample_epochs = int(getattr(args, "adv_resample_epochs", 8))
        if self.adv_resample_epochs <= 0:
            raise ValueError(f"adv_resample_epochs must be > 0, got {self.adv_resample_epochs}")

        self.adv_steps1 = int(getattr(args, "adv_steps1", 2))
        self.adv_steps2 = int(getattr(args, "adv_steps2", 2))
        self.adv_step_size = float(getattr(args, "adv_step_size", 3e-2))
        self.adv_lambda_reg = float(getattr(args, "adv_lambda_reg", 0.05))
        self.adv_promptir_patch_size = int(getattr(args, "adv_promptir_patch_size", getattr(args, "patch_size", 128)))
        self.adv_promptir_patch_overlap = int(getattr(args, "adv_promptir_patch_overlap", max(0, self.adv_promptir_patch_size // 4)))
        if self.adv_promptir_patch_size <= 0:
            raise ValueError(f"adv_promptir_patch_size must be > 0, got {self.adv_promptir_patch_size}")
        if self.adv_promptir_patch_overlap < 0 or self.adv_promptir_patch_overlap >= self.adv_promptir_patch_size:
            raise ValueError(
                "adv_promptir_patch_overlap must be in [0, adv_promptir_patch_size), "
                f"got overlap={self.adv_promptir_patch_overlap}, patch={self.adv_promptir_patch_size}"
            )

        self.adv_cache_root = Path(getattr(args, "adv_cache_root", PROMPTIR_ROOT / "data" / "Train" / "adv_pairs"))
        self.adv_samples_per_resample = int(getattr(args, "adv_samples_per_resample", 0))
        self.adv_aug_k = int(getattr(args, "adv_aug_k", 1))
        if self.adv_aug_k <= 0:
            raise ValueError(f"adv_aug_k must be >= 1, got {self.adv_aug_k}")

        self.base_len = len(self.base_dataset)
        self.adv_len = self._compute_adv_len()

        self.shared_cache_dir = self.adv_cache_root / "shared"
        self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
        self.adv_records: list[dict[str, Any]] = []
        self._last_resampled_epoch: int | None = None

        self.depth_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "depth"
        self.haze_source_indices = self._build_haze_source_indices()

    def _build_haze_source_indices(self) -> list[int]:
        indices: list[int] = []
        for i, sample in enumerate(self.base_dataset.sample_ids):
            try:
                de_id = int(sample.get("de_type", -1))
            except Exception:
                continue
            if de_id != 4:
                continue
            clean_path = self._resolve_dehaze_clean_path(str(sample.get("clean_id", "")))
            if clean_path is None:
                continue
            depth_path = self._resolve_depth_mat(clean_path)
            if depth_path is None:
                continue
            indices.append(i)
        return indices

    def _build_resample_pair_dirs(self, epoch: int) -> tuple[Path, Path, Path]:
        epoch_dir = self.shared_cache_dir / f"resample_epoch_{int(epoch):04d}"
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)
        input_dir = epoch_dir / "input"
        target_dir = epoch_dir / "target"
        shards_dir = epoch_dir / "shards"
        input_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)
        return epoch_dir, input_dir, target_dir

    def _epoch_dir(self, epoch: int) -> Path:
        return self.shared_cache_dir / f"resample_epoch_{int(epoch):04d}"

    def _manifest_path(self, epoch: int) -> Path:
        return self._epoch_dir(epoch) / "manifest.json"

    def _shards_dir(self, epoch: int) -> Path:
        return self._epoch_dir(epoch) / "shards"

    def _shard_manifest_path(self, epoch: int, rank: int) -> Path:
        return self._shards_dir(epoch) / f"manifest_rank_{int(rank):04d}.json"

    def _dist_rank_world(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def _local_target_and_offset(self, total: int, rank: int, world: int) -> tuple[int, int]:
        if total <= 0:
            return 0, 0
        if world <= 0:
            raise ValueError(f"world must be positive, got {world}")
        if rank < 0 or rank >= world:
            raise ValueError(f"rank must be in [0, world), got rank={rank}, world={world}")

        # Weighted split for DDP shards: rank0:others = 1:2:2:...
        weights = [1] + [2] * (world - 1)
        weight_sum = int(sum(weights))
        if weight_sum <= 0:
            raise RuntimeError("invalid shard weights: sum must be positive")

        targets = [int(total * w // weight_sum) for w in weights]
        remainder = int(total - sum(targets))
        for idx in range(remainder):
            targets[idx] += 1

        local_target = int(targets[rank])
        start_offset = int(sum(targets[:rank]))
        return local_target, start_offset

    def _write_manifest(self, epoch: int, records: list[dict[str, Any]]) -> None:
        manifest_path = self._manifest_path(epoch)
        obj = {"epoch": int(epoch), "count": len(records), "records": records}
        manifest_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def _write_shard_manifest(self, epoch: int, rank: int, records: list[dict[str, Any]]) -> None:
        shard_path = self._shard_manifest_path(epoch=epoch, rank=rank)
        obj = {"epoch": int(epoch), "rank": int(rank), "count": len(records), "records": records}
        shard_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def _load_shard_records(self, epoch: int, rank: int) -> list[dict[str, Any]]:
        shard_path = self._shard_manifest_path(epoch=epoch, rank=rank)
        if not shard_path.exists():
            return []
        obj = json.loads(shard_path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        if not isinstance(records, list):
            return []
        return [x for x in records if isinstance(x, dict)]

    def _merge_all_shards_to_manifest(self, epoch: int, world: int) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for rank in range(world):
            merged.extend(self._load_shard_records(epoch=epoch, rank=rank))

        def _pair_id_from_path(path_str: str) -> int:
            name = Path(path_str).stem
            try:
                return int(name)
            except ValueError:
                return 10**9

        merged.sort(key=lambda item: _pair_id_from_path(str(item.get("degraded_path", ""))))
        self._write_manifest(epoch=epoch, records=merged)
        return merged

    def _load_records_from_manifest(self, epoch: int) -> list[dict[str, Any]]:
        manifest_path = self._manifest_path(epoch)
        if not manifest_path.exists():
            return []

        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        if not isinstance(records, list):
            return []

        out: list[dict[str, Any]] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            degrad_path = item.get("degraded_path")
            clean_path = item.get("clean_path")
            if not isinstance(degrad_path, str) or not isinstance(clean_path, str):
                continue
            if not os.path.exists(degrad_path) or not os.path.exists(clean_path):
                continue
            out.append(
                {
                    "degraded_path": degrad_path,
                    "clean_path": clean_path,
                    "de_id": int(item.get("de_id", 4)),
                    "clean_name": str(item.get("clean_name", Path(clean_path).stem)),
                    "attack_subset": ["haze"],
                }
            )
        return out

    def _compute_adv_len(self) -> int:
        if self.base_len <= 0:
            return 0
        if self.adv_samples_per_resample > 0:
            return self.adv_samples_per_resample
        return max(1, int(round(self.base_len * self.adv_ratio)))

    def _resolve_dehaze_clean_path(self, hazy_path: str) -> str | None:
        if "/haze/reside_ots/haze/" not in hazy_path:
            return None

        clear_name = hazy_path.replace("/haze/reside_ots/haze/", "/haze/reside_ots/clear/")
        base_name = os.path.basename(clear_name)
        stem, ext = os.path.splitext(base_name)
        clean_stem = stem.split("_")[0]
        candidate = os.path.join(os.path.dirname(clear_name), clean_stem + ext)
        if os.path.exists(candidate):
            return candidate
        return None

    def _resolve_depth_mat(self, clean_path: str) -> Path | None:
        mat_path = self.depth_dir / f"{Path(clean_path).stem}.mat"
        if mat_path.exists():
            return mat_path
        return None

    def _tensor_to_uint8_hwc(self, image: torch.Tensor) -> np.ndarray:
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"expected image tensor with shape (C,H,W), got {tuple(image.shape)}")
        image = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return (image * 255.0 + 0.5).astype(np.uint8)

    def _save_rgb_np(self, image: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(path)

    def _save_rgb_tensor(self, image: torch.Tensor, path: Path) -> None:
        self._save_rgb_np(self._tensor_to_uint8_hwc(image), path)

    def _ensure_min_hw_tensor_pair(
        self,
        image: torch.Tensor,
        distance_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_size = int(self.base_dataset.args.patch_size)
        _, _, h, w = image.shape
        if h >= patch_size and w >= patch_size:
            return image, distance_map

        new_h = max(h, patch_size)
        new_w = max(w, patch_size)
        image = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        distance_map = torch.nn.functional.interpolate(distance_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return image, distance_map

    def _ensure_min_hw_numpy_pair(
        self,
        degrad_img: np.ndarray,
        clean_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        patch_size = int(self.base_dataset.args.patch_size)
        h, w = degrad_img.shape[:2]
        if h >= patch_size and w >= patch_size:
            return degrad_img, clean_img

        new_h = max(h, patch_size)
        new_w = max(w, patch_size)
        degrad_img = np.array(Image.fromarray(degrad_img).resize((new_w, new_h), Image.BICUBIC))
        clean_img = np.array(Image.fromarray(clean_img).resize((new_w, new_h), Image.BICUBIC))
        return degrad_img, clean_img

    def _expand_with_simple_augmentations(
        self,
        adv_np: np.ndarray,
        clean_np: np.ndarray,
        k: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if k <= 1:
            return [(adv_np, clean_np)]

        outputs: list[tuple[np.ndarray, np.ndarray]] = [(adv_np, clean_np)]
        for _ in range(k - 1):
            aug_adv, aug_clean = random_augmentation(adv_np, clean_np)
            outputs.append((aug_adv, aug_clean))
        return outputs

    def resample_main_process_then_sync(
        self,
        epoch: int,
        force: bool = False,
        promptir_model_override: torch.nn.Module | None = None,
    ) -> None:
        should_resample = force or (self._last_resampled_epoch is None) or (epoch % self.adv_resample_epochs == 0)
        if not should_resample:
            return

        rank, world = self._dist_rank_world()

        if world == 1:
            self.resample_adversarial(epoch=epoch, force=True, promptir_model_override=promptir_model_override)
            self.adv_records = self._load_records_from_manifest(epoch=epoch)
            self._last_resampled_epoch = epoch
            return

        if rank == 0:
            self._build_resample_pair_dirs(epoch=epoch)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        local_records = self._resample_local_shard(
            epoch=epoch,
            rank=rank,
            world=world,
            promptir_model_override=promptir_model_override,
        )
        self._write_shard_manifest(epoch=epoch, rank=rank, records=local_records)

        dist.barrier()
        if rank == 0:
            self._merge_all_shards_to_manifest(epoch=epoch, world=world)

        dist.barrier()

        self.adv_records = self._load_records_from_manifest(epoch=epoch)
        self._last_resampled_epoch = epoch

    def resample_adversarial(
        self,
        epoch: int,
        force: bool = False,
        promptir_model_override: torch.nn.Module | None = None,
    ) -> None:
        should_resample = force or (self._last_resampled_epoch is None) or (epoch % self.adv_resample_epochs == 0)
        if not should_resample:
            return

        if self.base_len <= 0 or self.adv_len <= 0:
            self.adv_records = []
            self._last_resampled_epoch = epoch
            return

        self._build_resample_pair_dirs(epoch=epoch)
        new_records = self._resample_local_shard(
            epoch=epoch,
            rank=0,
            world=1,
            promptir_model_override=promptir_model_override,
        )

        self.adv_records = new_records
        self._last_resampled_epoch = epoch
        self._write_manifest(epoch=epoch, records=new_records)
        print(
            f"[AdvMixHazeOnly] epoch={epoch} base={self.base_len} adv_target={self.adv_len} adv_ready={len(self.adv_records)} "
            f"k={self.adv_aug_k} resample_every={self.adv_resample_epochs} sample_root={self._epoch_dir(epoch)}"
        )

    def _resample_local_shard(
        self,
        epoch: int,
        rank: int,
        world: int,
        promptir_model_override: torch.nn.Module | None = None,
    ) -> list[dict[str, Any]]:
        epoch_dir = self._epoch_dir(epoch)
        input_dir = epoch_dir / "input"
        target_dir = epoch_dir / "target"

        local_target, pair_offset = self._local_target_and_offset(total=self.adv_len, rank=rank, world=world)
        if local_target <= 0:
            return []
        if promptir_model_override is None:
            raise ValueError("promptir_model_override is required for adversarial resampling")
        if len(self.haze_source_indices) == 0:
            print("[AdvMixHazeOnly] no valid dehaze+depth source sample found, skip resample")
            return []

        # Generate only ceil(local_target / k) adversarial samples, then duplicate by simple augmentation.
        base_adv_target = int(math.ceil(float(local_target) / float(self.adv_aug_k)))

        promptir_model = promptir_model_override
        try:
            device = next(promptir_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        promptir_params = list(promptir_model.parameters())
        original_requires_grad = [bool(param.requires_grad) for param in promptir_params]
        original_training = bool(promptir_model.training)
        for param in promptir_params:
            param.requires_grad = False
        promptir_model.eval()

        local_records: list[dict[str, Any]] = []
        produced_base_adv = 0
        attempts = 0
        max_attempts = max(base_adv_target * 4, 32)

        progress = tqdm(total=local_target, desc=f"adv-resample@epoch{epoch}:rank{rank}", leave=False)
        try:
            while len(local_records) < local_target and produced_base_adv < base_adv_target and attempts < max_attempts:
                attempts += 1

                src_idx = random.choice(self.haze_source_indices)
                sample = self.base_dataset.sample_ids[src_idx]

                clean_path = self._resolve_dehaze_clean_path(str(sample.get("clean_id", "")))
                if clean_path is None or not os.path.exists(clean_path):
                    continue

                depth_mat = self._resolve_depth_mat(clean_path)
                if depth_mat is None:
                    continue

                image = _load_image_rgb(Path(clean_path))
                _, _, h, w = image.shape
                distance_map, _ = _load_distance_from_mat(depth_mat, (h, w))

                image, distance_map = self._ensure_min_hw_tensor_pair(image, distance_map)
                target = image.clone()

                image = image.to(device)
                target = target.to(device)
                distance_map = distance_map.to(device)

                controller = build_haze_controller_from_image(image)
                result = run_single_image_adversarial_degradation_search(
                    image=image,
                    target=target,
                    promptir_model=promptir_model,
                    degradation_controller=controller,
                    distance_map=distance_map,
                    steps1=self.adv_steps1,
                    steps2=self.adv_steps2,
                    step_size=self.adv_step_size,
                    lambda_reg=self.adv_lambda_reg,
                    rain_topk=1,
                    save_dir=None,
                    record_history=False,
                    save_visual_maps=False,
                    allow_promptir_trainable_params=False,
                    promptir_patch_size=self.adv_promptir_patch_size,
                    promptir_patch_overlap=self.adv_promptir_patch_overlap,
                )

                adv_img = result["worst_degraded"]
                clean_img = image.detach().cpu()

                adv_np = self._tensor_to_uint8_hwc(adv_img)
                clean_np = self._tensor_to_uint8_hwc(clean_img)
                expanded_pairs = self._expand_with_simple_augmentations(adv_np=adv_np, clean_np=clean_np, k=self.adv_aug_k)

                clean_name = Path(clean_path).stem
                for aug_idx, (aug_adv_np, aug_clean_np) in enumerate(expanded_pairs):
                    if len(local_records) >= local_target:
                        break
                    pair_id = f"{pair_offset + len(local_records):06d}"
                    adv_path = input_dir / f"{pair_id}.png"
                    clean_out_path = target_dir / f"{pair_id}.png"
                    self._save_rgb_np(aug_adv_np, adv_path)
                    self._save_rgb_np(aug_clean_np, clean_out_path)

                    local_records.append(
                        {
                            "degraded_path": str(adv_path),
                            "clean_path": str(clean_out_path),
                            "de_id": 4,
                            "clean_name": clean_name,
                            "attack_subset": ["haze"],
                            "aug_index": int(aug_idx),
                        }
                    )
                    progress.update(1)

                produced_base_adv += 1
        finally:
            progress.close()
            for param, req_grad in zip(promptir_params, original_requires_grad):
                param.requires_grad = req_grad
            if original_training:
                promptir_model.train()
            else:
                promptir_model.eval()

        print(
            f"[AdvMixHazeOnlyShard] epoch={epoch} rank={rank}/{world} local_target={local_target} "
            f"base_adv_target={base_adv_target} base_adv_ready={produced_base_adv} local_ready={len(local_records)}"
        )
        return local_records

    def __len__(self) -> int:
        return self.base_len + self.adv_len

    def _load_adv_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        record = self.adv_records[idx]
        degrad_img = crop_img(np.array(Image.open(record["degraded_path"]).convert("RGB")), base=16)
        clean_img = crop_img(np.array(Image.open(record["clean_path"]).convert("RGB")), base=16)
        degrad_img, clean_img = self._ensure_min_hw_numpy_pair(degrad_img, clean_img)
        return degrad_img, clean_img, record

    def __getitem__(self, idx: int):
        if idx < self.base_len:
            return self.base_dataset[idx]

        if len(self.adv_records) == 0:
            mapped_idx = idx % max(1, self.base_len)
            return self.base_dataset[mapped_idx]

        adv_idx = idx - self.base_len
        if adv_idx >= len(self.adv_records):
            adv_idx = adv_idx % len(self.adv_records)

        degrad_img, clean_img, record = self._load_adv_pair(adv_idx)

        if self.base_dataset.data_split == "train":
            degrad_patch, clean_patch = random_augmentation(*self.base_dataset._crop_patch(degrad_img, clean_img))
        else:
            degrad_patch, clean_patch = self.base_dataset._center_crop_patch(degrad_img, clean_img)

        clean_patch = self.base_dataset.toTensor(clean_patch)
        degrad_patch = self.base_dataset.toTensor(degrad_patch)
        return [record["clean_name"], int(record["de_id"])], degrad_patch, clean_patch

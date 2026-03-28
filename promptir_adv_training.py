from __future__ import annotations

import os
import random
import shutil
import sys
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from utils.image_utils import crop_img, random_augmentation  # noqa: E402

from degradation import noise_rain_haze_degradation, random_degradation_configs_from_image  # noqa: E402
from promptir_attack import (  # noqa: E402
    _load_distance_from_mat,
    _load_image_rgb,
    _resize_to_max_side,
    load_promptir_model,
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

        self.adv_steps1 = int(getattr(args, "adv_steps1", 4))
        self.adv_steps2 = int(getattr(args, "adv_steps2", 4))
        self.adv_step_size = float(getattr(args, "adv_step_size", 3e-2))
        self.adv_lambda_reg = float(getattr(args, "adv_lambda_reg", 0.05))
        self.adv_rain_topk = int(getattr(args, "adv_rain_topk", 8))
        self.adv_max_side = int(getattr(args, "adv_max_side", 256))
        self.adv_promptir_ckpt = Path(getattr(args, "adv_promptir_ckpt"))
        self.adv_cache_root = Path(getattr(args, "adv_cache_root", PROMPTIR_ROOT / "data" / "Train" / "adv_pairs"))
        self.adv_attack_device = str(getattr(args, "adv_attack_device", "cuda")).lower()
        self.adv_samples_per_resample = int(getattr(args, "adv_samples_per_resample", 0))

        self.base_len = len(self.base_dataset)
        self.adv_len = self._compute_adv_len()

        self.shared_cache_dir = self.adv_cache_root / "shared"
        self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
        self.adv_records: list[dict[str, Any]] = []
        self._last_resampled_epoch: int | None = None
        self._promptir_model: torch.nn.Module | None = None
        self._promptir_device: torch.device | None = None

        self.depth_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "depth"

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

    def _is_main_process(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return int(os.environ.get("LOCAL_RANK", "0")) == 0

    def _dist_rank_world(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def _local_target_and_offset(self, total: int, rank: int, world: int) -> tuple[int, int]:
        base = total // world
        rem = total % world
        local_target = base + (1 if rank < rem else 0)
        start_offset = rank * base + min(rank, rem)
        return local_target, start_offset

    def _wait_for_manifest(self, epoch: int, timeout_sec: int = 3600) -> None:
        manifest_path = self._manifest_path(epoch)
        start = time.time()
        while True:
            if manifest_path.exists():
                return
            if (time.time() - start) > timeout_sec:
                raise TimeoutError(f"timed out waiting for manifest: {manifest_path}")
            time.sleep(1.0)

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
                    "attack_subset": list(item.get("attack_subset", ["rain", "haze"])),
                }
            )
        return out

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

    def _compute_adv_len(self) -> int:
        if self.base_len <= 0:
            return 0
        if self.adv_samples_per_resample > 0:
            return self.adv_samples_per_resample
        return max(1, int(round(self.base_len * self.adv_ratio)))

    def _get_attack_device(self) -> torch.device:
        if self.adv_attack_device == "cuda" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            num_cuda = torch.cuda.device_count()
            device_idx = min(local_rank, max(0, num_cuda - 1))
            return torch.device(f"cuda:{device_idx}")
        return torch.device("cpu")

    def _load_promptir_model(self) -> torch.nn.Module:
        device = self._get_attack_device()
        if self._promptir_model is None or self._promptir_device != device:
            self._promptir_model = load_promptir_model(self.adv_promptir_ckpt, device=device)
            self._promptir_device = device
        return self._promptir_model

    def _resolve_derain_clean_path(self, rainy_path: str) -> str:
        if "rainy/" in rainy_path and "rain-" in rainy_path:
            return rainy_path.split("rainy")[0] + "gt/norain-" + rainy_path.split("rain-")[-1]
        if "/input/" in rainy_path:
            return rainy_path.replace("/input/", "/target/")
        if "input/" in rainy_path:
            return rainy_path.replace("input/", "target/")
        raise ValueError(f"Unsupported derain path format: {rainy_path}")

    def _resolve_dehaze_clean_path(self, hazy_path: str) -> str:
        if "/haze/reside_ots/haze/" in hazy_path:
            clear_name = hazy_path.replace("/haze/reside_ots/haze/", "/haze/reside_ots/clear/")
            base_name = os.path.basename(clear_name)
            stem, ext = os.path.splitext(base_name)
            clean_stem = stem.split("_")[0]
            candidate = os.path.join(os.path.dirname(clear_name), clean_stem + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"failed to resolve dehaze clean path from {hazy_path}")

    def _resolve_clean_path(self, sample: dict[str, Any]) -> tuple[str, int, str]:
        de_id = int(sample["de_type"])
        source_path = str(sample["clean_id"])

        if de_id < 3:
            clean_path = source_path
        elif de_id == 3:
            clean_path = self._resolve_derain_clean_path(source_path)
        elif de_id == 4:
            clean_path = self._resolve_dehaze_clean_path(source_path)
        else:
            raise ValueError(f"unsupported de_type for adversarial generation: {de_id}")

        clean_name = Path(clean_path).stem
        return clean_path, de_id, clean_name

    def _resolve_depth_mat(self, clean_path: str) -> Path | None:
        mat_path = self.depth_dir / f"{Path(clean_path).stem}.mat"
        if mat_path.exists():
            return mat_path
        return None

    def _sample_attack_subset(self, has_depth: bool) -> list[str]:
        if has_depth:
            # Keep haze/rain coverage and avoid degenerating to noise-only.
            candidate_subsets = [
                ["noise", "haze"],
                ["noise", "rain"],
                ["rain", "haze"],
                ["noise", "rain", "haze"],
            ]
        else:
            candidate_subsets = [
                ["noise", "rain"],
                ["rain"],
            ]
        return random.choice(candidate_subsets)

    def _build_controller(self, image: torch.Tensor, enabled_subset: list[str]) -> noise_rain_haze_degradation:
        noise_cfg, rain_cfg, haze_cfg = random_degradation_configs_from_image(image)
        rain_haze_order = "haze_rain" if ("rain" in enabled_subset and "haze" in enabled_subset and random.random() > 0.5) else "rain_haze"
        return noise_rain_haze_degradation(
            noise_config=noise_cfg,
            rain_config=rain_cfg,
            haze_config=haze_cfg,
            enable_noise=("noise" in enabled_subset),
            enable_rain=("rain" in enabled_subset),
            enable_haze=("haze" in enabled_subset),
            rain_haze_order=rain_haze_order,
        )

    def _tensor_to_uint8_hwc(self, image: torch.Tensor) -> np.ndarray:
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"expected image tensor with shape (C,H,W), got {tuple(image.shape)}")
        image = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return (image * 255.0 + 0.5).astype(np.uint8)

    def _save_rgb_tensor(self, image: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._tensor_to_uint8_hwc(image), mode="RGB").save(path)

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
            f"[AdvMix] epoch={epoch} base={self.base_len} adv_target={self.adv_len} adv_ready={len(self.adv_records)} "
            f"resample_every={self.adv_resample_epochs} sample_root={self._epoch_dir(epoch)}"
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
            promptir_model = self._load_promptir_model()
            device = self._promptir_device if self._promptir_device is not None else torch.device("cpu")
        else:
            promptir_model = promptir_model_override
            try:
                device = next(promptir_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        local_records: list[dict[str, Any]] = []
        max_attempts = max(local_target, local_target * 3)
        attempts = 0

        progress = tqdm(total=local_target, desc=f"adv-resample@epoch{epoch}:rank{rank}", leave=False)
        while len(local_records) < local_target and attempts < max_attempts:
            attempts += 1
            src_idx = random.randrange(self.base_len)
            local_idx = len(local_records)
            sample = self.base_dataset.sample_ids[src_idx]
            clean_path, de_id, clean_name = self._resolve_clean_path(sample)
            if not os.path.exists(clean_path):
                continue

            image = _load_image_rgb(Path(clean_path))
            _, _, h, w = image.shape
            depth_mat = self._resolve_depth_mat(clean_path)
            has_depth = depth_mat is not None
            subset = self._sample_attack_subset(has_depth=has_depth)

            if "haze" in subset:
                if depth_mat is None:
                    subset = [x for x in subset if x != "haze"]
                else:
                    distance_map, _ = _load_distance_from_mat(depth_mat, (h, w))
            else:
                distance_map = torch.ones((1, 1, h, w), dtype=image.dtype)

            if "haze" in subset and depth_mat is None:
                # Skip invalid haze samples without depth map by design.
                continue

            image, distance_map = _resize_to_max_side(image, distance_map, max_side=self.adv_max_side)
            image, distance_map = self._ensure_min_hw_tensor_pair(image, distance_map)
            target = image.clone()

            image = image.to(device)
            target = target.to(device)
            distance_map = distance_map.to(device)

            controller = self._build_controller(image=image, enabled_subset=subset)
            effective_topk = max(1, min(self.adv_rain_topk, controller.rain_module.num_branches))
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
                rain_topk=effective_topk,
                save_dir=None,
                record_history=False,
                save_visual_maps=False,
                allow_promptir_trainable_params=promptir_model_override is not None,
            )

            adv_img = result["worst_degraded"]
            clean_img = image.detach().cpu()

            pair_id = f"{pair_offset + local_idx:06d}"
            adv_path = input_dir / f"{pair_id}.png"
            clean_out_path = target_dir / f"{pair_id}.png"
            self._save_rgb_tensor(adv_img, adv_path)
            self._save_rgb_tensor(clean_img, clean_out_path)

            local_records.append(
                {
                    "degraded_path": str(adv_path),
                    "clean_path": str(clean_out_path),
                    "de_id": int(de_id),
                    "clean_name": clean_name,
                    "attack_subset": subset,
                }
            )
            progress.update(1)
        progress.close()
        print(
            f"[AdvMixShard] epoch={epoch} rank={rank}/{world} local_target={local_target} local_ready={len(local_records)} "
            f"sample_root={epoch_dir}"
        )
        return local_records

    def __len__(self) -> int:
        # Keep dataset length stable for DistributedSampler across all epochs.
        # Actual adversarial records are refreshed in-place; missing slots fallback in __getitem__.
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

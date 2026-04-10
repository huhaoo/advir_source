from __future__ import annotations

import copy
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import StrategyRegistry
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from net.model import build_promptir_model_from_options  # noqa: E402
from options import options as opt  # noqa: E402
from utils.dataset_utils import PromptTrainDataset  # noqa: E402
from utils.image_utils import crop_img, random_augmentation  # noqa: E402
from utils.pytorch_ssim import ssim  # noqa: E402
from utils.schedulers import LinearWarmupCosineAnnealingLR  # noqa: E402


_VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_resume_checkpoint(resume_ckpt: str, ckpt_dir: str, auto_resume: bool) -> str | None:
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        return resume_ckpt

    if auto_resume:
        ckpt_candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_candidates:
            return None
        return max(ckpt_candidates, key=os.path.getmtime)

    return None


def select_multi_gpu_strategy() -> str | None:
    available = set(StrategyRegistry.available_strategies())
    candidates = [
        "ddp_find_unused_parameters_true",
        "ddp",
        "ddp_find_unused_parameters_false",
        "ddp_spawn",
    ]
    for name in candidates:
        if name in available:
            return name
    return None


def should_use_wandb(wblogger: str | None) -> bool:
    if wblogger is None:
        return False
    normalized = str(wblogger).strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return False
    return True


class promptir_static_adv_mix_dataset(Dataset):
    def __init__(self, base_dataset: PromptTrainDataset, args):
        super().__init__()
        self.base_dataset = base_dataset
        self.args = args

        self.adv_ratio = float(getattr(args, "adv_ratio", 0.5))
        if not (0.0 < self.adv_ratio < 1.0):
            raise ValueError(f"adv_ratio must be in (0, 1), got {self.adv_ratio}")

        self.adv_samples_per_resample = int(getattr(args, "adv_samples_per_resample", 0))
        self.base_len = len(self.base_dataset)
        self.adv_len = self._compute_adv_len()

        self.static_adv_root = Path(getattr(args, "adv_cache_root", "/home/huhao/adv_ir/dataset_ours/random_adv_haze"))
        self.input_dir = self.static_adv_root / "input"
        self.target_dir = self.static_adv_root / "target"
        self.adv_records = self._scan_static_adv_pairs()

        if len(self.adv_records) == 0:
            raise RuntimeError(f"no valid pairs found under static adversarial dataset root: {self.static_adv_root}")

    def _compute_adv_len(self) -> int:
        if self.base_len <= 0:
            return 0
        if self.adv_samples_per_resample > 0:
            return self.adv_samples_per_resample
        return max(1, int(round(self.base_len * self.adv_ratio)))

    def _resolve_target_path(self, rel_path: Path) -> Path | None:
        exact = self.target_dir / rel_path
        if exact.exists():
            return exact

        parent_dir = self.target_dir / rel_path.parent
        if not parent_dir.exists():
            return None

        stem = rel_path.stem
        for suffix in _VALID_IMAGE_SUFFIXES:
            candidate = parent_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def _scan_static_adv_pairs(self) -> list[dict[str, str | int]]:
        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(
                f"static adversarial dataset must contain input/target folders, got root={self.static_adv_root}"
            )

        records: list[dict[str, str | int]] = []
        for input_path in sorted(self.input_dir.rglob("*")):
            if not input_path.is_file():
                continue
            if input_path.suffix.lower() not in _VALID_IMAGE_SUFFIXES:
                continue

            rel_path = input_path.relative_to(self.input_dir)
            target_path = self._resolve_target_path(rel_path)
            if target_path is None:
                continue

            records.append(
                {
                    "degraded_path": str(input_path),
                    "clean_path": str(target_path),
                    "clean_name": target_path.stem,
                    "de_id": 4,
                }
            )

        return records

    def _ensure_min_hw_numpy_pair(self, degrad_img: np.ndarray, clean_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        patch_size = int(self.base_dataset.args.patch_size)
        h, w = degrad_img.shape[:2]
        if h >= patch_size and w >= patch_size:
            return degrad_img, clean_img

        new_h = max(h, patch_size)
        new_w = max(w, patch_size)
        degrad_img = np.array(Image.fromarray(degrad_img).resize((new_w, new_h), Image.BICUBIC))
        clean_img = np.array(Image.fromarray(clean_img).resize((new_w, new_h), Image.BICUBIC))
        return degrad_img, clean_img

    def _load_adv_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, str | int]]:
        record = self.adv_records[idx]
        degrad_img = crop_img(np.array(Image.open(record["degraded_path"]).convert("RGB")), base=16)
        clean_img = crop_img(np.array(Image.open(record["clean_path"]).convert("RGB")), base=16)
        degrad_img, clean_img = self._ensure_min_hw_numpy_pair(degrad_img, clean_img)
        return degrad_img, clean_img, record

    def __len__(self) -> int:
        return self.base_len + self.adv_len

    def __getitem__(self, idx: int):
        if idx < self.base_len:
            return self.base_dataset[idx]

        if len(self.adv_records) == 0:
            mapped_idx = idx % max(1, self.base_len)
            return self.base_dataset[mapped_idx]

        adv_idx = (idx - self.base_len) % len(self.adv_records)
        degrad_img, clean_img, record = self._load_adv_pair(adv_idx)

        if self.base_dataset.data_split == "train":
            degrad_patch, clean_patch = random_augmentation(*self.base_dataset._crop_patch(degrad_img, clean_img))
        else:
            degrad_patch, clean_patch = self.base_dataset._center_crop_patch(degrad_img, clean_img)

        clean_patch = self.base_dataset.toTensor(clean_patch)
        degrad_patch = self.base_dataset.toTensor(degrad_patch)
        return [str(record["clean_name"]), int(record["de_id"])], degrad_patch, clean_patch


class promptir_static_adv_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = build_promptir_model_from_options(opt, decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        val_loss = self.loss_fn(restored, clean_patch)
        mse = torch.mean((restored - clean_patch) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
        val_psnr = psnr.mean()
        val_ssim = ssim(torch.clamp(restored, 0, 1), torch.clamp(clean_patch, 0, 1), size_average=True)

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_psnr", val_psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ssim", val_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if all(k in metrics for k in ("val_loss", "val_psnr", "val_ssim")):
            self.print(
                f"[Val] epoch={self.current_epoch} "
                f"loss={metrics['val_loss'].item():.6f} "
                f"psnr={metrics['val_psnr'].item():.4f} "
                f"ssim={metrics['val_ssim'].item():.4f}"
            )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=opt.epochs,
        )
        return [optimizer], [scheduler]


def main() -> None:
    print("Options")
    print(opt)

    if should_use_wandb(opt.wblogger):
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-StaticAdvTrain")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset_base = PromptTrainDataset(opt)
    trainset = promptir_static_adv_mix_dataset(trainset_base, opt)
    print(
        f"[StaticAdvMix] base_len={trainset.base_len} adv_len={trainset.adv_len} "
        f"adv_available={len(trainset.adv_records)} adv_root={trainset.static_adv_root}"
    )

    val_opt = copy.deepcopy(opt)
    val_opt.data_split = "val"
    val_opt.degradation_size = None
    valset = PromptTrainDataset(val_opt)

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    model = promptir_static_adv_model()

    if torch.cuda.is_available() and opt.num_gpus > 0:
        devices = min(opt.num_gpus, torch.cuda.device_count())
        trainer_kwargs = dict(
            max_epochs=opt.epochs,
            accelerator="gpu",
            devices=devices,
            logger=logger,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=opt.accumulate_grad_batches,
            check_val_every_n_epoch=4,
            num_sanity_val_steps=0,
        )
        if devices > 1:
            strategy_name = select_multi_gpu_strategy()
            if strategy_name is not None:
                print(f"Using multi-GPU strategy: {strategy_name}")
                trainer_kwargs["strategy"] = strategy_name
    else:
        trainer_kwargs = dict(
            max_epochs=opt.epochs,
            accelerator="cpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=opt.accumulate_grad_batches,
            check_val_every_n_epoch=4,
            num_sanity_val_steps=0,
        )

    trainer = pl.Trainer(**trainer_kwargs)
    resume_path = resolve_resume_checkpoint(opt.resume_ckpt, opt.ckpt_dir, opt.auto_resume)
    if resume_path is not None:
        print(f"Resuming training from checkpoint: {resume_path}")

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()

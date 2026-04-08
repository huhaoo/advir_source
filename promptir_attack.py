from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from degradation import haze_degradation, random_degradation_configs_from_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
DEFAULT_CKPT = PROMPTIR_ROOT / "train_ckpt_8192" / "epoch=31-step=57344.ckpt"

if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_output_dir(subdir: str = "promptir_attack") -> Path:
    output_root = PROJECT_ROOT / "tmp_demo"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_dir = output_root / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_image_rgb(image_path: Path) -> Tensor:
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.clamp(0.0, 1.0)


def _load_distance_from_mat(depth_path: Path, image_hw: tuple[int, int]) -> tuple[Tensor, dict[str, Any]]:
    with h5py.File(depth_path, "r") as mat:
        if "depth" not in mat:
            raise KeyError(f"key 'depth' not found in {depth_path}")
        depth_np = mat["depth"][()]

    depth = torch.from_numpy(depth_np.copy()).float().unsqueeze(0).unsqueeze(0)
    raw_h, raw_w = int(depth.shape[-2]), int(depth.shape[-1])

    h, w = int(image_hw[0]), int(image_hw[1])
    transposed_to_match = False
    if (raw_h, raw_w) == (w, h):
        depth = depth.transpose(-2, -1).contiguous()
        transposed_to_match = True

    resized_to_match = False
    if depth.shape[-2:] != (h, w):
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        resized_to_match = True

    depth = depth.clamp_min(0.0)
    dmin = depth.amin()
    dmax = depth.amax()
    norm = (depth - dmin) / (dmax - dmin + 1e-6)
    info = {
        "image_hw": (h, w),
        "depth_raw_hw": (raw_h, raw_w),
        "depth_transposed_to_match": bool(transposed_to_match),
        "depth_resized_to_match": bool(resized_to_match),
    }
    return 0.1 + 2.8 * norm, info


def _forward_promptir_with_padding(promptir_model: nn.Module, degraded_image: Tensor, multiple: int = 8) -> Tensor:
    if degraded_image.ndim != 4:
        raise ValueError(f"degraded_image must be (B, C, H, W), got {tuple(degraded_image.shape)}")

    h, w = degraded_image.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return promptir_model(degraded_image)

    padded = F.pad(degraded_image, (0, pad_w, 0, pad_h), mode="reflect")
    restored = promptir_model(padded)
    return restored[:, :, :h, :w]


def _compute_patch_starts(length: int, patch_size: int, patch_overlap: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError(f"patch_overlap must be in [0, patch_size), got overlap={patch_overlap}, patch_size={patch_size}")

    if length <= patch_size:
        return [0]

    stride = patch_size - patch_overlap
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _promptir_task_and_grad(promptir_model: nn.Module, degraded_image: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    restored = _forward_promptir_with_padding(promptir_model, degraded_image)
    task_loss = F.l1_loss(restored, target)
    grad = torch.autograd.grad(task_loss, degraded_image, retain_graph=False, create_graph=False)[0]
    return task_loss, grad.detach(), restored


def _promptir_task_and_grad_tiled(
    promptir_model: nn.Module,
    degraded_image: Tensor,
    target: Tensor,
    patch_size: int,
    patch_overlap: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, h, w = degraded_image.shape
    hs_list = _compute_patch_starts(h, patch_size, patch_overlap)
    ws_list = _compute_patch_starts(w, patch_size, patch_overlap)

    grad_acc = torch.zeros_like(degraded_image)
    restored_acc = torch.zeros_like(degraded_image)
    weight_acc = torch.zeros_like(degraded_image)
    task_loss_sum = degraded_image.new_tensor(0.0)
    patch_count = 0

    for hs in hs_list:
        for ws in ws_list:
            he = min(h, hs + patch_size)
            we = min(w, ws + patch_size)

            degraded_patch = degraded_image[:, :, hs:he, ws:we]
            target_patch = target[:, :, hs:he, ws:we]

            restored_patch = _forward_promptir_with_padding(promptir_model, degraded_patch)
            task_loss_patch = F.l1_loss(restored_patch, target_patch)
            grad_patch = torch.autograd.grad(task_loss_patch, degraded_patch, retain_graph=False, create_graph=False)[0]

            grad_acc[:, :, hs:he, ws:we] += grad_patch
            restored_acc[:, :, hs:he, ws:we] += restored_patch.detach()
            weight_acc[:, :, hs:he, ws:we] += 1.0

            task_loss_sum += task_loss_patch.detach()
            patch_count += 1

    if patch_count <= 0:
        raise RuntimeError("tiled promptir pass produced zero patches")

    denom = float(patch_count)
    grad = grad_acc / denom
    restored = restored_acc / weight_acc.clamp_min(1.0)
    task_loss = task_loss_sum / denom
    return task_loss, grad.detach(), restored


def build_haze_controller_from_image(image: Tensor, beta_mean: float | None = None) -> haze_degradation:
    _, _, haze_cfg = random_degradation_configs_from_image(image)
    if beta_mean is not None:
        haze_cfg.beta_mean = float(beta_mean)
    return haze_degradation(haze_cfg)


def _tensor_to_uint8_hwc(image: Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image_np * 255.0 + 0.5).astype(np.uint8)


def _save_rgb_tensor(image: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tensor_to_uint8_hwc(image), mode="RGB").save(path)


def run_single_image_adversarial_degradation_search(
    image: Tensor,
    target: Tensor,
    promptir_model: nn.Module,
    degradation_controller: haze_degradation,
    distance_map: Tensor | None = None,
    steps1: int = 2,
    steps2: int = 2,
    step_size: float = 3e-2,
    lambda_reg: float = 0.05,
    rain_topk: int = 1,
    save_dir: Path | None = None,
    record_history: bool = True,
    save_visual_maps: bool = False,
    allow_promptir_trainable_params: bool = False,
    promptir_patch_size: int | None = None,
    promptir_patch_overlap: int = 0,
) -> dict[str, Any]:
    del rain_topk
    del save_visual_maps

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"image must be shape (1, C, H, W), got {tuple(image.shape)}")
    if target.shape != image.shape:
        raise ValueError(f"target shape must equal image shape, got image={tuple(image.shape)}, target={tuple(target.shape)}")
    if not isinstance(degradation_controller, haze_degradation):
        raise ValueError(f"degradation_controller must be haze_degradation, got {type(degradation_controller)!r}")

    if distance_map is None:
        distance_map = torch.ones((1, 1, image.shape[-2], image.shape[-1]), dtype=image.dtype, device=image.device)

    if steps1 <= 0 or steps2 <= 0:
        raise ValueError(f"steps1 and steps2 must be > 0, got steps1={steps1}, steps2={steps2}")

    if promptir_patch_size is not None and promptir_patch_size <= 0:
        raise ValueError(f"promptir_patch_size must be > 0 when provided, got {promptir_patch_size}")

    if promptir_patch_size is not None and (promptir_patch_overlap < 0 or promptir_patch_overlap >= promptir_patch_size):
        raise ValueError(
            f"promptir_patch_overlap must be in [0, promptir_patch_size), got overlap={promptir_patch_overlap}, patch={promptir_patch_size}"
        )

    if not allow_promptir_trainable_params:
        for parameter in promptir_model.parameters():
            if parameter.requires_grad:
                raise RuntimeError("promptir_model parameters must be frozen (requires_grad=False)")

    degradation_controller = degradation_controller.to(image.device)
    degradation_controller.train()
    degradation_controller.reset_parameters()

    attack_params = [parameter for parameter in degradation_controller.parameters() if parameter.requires_grad]
    if len(attack_params) == 0:
        raise RuntimeError("haze controller has no trainable parameters")

    optimizer = torch.optim.Adam(attack_params, lr=step_size)

    state = {"current_enabled": ["haze"], "current_order": ["haze"]}
    history: list[dict[str, Any]] = []

    best_attack_obj = float("-inf")
    best_task_loss = float("-inf")
    best_attack_step = (-1, -1)
    best_task_step = (-1, -1)

    initial_task_loss: Tensor | None = None
    initial_reg_loss: Tensor | None = None
    initial_attack_obj: Tensor | None = None

    worst_degraded: Tensor | None = None
    worst_restored: Tensor | None = None

    for outer_idx in range(steps1):
        x_deg_outer = degradation_controller(image, distance_map=distance_map).requires_grad_(True)
        if promptir_patch_size is None:
            task_loss_outer, fixed_grad, x_hat_outer = _promptir_task_and_grad(promptir_model, x_deg_outer, target)
        else:
            task_loss_outer, fixed_grad, x_hat_outer = _promptir_task_and_grad_tiled(
                promptir_model=promptir_model,
                degraded_image=x_deg_outer,
                target=target,
                patch_size=promptir_patch_size,
                patch_overlap=promptir_patch_overlap,
            )

        reg_loss_outer = degradation_controller.get_regularization_loss()
        attack_obj_outer = task_loss_outer - lambda_reg * reg_loss_outer

        if outer_idx == 0:
            initial_task_loss = task_loss_outer.detach().clone()
            initial_reg_loss = reg_loss_outer.detach().clone()
            initial_attack_obj = attack_obj_outer.detach().clone()

        attack_value = float(attack_obj_outer.detach().item())
        task_value = float(task_loss_outer.detach().item())

        if attack_value > best_attack_obj:
            best_attack_obj = attack_value
            best_attack_step = (outer_idx, -1)

        if task_value > best_task_loss:
            best_task_loss = task_value
            worst_degraded = x_deg_outer.detach().clone()
            worst_restored = x_hat_outer.detach().clone()
            best_task_step = (outer_idx, -1)

        for inner_idx in range(steps2):
            optimizer.zero_grad(set_to_none=True)
            x_deg_inner = degradation_controller(image, distance_map=distance_map)
            reg_loss_inner = degradation_controller.get_regularization_loss()

            # Use first-order surrogate with fixed PromptIR gradient from outer step.
            surrogate_task = (x_deg_inner * fixed_grad).mean()
            surrogate_obj = surrogate_task - lambda_reg * reg_loss_inner
            (-surrogate_obj).backward()
            optimizer.step()

            if record_history:
                history.append(
                    {
                        "outer_step": int(outer_idx),
                        "inner_step": int(inner_idx),
                        "task_loss_outer": float(task_value),
                        "attack_obj_outer": float(attack_value),
                        "surrogate_obj": float(surrogate_obj.detach().item()),
                    }
                )

    if initial_task_loss is None or initial_reg_loss is None or initial_attack_obj is None:
        raise RuntimeError("failed to initialize attack states")
    if worst_degraded is None or worst_restored is None:
        raise RuntimeError("failed to produce adversarial haze sample")

    final_reg_loss = float(degradation_controller.get_regularization_loss().detach().item())

    save_info: dict[str, Any] = {}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        degraded_path = save_dir / "worst_degraded.png"
        restored_path = save_dir / "worst_restored.png"
        target_path = save_dir / "target.png"
        _save_rgb_tensor(worst_degraded, degraded_path)
        _save_rgb_tensor(worst_restored, restored_path)
        _save_rgb_tensor(target, target_path)

        log_obj = {
            "state": state,
            "steps1": int(steps1),
            "steps2": int(steps2),
            "step_size": float(step_size),
            "lambda_reg": float(lambda_reg),
            "initial_task_loss": float(initial_task_loss.item()),
            "initial_reg_loss": float(initial_reg_loss.item()),
            "initial_attack_obj": float(initial_attack_obj.item()),
            "best_task_loss": float(best_task_loss),
            "best_attack_obj": float(best_attack_obj),
            "best_task_step": best_task_step,
            "best_attack_step": best_attack_step,
            "final_reg_loss": float(final_reg_loss),
            "history": history if record_history else [],
        }
        log_path = save_dir / "attack_log.json"
        log_path.write_text(json.dumps(log_obj, indent=2), encoding="utf-8")

        save_info = {
            "save_dir": str(save_dir),
            "worst_degraded": str(degraded_path),
            "worst_restored": str(restored_path),
            "target": str(target_path),
            "attack_log": str(log_path),
        }

    return {
        "state": state,
        "initial_task_loss": float(initial_task_loss.item()),
        "initial_reg_loss": float(initial_reg_loss.item()),
        "initial_attack_obj": float(initial_attack_obj.item()),
        "final_task_loss": float(best_task_loss),
        "final_reg_loss": float(final_reg_loss),
        "best_attack_obj": float(best_attack_obj),
        "best_task_loss": float(best_task_loss),
        "best_attack_step": best_attack_step,
        "best_task_step": best_task_step,
        "promptir_calls": int(steps1),
        "output_shape": tuple(worst_restored.shape),
        "history": history,
        "save_info": save_info,
        "rain_auto_params": [],
        "last_grad_norms": {"haze_low_res_param_grad_norm": None},
        "worst_degraded": worst_degraded.detach().cpu(),
        "worst_restored": worst_restored.detach().cpu(),
    }

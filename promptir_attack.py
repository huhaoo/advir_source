from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from degradation import (
    noise_rain_haze_degradation,
    random_degradation_configs_from_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
DEFAULT_CKPT = PROMPTIR_ROOT / "train_ckpt_8192" / "epoch=31-step=57344.ckpt"

if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))
from net.model import PromptIR  # noqa: E402


def _prepare_output_dir(subdir: str = "promptir_attack") -> Path:
    output_root = PROJECT_ROOT / "tmp_demo"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_dir = output_root / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_image_2d_or_3d(x: Tensor) -> Tensor:
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in {1, 3}:
        return x
    if x.ndim == 2:
        return x
    raise ValueError(f"unsupported tensor shape for visualization: {tuple(x.shape)}")


def _save_tensor_png(x: Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _to_image_2d_or_3d(x).detach().cpu()
    plt.figure(figsize=(5, 4))
    if t.ndim == 3:
        if t.shape[0] == 1:
            plt.imshow(t[0].clamp(0, 1).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
    else:
        plt.imshow(t.numpy(), cmap="magma")
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _find_reside_pair() -> tuple[Path, Path, str]:
    clear_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "clear"
    depth_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "depth"
    if not clear_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"reside_ots clear/depth dirs not found: {clear_dir}, {depth_dir}")

    clear_files = sorted(clear_dir.glob("*.jpg"))
    if len(clear_files) == 0:
        raise FileNotFoundError(f"no clear images found in {clear_dir}")

    for clear_path in clear_files:
        sample_id = clear_path.stem
        depth_path = depth_dir / f"{sample_id}.mat"
        if depth_path.exists():
            return clear_path, depth_path, sample_id
    raise FileNotFoundError("no matched clear jpg and depth mat pair found in reside_ots")


def _list_reside_pairs(limit: int | None = None) -> list[tuple[Path, Path, str]]:
    clear_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "clear"
    depth_dir = PROJECT_ROOT / "dataset" / "haze" / "reside_ots" / "depth"
    if not clear_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"reside_ots clear/depth dirs not found: {clear_dir}, {depth_dir}")

    clear_files = sorted(clear_dir.glob("*.jpg"))
    pairs: list[tuple[Path, Path, str]] = []
    for clear_path in clear_files:
        sample_id = clear_path.stem
        depth_path = depth_dir / f"{sample_id}.mat"
        if depth_path.exists():
            pairs.append((clear_path, depth_path, sample_id))
    if len(pairs) == 0:
        raise FileNotFoundError("no matched clear jpg and depth mat pair found in reside_ots")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be > 0 when provided, got {limit}")
        pairs = pairs[:limit]
    return pairs


def _load_image_rgb(image_path: Path) -> Tensor:
    img_np = plt.imread(image_path)
    if img_np.ndim != 3 or img_np.shape[2] not in {3, 4}:
        raise ValueError(f"image must be HxWx3/4, got shape {img_np.shape}")
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    image = torch.from_numpy(img_np.copy()).float()
    if image.max().item() > 1.0:
        image = image / 255.0
    return image.permute(2, 0, 1).unsqueeze(0).contiguous().clamp(0.0, 1.0)


def _load_distance_from_mat(depth_path: Path, image_hw: tuple[int, int]) -> tuple[Tensor, dict[str, Any]]:
    with h5py.File(depth_path, "r") as mat:
        if "depth" not in mat:
            raise KeyError(f"key 'depth' not found in {depth_path}")
        depth_np = mat["depth"][()]
    depth = torch.from_numpy(depth_np.copy()).float().unsqueeze(0).unsqueeze(0)
    raw_h, raw_w = int(depth.shape[-2]), int(depth.shape[-1])
    h, w = image_hw
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
        "image_hw": (int(h), int(w)),
        "depth_raw_hw": (raw_h, raw_w),
        "depth_transposed_to_match": bool(transposed_to_match),
        "depth_resized_to_match": bool(resized_to_match),
    }
    return 0.1 + 2.8 * norm, info


def _resize_to_max_side(image: Tensor, distance_map: Tensor, max_side: int | None) -> tuple[Tensor, Tensor]:
    if max_side is None:
        return image, distance_map
    if max_side <= 0:
        raise ValueError(f"max_side must be positive or None, got {max_side}")

    _, _, h, w = image.shape
    long_side = max(h, w)
    if long_side <= max_side:
        return image, distance_map

    scale = float(max_side) / float(long_side)
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    image_resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    distance_resized = F.interpolate(distance_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return image_resized, distance_resized


def load_promptir_model(checkpoint_path: Path, device: torch.device) -> PromptIR:
    try:
        checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint_obj["state_dict"] if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj else checkpoint_obj
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a valid state_dict")

    promptir_state: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            promptir_state[key[4:]] = value

    if len(promptir_state) == 0:
        raise ValueError("no 'net.' prefixed keys found in checkpoint state_dict")

    model = PromptIR(decoder=True)
    missing_keys, unexpected_keys = model.load_state_dict(promptir_state, strict=False)
    if len(unexpected_keys) > 0:
        raise RuntimeError(f"unexpected PromptIR keys while loading checkpoint: {unexpected_keys[:10]}")
    if len(missing_keys) > 0:
        raise RuntimeError(f"missing PromptIR keys while loading checkpoint: {missing_keys[:10]}")

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


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


def _promptir_task_and_grad(promptir_model: nn.Module, degraded_image: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    restored = _forward_promptir_with_padding(promptir_model, degraded_image)
    task_loss = F.l1_loss(restored, target)
    grad = torch.autograd.grad(task_loss, degraded_image, retain_graph=False, create_graph=False)[0]
    return task_loss, grad.detach(), restored


def _sample_num_enabled(enable_count_probs: tuple[float, float, float]) -> int:
    if len(enable_count_probs) != 3:
        raise ValueError("enable_count_probs must contain exactly 3 probabilities for selecting 1/2/3 degradations")
    probs = torch.tensor(enable_count_probs, dtype=torch.float32)
    if torch.any(probs < 0):
        raise ValueError(f"enable_count_probs must be non-negative, got {enable_count_probs}")
    total = float(probs.sum().item())
    if total <= 0:
        raise ValueError(f"enable_count_probs sum must be > 0, got {enable_count_probs}")
    probs = probs / total
    return int(torch.multinomial(probs, num_samples=1).item()) + 1


def _sample_enabled_subset(enable_count_probs: tuple[float, float, float]) -> list[str]:
    all_types = ["noise", "rain", "haze"]
    target_num = _sample_num_enabled(enable_count_probs)
    chosen_num = min(target_num, len(all_types))
    indices = torch.randperm(len(all_types))[:chosen_num].tolist()
    return [all_types[idx] for idx in indices]


def _sample_rain_haze_order(enabled_subset: list[str]) -> str:
    if "rain" in enabled_subset and "haze" in enabled_subset and bool(torch.randint(0, 2, (1,)).item()):
        return "haze_rain"
    return "rain_haze"


def build_default_controller_from_image(
    image: Tensor,
    enable_count_probs: tuple[float, float, float] = (0.5, 0.4, 0.1),
) -> noise_rain_haze_degradation:
    noise_cfg, rain_cfg, haze_cfg = random_degradation_configs_from_image(image)
    enabled_subset = _sample_enabled_subset(enable_count_probs)
    rain_haze_order = _sample_rain_haze_order(enabled_subset)
    return noise_rain_haze_degradation(
        noise_config=noise_cfg,
        rain_config=rain_cfg,
        haze_config=haze_cfg,
        enable_noise="noise" in enabled_subset,
        enable_rain="rain" in enabled_subset,
        enable_haze="haze" in enabled_subset,
        rain_haze_order=rain_haze_order,
    )


def build_rain_haze_max_controller_from_image(image: Tensor) -> noise_rain_haze_degradation:
    """Build a fixed rain+haze controller with maxed hyperparameters."""
    noise_cfg, rain_cfg, haze_cfg = random_degradation_configs_from_image(image)

    # Max settings based on current random sampling ranges in degradation.py
    rain_cfg.router_config.num_maps = 16
    haze_cfg.airlight_init = (1.0, 1.0, 1.0)
    haze_cfg.beta_mean = 0.5
    haze_cfg.beta_std = None

    return noise_rain_haze_degradation(
        noise_config=noise_cfg,
        rain_config=rain_cfg,
        haze_config=haze_cfg,
        enable_noise=False,
        enable_rain=True,
        enable_haze=True,
        rain_haze_order="rain_haze",
    )


def _extract_haze_maps(controller: noise_rain_haze_degradation, distance_map: Tensor) -> tuple[Tensor, Tensor]:
    haze_module = controller.haze_module
    beta_map = haze_module._compute_beta_map()
    transmission = torch.exp(-beta_map * distance_map)
    return beta_map, transmission


def _collect_gradient_norms(controller: noise_rain_haze_degradation) -> dict[str, float | None]:
    haze_param = controller.haze_module.beta_map_module.low_res_param
    rain_param = controller.rain_module.router.maps[0].low_res_param

    haze_grad = None if haze_param.grad is None else float(haze_param.grad.detach().norm().item())
    rain_grad = None if rain_param.grad is None else float(rain_param.grad.detach().norm().item())
    return {
        "haze_low_res_param_grad_norm": haze_grad,
        "rain_router_map0_grad_norm": rain_grad,
    }


def run_single_image_adversarial_degradation_search(
    image: Tensor,
    target: Tensor,
    promptir_model: nn.Module,
    degradation_controller: noise_rain_haze_degradation,
    distance_map: Tensor | None = None,
    steps1: int = 4,
    steps2: int = 4,
    step_size: float = 5e-2,
    lambda_reg: float = 0.05,
    rain_topk: int = 3,
    save_dir: Path | None = None,
    record_history: bool = True,
    save_visual_maps: bool = True,
    allow_promptir_trainable_params: bool = False,
) -> dict[str, Any]:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"image must be shape (1, C, H, W), got {tuple(image.shape)}")
    if target.shape != image.shape:
        raise ValueError(f"target shape must equal image shape, got image={tuple(image.shape)}, target={tuple(target.shape)}")
    if distance_map is None:
        distance_map = torch.ones((1, 1, image.shape[-2], image.shape[-1]), dtype=image.dtype, device=image.device)
    if distance_map.ndim != 4 or distance_map.shape[0] != 1:
        raise ValueError(f"distance_map must be shape (1, 1|C, H, W), got {tuple(distance_map.shape)}")
    if steps1 <= 0:
        raise ValueError(f"steps1 must be > 0, got {steps1}")
    if steps2 <= 0:
        raise ValueError(f"steps2 must be > 0, got {steps2}")

    device = image.device
    degradation_controller = degradation_controller.to(device)
    degradation_controller.train()

    if not allow_promptir_trainable_params:
        for parameter in promptir_model.parameters():
            if parameter.requires_grad:
                raise RuntimeError("promptir_model parameters must be frozen (requires_grad=False)")

    attack_params = []
    for parameter in degradation_controller.parameters():
        if parameter.requires_grad:
            attack_params.append(parameter)
    if len(attack_params) == 0:
        raise RuntimeError("degradation_controller has no trainable parameters to optimize")

    degradation_controller.reset_parameters()
    state = degradation_controller.get_current_state()
    rain_candidates: list[Tensor] | None = None
    if "rain" in state["current_enabled"]:
        rain_candidates = degradation_controller.rain_module.build_and_store_auto_candidates(image)

    optimizer = torch.optim.Adam(attack_params, lr=step_size)

    initial_deg: Tensor | None = None
    initial_hat: Tensor | None = None
    initial_task_loss: Tensor | None = None
    initial_reg_loss: Tensor | None = None
    initial_attack_obj: Tensor | None = None

    history: list[dict[str, Any]] = []
    best_attack_obj = float("-inf")
    best_task_loss = float("-inf")
    worst_deg: Tensor | None = None
    worst_hat: Tensor | None = None
    best_attack_step = (-1, -1)
    best_task_step = (-1, -1)
    last_grad_norms: dict[str, float | None] = {"haze_low_res_param_grad_norm": None, "rain_router_map0_grad_norm": None}

    # Hierarchical GD: expensive PromptIR call only steps1 times.
    for outer_idx in range(steps1):
        x_deg_outer = degradation_controller(image, distance_map=distance_map, rain_degraded_list=rain_candidates, rain_topk=rain_topk)
        x_deg_outer = x_deg_outer.requires_grad_(True)
        task_loss_outer, fixed_grad, x_hat_outer = _promptir_task_and_grad(promptir_model, x_deg_outer, target)
        reg_loss_outer = degradation_controller.get_regularization_loss()
        attack_obj_outer = task_loss_outer - lambda_reg * reg_loss_outer

        if outer_idx == 0:
            initial_deg = x_deg_outer.detach().clone()
            initial_hat = x_hat_outer.detach().clone()
            initial_task_loss = task_loss_outer.detach().clone()
            initial_reg_loss = reg_loss_outer.detach().clone()
            initial_attack_obj = attack_obj_outer.detach().clone()

        attack_outer_value = float(attack_obj_outer.detach().item())
        task_outer_value = float(task_loss_outer.detach().item())
        reg_outer_value = float(reg_loss_outer.detach().item())
        if attack_outer_value > best_attack_obj:
            best_attack_obj = attack_outer_value
            best_attack_step = (outer_idx, -1)
        if task_outer_value > best_task_loss:
            best_task_loss = task_outer_value
            worst_deg = x_deg_outer.detach().clone()
            worst_hat = x_hat_outer.detach().clone()
            best_task_step = (outer_idx, -1)

        for inner_idx in range(steps2):
            optimizer.zero_grad(set_to_none=True)
            x_deg_inner = degradation_controller(image, distance_map=distance_map, rain_degraded_list=rain_candidates, rain_topk=rain_topk)
            reg_loss_inner = degradation_controller.get_regularization_loss()

            # First-order surrogate using fixed PromptIR gradient from current outer step.
            surrogate_task = (x_deg_inner * fixed_grad).mean()
            surrogate_obj = surrogate_task - lambda_reg * reg_loss_inner
            loss_to_minimize = -surrogate_obj
            loss_to_minimize.backward()
            if record_history:
                last_grad_norms = _collect_gradient_norms(degradation_controller)
            optimizer.step()

            if record_history:
                history.append(
                    {
                        "outer_step": outer_idx,
                        "inner_step": inner_idx,
                        "task_loss_outer": task_outer_value,
                        "reg_loss_outer": reg_outer_value,
                        "attack_obj_outer": attack_outer_value,
                        "surrogate_task": float(surrogate_task.detach().item()),
                        "surrogate_obj": float(surrogate_obj.detach().item()),
                        "enabled": list(state["current_enabled"]),
                        "order": list(state["current_order"]),
                        "grad_norms": last_grad_norms,
                    }
                )

    if initial_deg is None or initial_hat is None or initial_task_loss is None or initial_reg_loss is None or initial_attack_obj is None:
        raise RuntimeError("failed to initialize hierarchical attack state")
    if worst_deg is None or worst_hat is None:
        raise RuntimeError("failed to collect worst-case degradation during attack optimization")

    final_task_loss = best_task_loss
    final_reg_loss = float(degradation_controller.get_regularization_loss().detach().item())

    save_info: dict[str, Any] = {}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_original = save_dir / "original.png"
        file_target = save_dir / "target.png"
        file_initial_deg = save_dir / "initial_degraded.png"
        file_worst_deg = save_dir / "worst_degraded.png"
        file_initial_hat = save_dir / "initial_restored.png"
        file_worst_hat = save_dir / "worst_restored.png"
        file_distance = save_dir / "distance_map.png"
        file_log_json = save_dir / "attack_log.json"
        file_log_txt = save_dir / "attack_log.txt"
        file_state_json = save_dir / "attack_state.json"

        _save_tensor_png(image, file_original, "clean_input")
        _save_tensor_png(target, file_target, "target")
        _save_tensor_png(initial_deg, file_initial_deg, "initial_degraded")
        _save_tensor_png(worst_deg, file_worst_deg, "worst_degraded")
        _save_tensor_png(initial_hat, file_initial_hat, "initial_restored")
        _save_tensor_png(worst_hat, file_worst_hat, "worst_restored")
        _save_tensor_png(distance_map[:, :1], file_distance, "distance_map")

        if save_visual_maps and "rain" in state["current_enabled"]:
            weight_maps = degradation_controller.rain_module.get_weight_maps(topk=rain_topk)
            for idx in range(weight_maps.shape[1]):
                _save_tensor_png(
                    weight_maps[0, idx, 0],
                    save_dir / f"rain_weight_map_{idx}.png",
                    f"rain_weight_map_{idx}",
                )

        if save_visual_maps and "haze" in state["current_enabled"]:
            haze_beta, haze_transmission = _extract_haze_maps(degradation_controller, distance_map)
            _save_tensor_png(haze_beta[0, 0], save_dir / "haze_beta_map.png", "haze_beta_map")
            _save_tensor_png(haze_transmission[0, 0], save_dir / "haze_transmission_map.png", "haze_transmission_map")

        log_obj = {
            "initial_task_loss": float(initial_task_loss.item()),
            "initial_reg_loss": float(initial_reg_loss.item()),
            "initial_attack_obj": float(initial_attack_obj.item()),
            "best_attack_obj": best_attack_obj,
            "best_attack_step": best_attack_step,
            "best_task_step": best_task_step,
            "final_task_loss": final_task_loss,
            "final_reg_loss": final_reg_loss,
            "steps1": steps1,
            "steps2": steps2,
            "step_size": step_size,
            "lambda_reg": lambda_reg,
            "rain_topk": rain_topk,
            "history": history if record_history else [],
        }
        file_log_json.write_text(json.dumps(log_obj, indent=2), encoding="utf-8")

        lines = [
            "promptir_single_image_attack",
            f"enabled={state['current_enabled']}",
            f"order={state['current_order']}",
            f"initial_task_loss={float(initial_task_loss.item()):.10f}",
            f"initial_reg_loss={float(initial_reg_loss.item()):.10f}",
            f"initial_attack_obj={float(initial_attack_obj.item()):.10f}",
            f"best_attack_obj={best_attack_obj:.10f}",
            f"best_attack_step={best_attack_step}",
            f"best_task_step={best_task_step}",
            f"final_task_loss={final_task_loss:.10f}",
            f"final_reg_loss={final_reg_loss:.10f}",
            f"promptir_calls={steps1}",
            f"output_shape={tuple(worst_hat.shape)}",
        ]
        file_log_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        file_state_json.write_text(json.dumps(state, indent=2), encoding="utf-8")

        save_info = {
            "save_dir": str(save_dir),
            "original": str(file_original),
            "target": str(file_target),
            "initial_degraded": str(file_initial_deg),
            "worst_degraded": str(file_worst_deg),
            "initial_restored": str(file_initial_hat),
            "worst_restored": str(file_worst_hat),
            "distance_map": str(file_distance),
            "attack_log_json": str(file_log_json),
            "attack_log_txt": str(file_log_txt),
            "state_json": str(file_state_json),
        }

    rain_auto_params = getattr(degradation_controller.rain_module, "last_auto_rain_params", None)
    if rain_auto_params is None:
        rain_auto_params = []

    return {
        "state": state,
        "initial_task_loss": float(initial_task_loss.item()),
        "initial_reg_loss": float(initial_reg_loss.item()),
        "initial_attack_obj": float(initial_attack_obj.item()),
        "final_task_loss": final_task_loss,
        "final_reg_loss": final_reg_loss,
        "best_attack_obj": best_attack_obj,
        "best_task_loss": best_task_loss,
        "best_attack_step": best_attack_step,
        "best_task_step": best_task_step,
        "promptir_calls": steps1,
        "output_shape": tuple(worst_hat.shape),
        "history": history,
        "save_info": save_info,
        "rain_auto_params": list(rain_auto_params),
        "last_grad_norms": last_grad_norms,
        "worst_degraded": worst_deg.detach().cpu(),
        "worst_restored": worst_hat.detach().cpu(),
    }


def validate_random_selection_rules(
    num_trials: int = 1000,
    enable_count_probs: tuple[float, float, float] = (0.5, 0.4, 0.1),
) -> dict[str, Any]:
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")

    count_by_num = {1: 0, 2: 0, 3: 0}
    count_by_order = {"rain_then_haze": 0, "haze_then_rain": 0}
    noise_not_last_violations = 0

    for _ in range(num_trials):
        enabled = _sample_enabled_subset(enable_count_probs)
        rain_haze_order = _sample_rain_haze_order(enabled)

        order: list[str] = []
        has_rain = "rain" in enabled
        has_haze = "haze" in enabled
        has_noise = "noise" in enabled
        if has_rain and has_haze:
            order.extend(["rain", "haze"] if rain_haze_order == "rain_haze" else ["haze", "rain"])
        elif has_rain:
            order.append("rain")
        elif has_haze:
            order.append("haze")
        if has_noise:
            order.append("noise")

        count_by_num[len(enabled)] += 1
        if "noise" in enabled and order[-1] != "noise":
            noise_not_last_violations += 1
        if "rain" in enabled and "haze" in enabled:
            if order.index("rain") < order.index("haze"):
                count_by_order["rain_then_haze"] += 1
            else:
                count_by_order["haze_then_rain"] += 1

    ratio_by_num = {k: v / float(num_trials) for k, v in count_by_num.items()}
    return {
        "num_trials": num_trials,
        "count_by_num": count_by_num,
        "ratio_by_num": ratio_by_num,
        "rain_haze_order_counts": count_by_order,
        "noise_not_last_violations": noise_not_last_violations,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_promptir_single_image_attack(
    checkpoint_path: Path = DEFAULT_CKPT,
    steps1: int = 4,
    steps2: int = 4,
    step_size: float = 5e-2,
    lambda_reg: float = 0.05,
    device_name: str = "cpu",
    num_gpus: int = 2,
    max_side: int = 256,
    seed: int = 123,
) -> None:
    _seed_everything(seed)
    output_dir = _prepare_output_dir("promptir_attack")

    clear_path, depth_path, sample_id = _find_reside_pair()
    image = _load_image_rgb(clear_path)
    target = image.clone()
    _, _, h, w = image.shape
    distance_map, depth_match_info = _load_distance_from_mat(depth_path, (h, w))
    image, distance_map = _resize_to_max_side(image, distance_map, max_side=max_side)
    target = image.clone()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device_name='cuda' but CUDA is not available")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError(f"device_name must be 'cpu' or 'cuda', got {device_name}")
    device = torch.device(device_name)
    image = image.to(device)
    target = target.to(device)
    distance_map = distance_map.to(device)

    available_cuda = torch.cuda.device_count() if device_name == "cuda" else 0
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    used_gpus = min(num_gpus, available_cuda) if device_name == "cuda" else 0

    base_promptir_model = load_promptir_model(checkpoint_path, device=device)
    promptir_model: nn.Module = base_promptir_model
    if device_name == "cuda" and used_gpus > 1:
        device_ids = list(range(used_gpus))
        promptir_model = nn.DataParallel(base_promptir_model, device_ids=device_ids)

    controller = build_rain_haze_max_controller_from_image(image)
    effective_rain_topk = controller.rain_module.num_branches

    result = run_single_image_adversarial_degradation_search(
        image=image,
        target=target,
        promptir_model=promptir_model,
        degradation_controller=controller,
        distance_map=distance_map,
        steps1=steps1,
        steps2=steps2,
        step_size=step_size,
        lambda_reg=lambda_reg,
        rain_topk=effective_rain_topk,
        save_dir=output_dir,
    )

    all_frozen = all(not p.requires_grad for p in promptir_model.parameters())
    final_grad_norms = result["history"][-1]["grad_norms"] if len(result["history"]) > 0 else {}

    print(f"sample_id={sample_id}")
    print(f"ckpt={checkpoint_path}")
    print(f"device={device_name}")
    print(f"num_gpus_requested={num_gpus}")
    print(f"num_gpus_used={used_gpus}")
    print(f"depth_match_info={depth_match_info}")
    print(f"max_side={max_side}")
    print(f"promptir_frozen={all_frozen}")
    print(f"current_enabled={result['state']['current_enabled']}")
    print(f"current_order={result['state']['current_order']}")
    print(f"fixed_policy=rain+haze_max")
    print(f"effective_rain_topk={effective_rain_topk}")
    print(f"steps1={steps1}")
    print(f"steps2={steps2}")
    print(f"promptir_calls={result['promptir_calls']}")
    print(f"initial_task_loss={result['initial_task_loss']:.10f}")
    print(f"final_task_loss={result['final_task_loss']:.10f}")
    print(f"best_task_loss={result['best_task_loss']:.10f}")
    print(f"best_attack_obj={result['best_attack_obj']:.10f}")
    print(f"best_attack_step={result['best_attack_step']}")
    print(f"best_task_step={result['best_task_step']}")
    print(f"grad_norms_last_step={final_grad_norms}")
    print(f"output_shape={result['output_shape']}")
    print(f"rain_auto_params={result['rain_auto_params']}")
    print(f"save_dir={output_dir}")


def _run_worker_on_gpu(
    gpu_index: int,
    assigned_pairs: list[tuple[str, str, str]],
    output_root: str,
    checkpoint_path: str,
    steps1: int,
    steps2: int,
    step_size: float,
    lambda_reg: float,
    max_side: int,
    seed: int,
    save_per_sample: bool,
) -> list[dict[str, Any]]:
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    _seed_everything(seed + gpu_index)

    model = load_promptir_model(Path(checkpoint_path), device=device)
    worker_results: list[dict[str, Any]] = []

    for idx, (clear_path_s, depth_path_s, sample_id) in enumerate(assigned_pairs):
        t0 = time.perf_counter()
        clear_path = Path(clear_path_s)
        depth_path = Path(depth_path_s)

        image = _load_image_rgb(clear_path)
        _, _, h, w = image.shape
        distance_map, depth_match_info = _load_distance_from_mat(depth_path, (h, w))
        image, distance_map = _resize_to_max_side(image, distance_map, max_side=max_side)
        target = image.clone()

        image = image.to(device)
        target = target.to(device)
        distance_map = distance_map.to(device)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        controller = build_rain_haze_max_controller_from_image(image)
        effective_rain_topk = controller.rain_module.num_branches

        save_dir = Path(output_root) / f"sample_{sample_id}" if save_per_sample else None
        result = run_single_image_adversarial_degradation_search(
            image=image,
            target=target,
            promptir_model=model,
            degradation_controller=controller,
            distance_map=distance_map,
            steps1=steps1,
            steps2=steps2,
            step_size=step_size,
            lambda_reg=lambda_reg,
            rain_topk=effective_rain_topk,
            save_dir=save_dir,
        )
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        save_time_sec = 0.0
        if save_per_sample:
            # Saving is included in attack call; keep explicit field for easier profiling readout.
            save_time_sec = max(0.0, t2 - t1)

        worker_results.append(
            {
                "gpu_index": gpu_index,
                "worker_local_index": idx,
                "sample_id": sample_id,
                "depth_match_info": depth_match_info,
                "initial_task_loss": result["initial_task_loss"],
                "final_task_loss": result["final_task_loss"],
                "best_task_loss": result["best_task_loss"],
                "save_dir": None if save_dir is None else str(save_dir),
                "timing": {
                    "load_prepare_sec": float(max(0.0, t1 - t0)),
                    "attack_and_optional_save_sec": float(max(0.0, t2 - t1)),
                    "save_component_note_sec": float(save_time_sec),
                    "total_sample_sec": float(max(0.0, t2 - t0)),
                },
            }
        )
    return worker_results


def run_promptir_multi_image_attack_queue(
    checkpoint_path: Path,
    num_samples: int,
    steps1: int,
    steps2: int,
    step_size: float,
    lambda_reg: float,
    num_gpus: int,
    max_side: int,
    seed: int,
    save_per_sample: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("multi-image queue mode requires CUDA")
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be >=1, got {num_gpus}")
    available = torch.cuda.device_count()
    used_gpus = min(num_gpus, available)
    if used_gpus <= 0:
        raise RuntimeError("no CUDA device available for queue mode")

    pairs = _list_reside_pairs(limit=num_samples)
    output_root = _prepare_output_dir("promptir_attack")

    chunks: list[list[tuple[str, str, str]]] = [[] for _ in range(used_gpus)]
    for i, (cp, dp, sid) in enumerate(pairs):
        chunks[i % used_gpus].append((str(cp), str(dp), sid))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=used_gpus) as pool:
        async_results = []
        for gpu_idx in range(used_gpus):
            if len(chunks[gpu_idx]) == 0:
                continue
            async_results.append(
                pool.apply_async(
                    _run_worker_on_gpu,
                    kwds={
                        "gpu_index": gpu_idx,
                        "assigned_pairs": chunks[gpu_idx],
                        "output_root": str(output_root),
                        "checkpoint_path": str(checkpoint_path),
                        "steps1": steps1,
                        "steps2": steps2,
                        "step_size": step_size,
                        "lambda_reg": lambda_reg,
                        "max_side": max_side,
                        "seed": seed,
                        "save_per_sample": save_per_sample,
                    },
                )
            )

        merged: list[dict[str, Any]] = []
        for item in async_results:
            merged.extend(item.get())

    merged = sorted(merged, key=lambda x: x["sample_id"])
    summary = {
        "num_samples": len(merged),
        "num_gpus_requested": num_gpus,
        "num_gpus_used": used_gpus,
        "steps1": steps1,
        "steps2": steps2,
        "save_per_sample": bool(save_per_sample),
        "checkpoint": str(checkpoint_path),
        "avg_initial_task_loss": float(sum(x["initial_task_loss"] for x in merged) / max(1, len(merged))),
        "avg_final_task_loss": float(sum(x["final_task_loss"] for x in merged) / max(1, len(merged))),
        "avg_total_sample_sec": float(sum(x["timing"]["total_sample_sec"] for x in merged) / max(1, len(merged))),
        "avg_load_prepare_sec": float(sum(x["timing"]["load_prepare_sec"] for x in merged) / max(1, len(merged))),
        "avg_attack_and_optional_save_sec": float(sum(x["timing"]["attack_and_optional_save_sec"] for x in merged) / max(1, len(merged))),
        "samples": merged,
    }
    summary_path = output_root / "queue_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"queue_mode=enabled")
    print(f"num_samples={len(merged)}")
    print(f"num_gpus_requested={num_gpus}")
    print(f"num_gpus_used={used_gpus}")
    print(f"save_per_sample={bool(save_per_sample)}")
    print(f"avg_total_sample_sec={summary['avg_total_sample_sec']:.4f}")
    print(f"avg_load_prepare_sec={summary['avg_load_prepare_sec']:.4f}")
    print(f"avg_attack_and_optional_save_sec={summary['avg_attack_and_optional_save_sec']:.4f}")
    print(f"output_root={output_root}")
    print(f"summary_path={summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PromptIR single-image adversarial degradation search")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="PromptIR checkpoint path")
    parser.add_argument("--steps1", type=int, default=4, help="expensive PromptIR call count")
    parser.add_argument("--steps2", type=int, default=4, help="degradation updates per fixed PromptIR gradient")
    parser.add_argument("--num_steps", type=int, default=None, help="deprecated alias for steps1")
    parser.add_argument("--step_size", type=float, default=5e-2, help="optimizer learning rate for degradation params")
    parser.add_argument("--lambda_reg", type=float, default=0.05, help="regularization weight in attack objective")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="execution device")
    parser.add_argument("--num_gpus", type=int, default=2, help="number of GPUs to use when device=cuda")
    parser.add_argument("--num_samples", type=int, default=1, help="number of images to process in one run")
    parser.add_argument("--queue_mode", action="store_true", help="enable persistent multi-process GPU queue for multi-image runs")
    parser.add_argument(
        "--queue_save_samples",
        action="store_true",
        help="save per-sample images/logs in queue mode (disabled by default for better GPU utilization)",
    )
    parser.add_argument("--max_side", type=int, default=256, help="resize input long side to this value for memory-safe run")
    parser.add_argument("--seed", type=int, default=123, help="global random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    effective_steps1 = args.steps1 if args.num_steps is None else args.num_steps
    if args.queue_mode or args.num_samples > 1:
        run_promptir_multi_image_attack_queue(
            checkpoint_path=Path(args.ckpt),
            num_samples=args.num_samples,
            steps1=effective_steps1,
            steps2=args.steps2,
            step_size=args.step_size,
            lambda_reg=args.lambda_reg,
            num_gpus=args.num_gpus,
            max_side=args.max_side,
            seed=args.seed,
            save_per_sample=args.queue_save_samples,
        )
        raise SystemExit(0)

    run_promptir_single_image_attack(
        checkpoint_path=Path(args.ckpt),
        steps1=effective_steps1,
        steps2=args.steps2,
        step_size=args.step_size,
        lambda_reg=args.lambda_reg,
        device_name=args.device,
        num_gpus=args.num_gpus,
        max_side=args.max_side,
        seed=args.seed,
    )

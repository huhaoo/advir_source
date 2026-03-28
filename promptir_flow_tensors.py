from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from promptir_attack import (
    PROJECT_ROOT,
    _find_reside_pair,
    _forward_promptir_with_padding,
    _load_distance_from_mat,
    _load_image_rgb,
    _resize_to_max_side,
    _seed_everything,
    load_promptir_model,
)
from degradation import noise_rain_haze_degradation, random_degradation_configs_from_image

DEFAULT_FLOW_CKPT = PROJECT_ROOT / "PromptIR" / "train_ckpt_repro_8192" / "epoch=63-step=114688.ckpt"


def _prepare_output_dir(subdir: str = "promptir_flow") -> Path:
    output_dir = PROJECT_ROOT / "tmp_demo" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_mild_rain_haze_controller(image: Tensor) -> noise_rain_haze_degradation:
    noise_cfg, rain_cfg, haze_cfg = random_degradation_configs_from_image(image)
    rain_cfg.router_config.num_maps = 4
    haze_cfg.airlight_init = (0.88, 0.90, 0.92)
    haze_cfg.beta_mean = 0.12
    haze_cfg.beta_std = 0.02
    return noise_rain_haze_degradation(
        noise_config=noise_cfg,
        rain_config=rain_cfg,
        haze_config=haze_cfg,
        enable_noise=False,
        enable_rain=True,
        enable_haze=True,
        rain_haze_order="rain_haze",
    )


def _save_npy(path: Path, x: Tensor) -> None:
    arr = x.detach().cpu().numpy()
    np.save(path, arr)


def export_flow_tensors(
    checkpoint_path: Path,
    device_name: str,
    max_side: int,
    seed: int,
    output_subdir: str,
    update_lr: float,
    lambda_reg: float,
) -> dict[str, Any]:
    _seed_everything(seed)

    clear_path, depth_path, sample_id = _find_reside_pair()
    image = _load_image_rgb(clear_path)
    _, _, h, w = image.shape
    distance_map, depth_match_info = _load_distance_from_mat(depth_path, (h, w))
    image, distance_map = _resize_to_max_side(image, distance_map, max_side=max_side)
    gt = image.clone()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device_name='cuda' but CUDA is not available")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError(f"device_name must be 'cpu' or 'cuda', got {device_name}")

    device = torch.device(device_name)
    image = image.to(device)
    gt = gt.to(device)
    distance_map = distance_map.to(device)

    model = load_promptir_model(checkpoint_path, device=device)

    controller = build_mild_rain_haze_controller(image).to(device)
    controller.train()
    controller.reset_parameters()
    state = controller.get_current_state()

    rain_candidates: list[Tensor] | None = None
    if "rain" in state["current_enabled"]:
        rain_candidates = controller.rain_module.build_and_store_auto_candidates(image)
    rain_topk = min(2, controller.rain_module.num_branches)

    attack_params = [p for p in controller.parameters() if p.requires_grad]
    if len(attack_params) == 0:
        raise RuntimeError("attack controller has no trainable parameters")
    optimizer = torch.optim.Adam(attack_params, lr=update_lr)

    input1 = controller(
        image,
        distance_map=distance_map,
        rain_degraded_list=rain_candidates,
        rain_topk=rain_topk,
    ).requires_grad_(True)

    yhat1 = _forward_promptir_with_padding(model, input1)
    loss1 = F.l1_loss(yhat1, gt)
    grad1 = torch.autograd.grad(loss1, input1, retain_graph=False, create_graph=False)[0].detach()

    optimizer.zero_grad(set_to_none=True)
    update_input = controller(
        image,
        distance_map=distance_map,
        rain_degraded_list=rain_candidates,
        rain_topk=rain_topk,
    )
    update_yhat = _forward_promptir_with_padding(model, update_input)
    update_loss = F.l1_loss(update_yhat, gt)
    reg_loss = controller.get_regularization_loss()
    attack_obj = update_loss - lambda_reg * reg_loss
    (-attack_obj).backward()
    optimizer.step()

    input2 = controller(
        image,
        distance_map=distance_map,
        rain_degraded_list=rain_candidates,
        rain_topk=rain_topk,
    ).requires_grad_(True)
    yhat2 = _forward_promptir_with_padding(model, input2)
    loss2 = F.l1_loss(yhat2, gt)
    grad2 = torch.autograd.grad(loss2, input2, retain_graph=False, create_graph=False)[0].detach()

    output_dir = _prepare_output_dir(output_subdir)

    gt_path = output_dir / "gt.npy"
    input1_path = output_dir / "input1.npy"
    yhat1_path = output_dir / "yhat1.npy"
    grad1_path = output_dir / "grad1.npy"
    input2_path = output_dir / "input2.npy"
    grad2_path = output_dir / "grad2.npy"
    pack_path = output_dir / "flow_tensors.pt"

    _save_npy(gt_path, gt)
    _save_npy(input1_path, input1)
    _save_npy(yhat1_path, yhat1)
    _save_npy(grad1_path, grad1)
    _save_npy(input2_path, input2)
    _save_npy(grad2_path, grad2)

    torch.save(
        {
            "gt": gt.detach().cpu(),
            "input1": input1.detach().cpu(),
            "yhat1": yhat1.detach().cpu(),
            "grad1": grad1.detach().cpu(),
            "input2": input2.detach().cpu(),
            "grad2": grad2.detach().cpu(),
        },
        pack_path,
    )

    summary = {
        "sample_id": sample_id,
        "clear_path": str(clear_path),
        "depth_path": str(depth_path),
        "ckpt": str(checkpoint_path),
        "device": device_name,
        "depth_match_info": depth_match_info,
        "state": state,
        "rain_topk": rain_topk,
        "loss1": float(loss1.detach().item()),
        "loss2": float(loss2.detach().item()),
        "update_lr": float(update_lr),
        "lambda_reg": float(lambda_reg),
        "shape": tuple(gt.shape),
        "gt": str(gt_path),
        "input1": str(input1_path),
        "yhat1": str(yhat1_path),
        "grad1": str(grad1_path),
        "input2": str(input2_path),
        "grad2": str(grad2_path),
        "pack_pt": str(pack_path),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "flow_tensors_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export gt/input1/yhat1/grad1/input2/grad2 tensors for one attack update")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_FLOW_CKPT), help="PromptIR checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="execution device")
    parser.add_argument("--max_side", type=int, default=256, help="resize long side to control memory")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--output_subdir", type=str, default="promptir_flow", help="subdir under tmp_demo")
    parser.add_argument("--update_lr", type=float, default=5e-2, help="attacker one-step update learning rate")
    parser.add_argument("--lambda_reg", type=float, default=0.05, help="regularization coefficient for attacker update")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_flow_tensors(
        checkpoint_path=Path(args.ckpt),
        device_name=args.device,
        max_side=args.max_side,
        seed=args.seed,
        output_subdir=args.output_subdir,
        update_lr=args.update_lr,
        lambda_reg=args.lambda_reg,
    )

    print("export_flow_tensors=enabled")
    print(f"sample_id={summary['sample_id']}")
    print(f"ckpt={summary['ckpt']}")
    print(f"device={summary['device']}")
    print(f"shape={summary['shape']}")
    print(f"loss1={summary['loss1']:.10f}")
    print(f"loss2={summary['loss2']:.10f}")
    print(f"gt={summary['gt']}")
    print(f"input1={summary['input1']}")
    print(f"yhat1={summary['yhat1']}")
    print(f"grad1={summary['grad1']}")
    print(f"input2={summary['input2']}")
    print(f"grad2={summary['grad2']}")
    print(f"pack_pt={summary['pack_pt']}")
    print(f"summary_path={summary['summary_path']}")


if __name__ == "__main__":
    main()

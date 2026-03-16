from __future__ import annotations

from pathlib import Path
import shutil

import h5py
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from control_map import ControlMapConfig, ControlMapRouterConfig
from degradation import (
    HazeDegradationConfig,
    NoiseDegradationConfig,
    RainDegradationConfig,
    haze_degradation,
    noise_degradation,
    noise_rain_haze_degradation,
    rain_degradation,
)


def _prepare_demo_output_dir() -> Path:
    """Reset /tmp_demo before each demo run."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "tmp_demo"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_image_2d_or_3d(x: Tensor) -> Tensor:
    if x.ndim == 4: x = x[0]
    if x.ndim == 3 and x.shape[0] in {1, 3}: return x
    if x.ndim == 2: return x
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
    project_root = Path(__file__).resolve().parents[1]
    clear_dir = project_root / "dataset" / "haze" / "reside_ots" / "clear"
    depth_dir = project_root / "dataset" / "haze" / "reside_ots" / "depth"
    if not clear_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"reside_ots clear/depth dirs not found: {clear_dir}, {depth_dir}")

    clear_files = sorted(clear_dir.glob("*.jpg"))
    if len(clear_files) == 0:
        raise FileNotFoundError(f"no clear images found in {clear_dir}")

    for clear_path in clear_files:
        sample_id = clear_path.stem
        depth_path = depth_dir / f"{sample_id}.mat"
        if depth_path.exists(): return clear_path, depth_path, sample_id
    raise FileNotFoundError("no matched clear jpg and depth mat pair found in reside_ots")


def _load_clean_image_and_depth(clear_path: Path, depth_path: Path) -> tuple[Tensor, Tensor]:
    img_np = plt.imread(clear_path)
    if img_np.ndim != 3 or img_np.shape[2] not in {3, 4}:
        raise ValueError(f"clear image must be HxWx3/4, got shape {img_np.shape}")
    if img_np.shape[2] == 4: img_np = img_np[:, :, :3]

    image = torch.from_numpy(img_np.copy()).float()
    if image.max().item() > 1.0: image = image / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().clamp(0.0, 1.0)

    with h5py.File(depth_path, "r") as mat:
        if "depth" not in mat: raise KeyError(f"key 'depth' not found in {depth_path}")
        depth_np = mat["depth"][()]
    depth = torch.from_numpy(depth_np.copy()).float().unsqueeze(0).unsqueeze(0)

    _, _, h, w = image.shape
    if depth.shape[-2:] != (h, w):
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
    depth = depth.clamp_min(0.0)
    return image, depth


def _init_rain_router_patterns(module: rain_degradation) -> None:
    lh, lw = module.router.low_res_height, module.router.low_res_width
    ys = torch.linspace(-1.0, 1.0, lh, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, lw, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    keep_original = -2.8 * (xx.abs() + yy.abs())
    vertical_streak = 3.0 * torch.sin(xx * 10.0)
    diag_streak = 3.0 * torch.sin((xx + yy) * 8.0)
    patterns = [keep_original, vertical_streak, diag_streak]

    with torch.no_grad():
        for idx, p in enumerate(patterns):
            module.router.maps[idx].get_low_res_map().copy_(p.unsqueeze(0).unsqueeze(0))


def _make_rain_candidate_vertical(image: Tensor) -> Tensor:
    b, c, h, w = image.shape
    xs = torch.linspace(0.0, 1.0, w, dtype=image.dtype, device=image.device).view(1, 1, 1, w)
    streak = (torch.sin(xs * 140.0) > 0.82).to(image.dtype) * 0.55
    streak = streak.expand(b, c, h, w)
    return (image + streak).clamp(0.0, 1.0)


def _make_rain_candidate_diag(image: Tensor) -> Tensor:
    b, c, h, w = image.shape
    ys = torch.linspace(0.0, 1.0, h, dtype=image.dtype, device=image.device).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, dtype=image.dtype, device=image.device).view(1, 1, 1, w)
    streak = (torch.sin((xs + ys) * 85.0) > 0.86).to(image.dtype) * 0.5
    streak = streak.expand(b, c, h, w)
    return (image + streak).clamp(0.0, 1.0)


def _rgb_to_gray(image: Tensor) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be RGB tensor with shape (B, 3, H, W), got {tuple(image.shape)}")
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _sobel_edge_map(image: Tensor) -> Tensor:
    gray = _rgb_to_gray(image)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def run_degradation_showcase() -> None:
    """Run edge-driven 8-step gradient descent for noise/rain/haze degradations."""
    output_dir = _prepare_demo_output_dir()
    clear_path, depth_path, sample_id = _find_reside_pair()
    image, depth_map = _load_clean_image_and_depth(clear_path, depth_path)
    _, _, h, w = image.shape
    edge_target = _sobel_edge_map(image).detach()

    noise_module = noise_degradation(
        NoiseDegradationConfig(
            image_height=h,
            image_width=w,
            num_channels=3,
            noise_strength=0.6,
        )
    )
    noise_module.reset_parameters()
    noise_before = noise_module(image)
    noise_map_vis = ((noise_module.noise_map - noise_module.noise_map.min()) / (noise_module.noise_map.max() - noise_module.noise_map.min() + 1e-6))

    noise_strength_var = nn.Parameter(torch.tensor(float(noise_module.noise_strength), dtype=image.dtype))
    noise_optimizer = torch.optim.Adam([noise_strength_var], lr=0.25)
    noise_losses: list[float] = []
    for _ in range(8):
        noise_optimizer.zero_grad()
        noise_strength = noise_strength_var.clamp_min(0.0)
        noise_out = (image + noise_strength * noise_module.noise_map).clamp(0.0, 1.0)
        noise_edge = _sobel_edge_map(noise_out)
        noise_loss = (noise_edge * edge_target).mean()
        noise_loss.backward()
        noise_optimizer.step()
        noise_losses.append(float(noise_loss.detach().item()))
    noise_module.noise_strength = float(noise_strength_var.detach().clamp_min(0.0).item())
    noise_after = noise_module(image)

    rain_module = rain_degradation(
        RainDegradationConfig(
            router_config=ControlMapRouterConfig(
                num_maps=3,
                map_config=ControlMapConfig(
                    low_res_height=16,
                    low_res_width=16,
                    high_res_height=h,
                    high_res_width=w,
                    interp_mode="bicubic",
                    align_corners=False,
                    lambda_first_order=2e-2,
                    lambda_second_order=1e-2,
                    init_mode="zeros",
                    init_value=0.0,
                    init_scale=1.0,
                ),
                temperature=0.25,
            )
        )
    )
    _init_rain_router_patterns(rain_module)
    rain_candidate_1 = _make_rain_candidate_vertical(image)
    rain_candidate_2 = _make_rain_candidate_diag(image)
    rain_before = rain_module(image, [rain_candidate_1, rain_candidate_2], topk=2)

    rain_optimizer = torch.optim.Adam(rain_module.parameters(), lr=0.12)
    rain_losses: list[float] = []
    for _ in range(8):
        rain_optimizer.zero_grad()
        rain_out = rain_module(image, [rain_candidate_1, rain_candidate_2], topk=2)
        rain_edge = _sobel_edge_map(rain_out)
        rain_task = (rain_edge * edge_target).mean()
        rain_reg = 0.05 * rain_module.get_regularization_loss()
        rain_loss = rain_task + rain_reg
        rain_loss.backward()
        rain_optimizer.step()
        rain_losses.append(float(rain_loss.detach().item()))

    rain_after = rain_module(image, [rain_candidate_1, rain_candidate_2], topk=2)
    rain_weights = rain_module.get_weight_maps(topk=2)

    depth_min = depth_map.amin()
    depth_max = depth_map.amax()
    distance_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    distance_map = 0.1 + 3.2 * distance_map
    haze_module = haze_degradation(
        HazeDegradationConfig(
            map_config=ControlMapConfig(
                low_res_height=12,
                low_res_width=16,
                high_res_height=h,
                high_res_width=w,
                interp_mode="bicubic",
                align_corners=False,
                lambda_first_order=1e-2,
                lambda_second_order=1e-3,
                init_mode="normal",
                init_value=0.0,
                init_scale=0.6,
            ),
            airlight_init=(1.0, 0.95, 0.9),
            beta_mean=2.2,
            beta_std=1.1,
            min_beta=1e-3,
        )
    )
    haze_before = haze_module(image, distance_map)

    haze_optimizer = torch.optim.Adam(haze_module.parameters(), lr=0.08)
    haze_losses: list[float] = []
    for _ in range(8):
        haze_optimizer.zero_grad()
        haze_out = haze_module(image, distance_map)
        haze_edge = _sobel_edge_map(haze_out)
        haze_task = (haze_edge * edge_target).mean()
        haze_reg = 0.02 * haze_module.get_regularization_loss()
        haze_loss = haze_task + haze_reg
        haze_loss.backward()
        haze_optimizer.step()
        haze_losses.append(float(haze_loss.detach().item()))

    haze_after = haze_module(image, distance_map)
    bounded_airlight = haze_module.get_airlight().detach()
    base_beta = haze_module.beta_map_module()
    beta = ((base_beta - base_beta.mean()) / (base_beta.std(unbiased=False) + 1e-6) * haze_module.beta_std + haze_module.beta_mean).clamp_min(haze_module.min_beta)
    transmission = torch.exp(-beta * distance_map)

    files = {
        "original": output_dir / "scene_original.png",
        "edge_target": output_dir / "edge_target_map.png",
        "noise_before": output_dir / "noise_before_opt.png",
        "noise_after": output_dir / "noise_after_opt.png",
        "noise_map": output_dir / "noise_map_visual.png",
        "rain_candidate_1": output_dir / "rain_candidate_vertical.png",
        "rain_candidate_2": output_dir / "rain_candidate_diagonal.png",
        "rain_before": output_dir / "rain_before_opt.png",
        "rain_after": output_dir / "rain_after_opt.png",
        "rain_weight_0": output_dir / "rain_weight_branch0.png",
        "rain_weight_1": output_dir / "rain_weight_branch1.png",
        "rain_weight_2": output_dir / "rain_weight_branch2.png",
        "haze_distance": output_dir / "haze_distance_map.png",
        "haze_beta": output_dir / "haze_beta_map.png",
        "haze_transmission": output_dir / "haze_transmission_map.png",
        "haze_before": output_dir / "haze_before_opt.png",
        "haze_after": output_dir / "haze_after_opt.png",
        "stats": output_dir / "degradation_showcase_stats.txt",
    }

    _save_tensor_png(image, files["original"], "original_scene")
    _save_tensor_png(edge_target[0, 0], files["edge_target"], "edge_target_map")
    _save_tensor_png(noise_before, files["noise_before"], "noise_before_8step_opt")
    _save_tensor_png(noise_after, files["noise_after"], "noise_after_8step_opt")
    _save_tensor_png(noise_map_vis, files["noise_map"], "noise_map_visual")
    _save_tensor_png(rain_candidate_1, files["rain_candidate_1"], "rain_candidate_vertical")
    _save_tensor_png(rain_candidate_2, files["rain_candidate_2"], "rain_candidate_diagonal")
    _save_tensor_png(rain_before, files["rain_before"], "rain_before_8step_opt")
    _save_tensor_png(rain_after, files["rain_after"], "rain_after_8step_opt")
    _save_tensor_png(rain_weights[0, 0, 0], files["rain_weight_0"], "rain_weight_branch0")
    _save_tensor_png(rain_weights[0, 1, 0], files["rain_weight_1"], "rain_weight_branch1")
    _save_tensor_png(rain_weights[0, 2, 0], files["rain_weight_2"], "rain_weight_branch2")
    _save_tensor_png(distance_map[0, 0], files["haze_distance"], "haze_distance_map")
    _save_tensor_png(beta[0, 0], files["haze_beta"], "haze_beta_map")
    _save_tensor_png(transmission[0, 0], files["haze_transmission"], "haze_transmission")
    _save_tensor_png(haze_before, files["haze_before"], "haze_before_8step_opt")
    _save_tensor_png(haze_after, files["haze_after"], "haze_after_8step_opt")

    stats_lines = [
        "degradation_showcase",
        f"sample_id={sample_id}",
        f"gradient_steps=8",
        f"clear_path={clear_path}",
        f"depth_path={depth_path}",
        f"image_shape={tuple(image.shape)}",
        f"distance_min={distance_map.min().item():.6f}",
        f"distance_max={distance_map.max().item():.6f}",
        f"noise_strength={noise_module.noise_strength}",
        f"noise_losses={noise_losses}",
        f"rain_num_branches={rain_module.num_branches}",
        f"rain_topk=2",
        f"rain_losses={rain_losses}",
        f"haze_beta_mean={haze_module.beta_mean}",
        f"haze_beta_std={haze_module.beta_std}",
        f"haze_min_beta={haze_module.min_beta}",
        f"haze_airlight_actual_min={bounded_airlight.min().item():.6f}",
        f"haze_airlight_actual_max={bounded_airlight.max().item():.6f}",
        f"haze_losses={haze_losses}",
        *[f"{k}={v}" for k, v in files.items()],
    ]
    files["stats"].write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"generated_files={[str(v) for v in files.values()]}")


def run_noise_rain_haze_controller_demo() -> None:
    """Minimal demo for the combined noise_rain_haze_degradation controller."""
    output_dir = _prepare_demo_output_dir()
    clear_path, depth_path, sample_id = _find_reside_pair()
    image, depth_map = _load_clean_image_and_depth(clear_path, depth_path)
    _, _, h, w = image.shape

    depth_min = depth_map.amin()
    depth_max = depth_map.amax()
    distance_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    distance_map = 0.1 + 2.8 * distance_map

    noise_cfg = NoiseDegradationConfig(
        image_height=h,
        image_width=w,
        num_channels=3,
        noise_strength=0.35,
    )
    rain_cfg = RainDegradationConfig(
        router_config=ControlMapRouterConfig(
            num_maps=3,
            map_config=ControlMapConfig(
                low_res_height=16,
                low_res_width=16,
                high_res_height=h,
                high_res_width=w,
                interp_mode="bicubic",
                align_corners=False,
                lambda_first_order=0.02,
                lambda_second_order=0.01,
                init_mode="zeros",
                init_value=0.0,
                init_scale=1.0,
            ),
            temperature=0.35,
        ),
    )
    haze_cfg = HazeDegradationConfig(
        map_config=ControlMapConfig(
            low_res_height=12,
            low_res_width=16,
            high_res_height=h,
            high_res_width=w,
            interp_mode="bicubic",
            align_corners=False,
            lambda_first_order=0.01,
            lambda_second_order=0.005,
            init_mode="normal",
            init_value=0.0,
            init_scale=0.4,
        ),
        airlight_init=0.92,
        beta_mean=1.6,
        beta_std=0.5,
        min_beta=1e-3,
    )

    controller = noise_rain_haze_degradation(
        noise_config=noise_cfg,
        rain_config=rain_cfg,
        haze_config=haze_cfg,
        enable_noise=True,
        enable_rain=True,
        enable_haze=True,
    )
    controller.reset_parameters()

    output = controller(
        image=image,
        distance_map=distance_map,
        rain_degraded_list=None,
        rain_topk=2,
    )
    state = controller.get_current_state()
    reg_items = controller.get_regularization_items()
    auto_rain_params = list(controller.rain_module.last_auto_rain_params)
    rain_angles = [float(p["angle"]) for p in auto_rain_params]
    rain_angle_gap = (max(rain_angles) - min(rain_angles)) if len(rain_angles) >= 2 else 0.0
    rain_preview = controller.rain_module.add_rain(image, **auto_rain_params[0]) if len(auto_rain_params) > 0 else image

    output_png = output_dir / "controller_output.png"
    rain_png = output_dir / "controller_rain_auto_sample.png"
    stats_txt = output_dir / "controller_demo_stats.txt"
    _save_tensor_png(image, output_dir / "controller_input.png", "controller_input")
    _save_tensor_png(rain_preview, rain_png, "controller_rain_auto_sample")
    _save_tensor_png(output, output_png, "controller_output")

    lines = [
        "noise_rain_haze_controller_demo",
        f"sample_id={sample_id}",
        f"clear_path={clear_path}",
        f"depth_path={depth_path}",
        f"input_shape={tuple(image.shape)}",
        f"output_shape={tuple(output.shape)}",
        f"rain_auto_params={auto_rain_params}",
        f"rain_angle_gap={rain_angle_gap:.6f}",
        f"current_state={state}",
        f"regularization_items={{'loss_noise': {reg_items['loss_noise'].item():.10f}, 'loss_rain': {reg_items['loss_rain'].item():.10f}, 'loss_haze': {reg_items['loss_haze'].item():.10f}, 'loss_total': {reg_items['loss_total'].item():.10f}}}",
        f"rain_png={rain_png}",
        f"output_png={output_png}",
    ]
    stats_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"controller_state={state}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"rain_auto_params={auto_rain_params}")
    print(f"rain_angle_gap={rain_angle_gap:.6f}")
    print(
        "regularization_items="
        f"{{'loss_noise': {reg_items['loss_noise'].item():.10f}, "
        f"'loss_rain': {reg_items['loss_rain'].item():.10f}, "
        f"'loss_haze': {reg_items['loss_haze'].item():.10f}, "
        f"'loss_total': {reg_items['loss_total'].item():.10f}}}"
    )
    print(f"output_dir={output_dir}")
    print(f"generated_files={[str(output_dir / 'controller_input.png'), str(rain_png), str(output_png), str(stats_txt)]}")


if __name__ == "__main__":
    run_noise_rain_haze_controller_demo()

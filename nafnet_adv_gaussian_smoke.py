from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from degradation import random_degradation_configs_from_image
except ModuleNotFoundError:
    from .degradation import random_degradation_configs_from_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp_demo" / "nafnet_adv_gaussian_smoke"

def run_smoke(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    image = torch.rand(1, 3, 96, 128)

    _, default_rain_cfg, default_haze_cfg = random_degradation_configs_from_image(image)
    _, gaussian_rain_cfg, gaussian_haze_cfg = random_degradation_configs_from_image(
        image,
        interp_mode="gaussian",
        gaussian_radius=4,
        gaussian_sigma=1.25,
        gaussian_extra_cells=2,
    )

    summary = {
        "demo_name": "nafnet_adv_gaussian_smoke",
        "baseline_without_args": {
            "rain_interp_mode": default_rain_cfg.router_config.map_config.interp_mode,
            "haze_interp_mode": default_haze_cfg.map_config.interp_mode,
        },
        "with_gaussian_args": {
            "args": {
                "interp_mode": "gaussian",
                "gaussian_radius": 4,
                "gaussian_sigma": 1.25,
                "gaussian_extra_cells": 2,
            },
            "rain_interp_mode": gaussian_rain_cfg.router_config.map_config.interp_mode,
            "haze_interp_mode": gaussian_haze_cfg.map_config.interp_mode,
            "rain_gaussian_radius": gaussian_rain_cfg.router_config.map_config.gaussian_radius,
            "rain_gaussian_sigma": gaussian_rain_cfg.router_config.map_config.gaussian_sigma,
            "rain_gaussian_extra_cells": gaussian_rain_cfg.router_config.map_config.gaussian_extra_cells,
            "haze_gaussian_radius": gaussian_haze_cfg.map_config.gaussian_radius,
            "haze_gaussian_sigma": gaussian_haze_cfg.map_config.gaussian_sigma,
            "haze_gaussian_extra_cells": gaussian_haze_cfg.map_config.gaussian_extra_cells,
        },
    }

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke demo for gaussian interpolation settings in NAFNet adversarial config")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_smoke(output_root=args.output_root)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NAFNet evaluation on RESIDE-OTS test split with generated test config."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to NAFNet checkpoint (.pth). Required unless --dry_run is set.",
    )
    parser.add_argument(
        "--nafnet_root",
        type=str,
        default="/home/huhao/adv_ir/NAFNet",
        help="NAFNet repository root.",
    )
    parser.add_argument(
        "--template_opt",
        type=str,
        default="/home/huhao/adv_ir/NAFNet/options/test/RESIDE_OTS/NAFNet-width32-test.yml",
        help="Template test yaml file.",
    )
    parser.add_argument(
        "--test_lq_dir",
        type=str,
        default="/home/huhao/adv_ir/dataset/haze/reside_ots_nafnet/test/lq",
        help="LQ test directory for PairedImageDataset.",
    )
    parser.add_argument(
        "--test_gt_dir",
        type=str,
        default="/home/huhao/adv_ir/dataset/haze/reside_ots_nafnet/test/gt",
        help="GT test directory for PairedImageDataset.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="NAFNet-RESIDE-OTS-width32-test-runner",
        help="Experiment name used by BasicSR test output.",
    )
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument(
        "--launcher",
        type=str,
        default="none",
        choices=["none", "pytorch", "slurm"],
    )
    parser.add_argument(
        "--save_img",
        action="store_true",
        help="Save restored images during test.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/huhao/adv_ir/tmp_demo/nafnet_reside_ots_test_runner",
        help="Directory to save generated config and run summary.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only generate test yaml and summary without launching basicsr/test.py.",
    )
    return parser.parse_args()


def _must_exist(path_value: Path, kind: str) -> None:
    if not path_value.exists():
        raise FileNotFoundError(f"{kind} not found: {path_value}")


def _load_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"template yaml is not a dict: {yaml_path}")
    return obj


def _write_yaml(obj: dict, yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _make_summary(
    args: argparse.Namespace,
    generated_opt_path: Path,
    result_dir: Path,
    status: str,
    return_code: int,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "return_code": int(return_code),
        "dry_run": bool(args.dry_run),
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else "",
        "nafnet_root": str(Path(args.nafnet_root).resolve()),
        "template_opt": str(Path(args.template_opt).resolve()),
        "generated_opt": str(generated_opt_path.resolve()),
        "test_lq_dir": str(Path(args.test_lq_dir).resolve()),
        "test_gt_dir": str(Path(args.test_gt_dir).resolve()),
        "run_name": args.run_name,
        "num_gpu": int(args.num_gpu),
        "launcher": args.launcher,
        "save_img": bool(args.save_img),
        "result_dir": str(result_dir.resolve()),
        "result_dir_exists": bool(result_dir.exists()),
    }


def main() -> None:
    args = parse_args()

    if args.num_gpu < 0:
        raise ValueError(f"num_gpu must be >= 0, got {args.num_gpu}")

    nafnet_root = Path(args.nafnet_root)
    template_opt = Path(args.template_opt)
    test_lq_dir = Path(args.test_lq_dir)
    test_gt_dir = Path(args.test_gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _must_exist(nafnet_root, "nafnet_root")
    _must_exist(template_opt, "template_opt")
    _must_exist(test_lq_dir, "test_lq_dir")
    _must_exist(test_gt_dir, "test_gt_dir")

    model_path = Path(args.model_path) if args.model_path else Path()
    if not args.dry_run:
        if args.model_path.strip() == "":
            raise ValueError("model_path is required when dry_run is false")
        _must_exist(model_path, "model_path")

    opt = _load_yaml(template_opt)
    opt["name"] = args.run_name
    opt["num_gpu"] = int(args.num_gpu)
    opt.setdefault("datasets", {}).setdefault("test", {})
    opt["datasets"]["test"]["dataroot_lq"] = str(test_lq_dir)
    opt["datasets"]["test"]["dataroot_gt"] = str(test_gt_dir)
    opt.setdefault("path", {})
    opt["path"]["pretrain_network_g"] = str(model_path) if args.model_path else ""
    opt.setdefault("val", {})
    opt["val"]["save_img"] = bool(args.save_img)

    generated_opt_path = output_dir / "generated_test_opt.yml"
    _write_yaml(opt, generated_opt_path)

    result_dir = nafnet_root / "results" / args.run_name
    summary_path = output_dir / "run_summary.json"

    if args.dry_run:
        summary = _make_summary(
            args=args,
            generated_opt_path=generated_opt_path,
            result_dir=result_dir,
            status="dry_run_ready",
            return_code=0,
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    cmd = [
        sys.executable,
        "basicsr/test.py",
        "-opt",
        str(generated_opt_path),
        "--launcher",
        args.launcher,
    ]
    completed = subprocess.run(cmd, cwd=str(nafnet_root), check=False)

    status = "ok" if completed.returncode == 0 else "failed"
    summary = _make_summary(
        args=args,
        generated_opt_path=generated_opt_path,
        result_dir=result_dir,
        status=status,
        return_code=completed.returncode,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
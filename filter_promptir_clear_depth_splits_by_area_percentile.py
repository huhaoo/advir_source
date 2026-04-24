#!/usr/bin/env python3
"""Trim clear-image manifests by global image-area rank percentiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from PIL import Image


PROJECT_ROOT = Path("/home/huhao/adv_ir")
DEFAULT_SPLITS_JSON = PROJECT_ROOT / "dataset_path" / "promptir_clear_depth_splits.json"
DEFAULT_SETS_ONLY_JSON = PROJECT_ROOT / "dataset_path" / "promptir_clear_depth_sets_only.json"
DEFAULT_STATS_JSON = PROJECT_ROOT / "dataset_path" / "promptir_clear_depth_splits_stats.json"
DEFAULT_DEMO_DIR = PROJECT_ROOT / "tmp_demo" / "dataset_path_trim_by_area"


@dataclass(frozen=True)
class ClearImageInfo:
    split: str
    clear_path: str
    area: int


def _read_area(clear_path: str) -> int:
    with Image.open(clear_path) as image:
        width, height = image.size
    return int(width) * int(height)


def _build_clear_infos(splits_obj: Dict) -> List[ClearImageInfo]:
    infos: List[ClearImageInfo] = []
    for split in ("train", "val", "test"):
        for clear_path in splits_obj["splits"][split]["all_clear_paths"]:
            infos.append(
                ClearImageInfo(split=split, clear_path=clear_path, area=_read_area(clear_path))
            )
    return infos


def _compute_drop_sets(
    infos: Sequence[ClearImageInfo], trim_ratio_each_side: float
) -> Tuple[Set[str], Set[str], List[ClearImageInfo]]:
    sorted_infos = sorted(infos, key=lambda item: (item.area, item.clear_path))
    trim_count_each_side = int(len(sorted_infos) * trim_ratio_each_side)
    low_drop = {item.clear_path for item in sorted_infos[:trim_count_each_side]}
    high_drop = {item.clear_path for item in sorted_infos[-trim_count_each_side:]}
    return low_drop, high_drop, sorted_infos


def _filter_split(split_obj: Dict, keep_clear_paths: Set[str]) -> Dict:
    by_task = split_obj["by_task"]

    noisy_clear_paths = [
        path for path in by_task["noisy"]["clear_paths"] if path in keep_clear_paths
    ]

    rainy_input_paths: List[str] = []
    rainy_clear_paths: List[str] = []
    for input_path, clear_path in zip(
        by_task["rainy"]["input_paths"], by_task["rainy"]["clear_paths"]
    ):
        if clear_path in keep_clear_paths:
            rainy_input_paths.append(input_path)
            rainy_clear_paths.append(clear_path)

    hazy_input_paths: List[str] = []
    hazy_clear_paths: List[str] = []
    hazy_depth_paths: List[str] = []
    for hazy_path, clear_path, depth_path in zip(
        by_task["hazy"]["hazy_paths"],
        by_task["hazy"]["clear_paths"],
        by_task["hazy"]["depth_paths"],
    ):
        if clear_path in keep_clear_paths:
            hazy_input_paths.append(hazy_path)
            hazy_clear_paths.append(clear_path)
            hazy_depth_paths.append(depth_path)

    all_clear_paths = [path for path in split_obj["all_clear_paths"] if path in keep_clear_paths]
    all_depth_paths = list(hazy_depth_paths)

    missing_paths = {
        "noisy_clear_missing": [path for path in noisy_clear_paths if not Path(path).exists()],
        "rainy_input_missing": [path for path in rainy_input_paths if not Path(path).exists()],
        "rainy_clear_missing": [path for path in rainy_clear_paths if not Path(path).exists()],
        "hazy_input_missing": [path for path in hazy_input_paths if not Path(path).exists()],
        "hazy_clear_missing": [path for path in hazy_clear_paths if not Path(path).exists()],
        "hazy_depth_missing": [path for path in hazy_depth_paths if not Path(path).exists()],
    }

    counts = {
        "all_clear": len(all_clear_paths),
        "all_depth": len(all_depth_paths),
        "noisy_clear": len(noisy_clear_paths),
        "rainy_input": len(rainy_input_paths),
        "rainy_clear": len(rainy_clear_paths),
        "hazy_input": len(hazy_input_paths),
        "hazy_clear": len(hazy_clear_paths),
        "hazy_depth": len(hazy_depth_paths),
        "missing_noisy_clear": len(missing_paths["noisy_clear_missing"]),
        "missing_rainy_input": len(missing_paths["rainy_input_missing"]),
        "missing_rainy_clear": len(missing_paths["rainy_clear_missing"]),
        "missing_hazy_input": len(missing_paths["hazy_input_missing"]),
        "missing_hazy_clear": len(missing_paths["hazy_clear_missing"]),
        "missing_hazy_depth": len(missing_paths["hazy_depth_missing"]),
    }

    return {
        "all_clear_paths": all_clear_paths,
        "all_depth_paths": all_depth_paths,
        "by_task": {
            "noisy": {"clear_paths": noisy_clear_paths},
            "rainy": {"input_paths": rainy_input_paths, "clear_paths": rainy_clear_paths},
            "hazy": {
                "hazy_paths": hazy_input_paths,
                "clear_paths": hazy_clear_paths,
                "depth_paths": hazy_depth_paths,
            },
        },
        "missing_paths": missing_paths,
        "counts": counts,
    }


def _build_stats_obj(meta: Dict, filtered_splits: Dict) -> Dict:
    stats_splits = {}
    for split in ("train", "val", "test"):
        split_obj = filtered_splits[split]
        stats_splits[split] = {
            "counts": split_obj["counts"],
            "examples": {
                "all_clear_first5": split_obj["all_clear_paths"][:5],
                "all_depth_first5": split_obj["all_depth_paths"][:5],
                "hazy_input_first5": split_obj["by_task"]["hazy"]["hazy_paths"][:5],
            },
        }
    return {"meta": meta, "splits": stats_splits}


def _build_sets_only_obj(meta: Dict, filtered_splits: Dict) -> Dict:
    return {
        "meta": meta,
        "splits": {
            split: {
                "clear_paths": filtered_splits[split]["all_clear_paths"],
                "depth_paths": filtered_splits[split]["all_depth_paths"],
            }
            for split in ("train", "val", "test")
        },
    }


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trim promptir_clear_depth manifests by global image-area rank, "
            "dropping low/high tails equally."
        )
    )
    parser.add_argument("--input_splits_json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--output_splits_json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--output_sets_only_json", type=Path, default=DEFAULT_SETS_ONLY_JSON)
    parser.add_argument("--output_stats_json", type=Path, default=DEFAULT_STATS_JSON)
    parser.add_argument("--trim_ratio_each_side", type=float, default=0.05)
    parser.add_argument("--demo_dir", type=Path, default=DEFAULT_DEMO_DIR)
    args = parser.parse_args()

    if not (0.0 <= args.trim_ratio_each_side < 0.5):
        raise ValueError("trim_ratio_each_side must be in [0.0, 0.5).")

    splits_obj = json.loads(args.input_splits_json.read_text(encoding="utf-8"))
    clear_infos = _build_clear_infos(splits_obj)

    all_clear_paths = [item.clear_path for item in clear_infos]
    if len(all_clear_paths) != len(set(all_clear_paths)):
        raise ValueError("Duplicate clear paths found; expected deduplicated all_clear_paths.")

    low_drop, high_drop, sorted_infos = _compute_drop_sets(
        infos=clear_infos, trim_ratio_each_side=args.trim_ratio_each_side
    )
    drop_clear_paths = low_drop | high_drop
    keep_clear_paths = set(all_clear_paths) - drop_clear_paths

    filtered_splits = {}
    for split in ("train", "val", "test"):
        filtered_splits[split] = _filter_split(splits_obj["splits"][split], keep_clear_paths)

    meta = dict(splits_obj["meta"])
    notes = list(meta.get("notes", []))
    notes.append(
        "all_clear_paths were globally trimmed by image area rank: "
        f"drop lowest {args.trim_ratio_each_side:.2%} and highest {args.trim_ratio_each_side:.2%}"
    )
    meta["notes"] = notes

    output_splits_obj = {"meta": meta, "splits": filtered_splits}
    output_sets_only_obj = _build_sets_only_obj(meta=meta, filtered_splits=filtered_splits)
    output_stats_obj = _build_stats_obj(meta=meta, filtered_splits=filtered_splits)

    _write_json(args.output_splits_json, output_splits_obj)
    _write_json(args.output_sets_only_json, output_sets_only_obj)
    _write_json(args.output_stats_json, output_stats_obj)

    sorted_areas = [item.area for item in sorted_infos]
    trim_count_each_side = int(len(sorted_infos) * args.trim_ratio_each_side)
    kept_infos = [item for item in sorted_infos if item.clear_path in keep_clear_paths]
    kept_areas = [item.area for item in kept_infos]

    args.demo_dir.mkdir(parents=True, exist_ok=True)
    demo_summary = {
        "input_splits_json": str(args.input_splits_json),
        "output_splits_json": str(args.output_splits_json),
        "output_sets_only_json": str(args.output_sets_only_json),
        "output_stats_json": str(args.output_stats_json),
        "trim_ratio_each_side": args.trim_ratio_each_side,
        "num_total_clear": len(sorted_infos),
        "trim_count_each_side": trim_count_each_side,
        "num_removed_total": len(drop_clear_paths),
        "num_kept_total": len(keep_clear_paths),
        "removed_area_low_max": max(
            [item.area for item in sorted_infos if item.clear_path in low_drop], default=None
        ),
        "removed_area_high_min": min(
            [item.area for item in sorted_infos if item.clear_path in high_drop], default=None
        ),
        "kept_area_min": min(kept_areas) if kept_areas else None,
        "kept_area_max": max(kept_areas) if kept_areas else None,
        "all_area_min": min(sorted_areas) if sorted_areas else None,
        "all_area_max": max(sorted_areas) if sorted_areas else None,
        "split_counts": {
            split: filtered_splits[split]["counts"] for split in ("train", "val", "test")
        },
    }
    _write_json(args.demo_dir / "summary.json", demo_summary)

    print(json.dumps(demo_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

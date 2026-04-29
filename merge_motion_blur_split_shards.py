from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_split_shards(
    split_root: Path,
    manifest_glob: str,
    summary_glob: str,
    output_manifest: str,
    output_summary: str,
) -> dict[str, object]:
    if not split_root.exists():
        raise FileNotFoundError(f"split_root not found: {split_root}")

    manifest_paths = sorted(split_root.glob(manifest_glob))
    summary_paths = sorted(split_root.glob(summary_glob))
    if len(manifest_paths) == 0:
        raise RuntimeError(f"no shard manifests matched: {split_root / manifest_glob}")
    if len(summary_paths) == 0:
        raise RuntimeError(f"no shard summaries matched: {split_root / summary_glob}")

    merged_records: list[dict[str, object]] = []
    used_ids: set[str] = set()

    manifest_objs = [_load_json(p) for p in manifest_paths]
    summary_objs = [_load_json(p) for p in summary_paths]

    for mp, manifest_obj in zip(manifest_paths, manifest_objs):
        records = manifest_obj.get("records", [])
        if not isinstance(records, list):
            raise TypeError(f"records is not list in {mp}")
        for rec in records:
            image_id = str(rec.get("id", ""))
            if image_id in used_ids:
                raise RuntimeError(f"duplicate id during merge: {image_id}")
            used_ids.add(image_id)
            merged_records.append(rec)

    merged_records.sort(key=lambda x: str(x.get("id", "")))

    base_manifest = dict(manifest_objs[0])
    base_manifest["count"] = int(len(merged_records))
    base_manifest["records"] = merged_records
    base_manifest["merged_from_manifests"] = [str(p) for p in manifest_paths]

    base_summary = dict(summary_objs[0])
    base_summary["count"] = int(len(merged_records))
    base_summary["merged_from_summaries"] = [str(p) for p in summary_paths]

    manifest_out = split_root / output_manifest
    summary_out = split_root / output_summary
    manifest_out.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    summary_out.write_text(json.dumps(base_summary, indent=2), encoding="utf-8")

    result = {
        "split_root": str(split_root),
        "manifest_count": len(manifest_paths),
        "summary_count": len(summary_paths),
        "merged_count": int(len(merged_records)),
        "output_manifest": str(manifest_out),
        "output_summary": str(summary_out),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard manifest/summary files for one split.")
    parser.add_argument("--split_root", type=str, required=True, help="Split directory root, e.g. .../train")
    parser.add_argument("--manifest_glob", type=str, default="manifest_shard*.json")
    parser.add_argument("--summary_glob", type=str, default="summary_shard*.json")
    parser.add_argument("--output_manifest", type=str, default="manifest.json")
    parser.add_argument("--output_summary", type=str, default="summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_split_shards(
        split_root=Path(args.split_root),
        manifest_glob=str(args.manifest_glob),
        summary_glob=str(args.summary_glob),
        output_manifest=str(args.output_manifest),
        output_summary=str(args.output_summary),
    )


if __name__ == "__main__":
    main()

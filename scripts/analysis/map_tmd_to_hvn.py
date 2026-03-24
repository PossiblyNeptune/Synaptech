#!/usr/bin/env python3
"""
Map TMD target labels to project classes: human / vehicle / noise.

This script helps answer whether the dataset can be merged into your
seismic intrusion taxonomy and where label gaps remain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_MAP = {
    # Human-like classes
    "walking": "human",
    "walk": "human",
    "run": "human",
    "running": "human",
    "on_foot": "human",

    # Vehicle-like classes
    "bus": "vehicle",
    "car": "vehicle",
    "taxi": "vehicle",
    "tram": "vehicle",
    "metro": "vehicle",
    "subway": "vehicle",
    "motorcycle": "vehicle",
    "bike": "vehicle",  # optional, keep or remove based on your scope

    # Noise/other
    "still": "noise",
    "stationary": "noise",
    "idle": "noise",
    "unknown": "noise",
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def load_mapping(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_MAP
    data = json.loads(path.read_text(encoding="utf-8"))
    # Normalize keys to make matching robust
    return {normalize_label(k): v for k, v in data.items()}


def map_label(raw: str, mapping: dict) -> str:
    norm = normalize_label(raw)
    return mapping.get(norm, "unmapped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map TMD labels to human/vehicle/noise")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument("--mapping", default=None, help="Optional JSON mapping file")
    parser.add_argument("--out", default="reports/tmd_hvn_mapping.json", help="Output report path")
    args = parser.parse_args()

    mapping = load_mapping(Path(args.mapping) if args.mapping else None)

    label_counts = Counter()
    hvn_counts = Counter()
    unmapped_examples = Counter()

    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            raw = str(obj.get("target", "<missing>")).strip()
            label_counts[raw] += 1

            cls = map_label(raw, mapping)
            hvn_counts[cls] += 1
            if cls == "unmapped":
                unmapped_examples[raw] += 1

    out = {
        "raw_target_counts": dict(label_counts.most_common()),
        "mapped_hvn_counts": dict(hvn_counts),
        "unmapped_targets": dict(unmapped_examples.most_common()),
        "mapping_size": len(mapping),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print("Mapped counts:", dict(hvn_counts))
    if unmapped_examples:
        print("Unmapped targets (top):", dict(unmapped_examples.most_common(10)))


if __name__ == "__main__":
    main()

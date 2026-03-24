#!/usr/bin/env python3
"""
Quick quality/usefulness audit for TMD-style JSONL datasets.

- Reads newline-delimited JSON objects (one record per line)
- Computes parse quality, class balance, missingness, and simple anomaly checks
- Writes a JSON summary and optional Markdown report

No external dependencies required.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def add(self, x: float) -> None:
        self.count += 1
        self.total += x
        self.total_sq += x * x
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    def std(self) -> float | None:
        if self.count < 2:
            return None
        mu = self.total / self.count
        var = (self.total_sq / self.count) - (mu * mu)
        if var < 0:
            var = 0.0
        return math.sqrt(var)


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield line_num, {"__PARSE_ERROR__": raw}
                continue
            if isinstance(obj, dict):
                yield line_num, obj
            else:
                yield line_num, {"__NON_DICT__": str(type(obj))}


def analyze(path: Path, max_bad_examples: int = 20) -> dict:
    total_records = 0
    parse_errors = 0
    non_dict_records = 0

    bad_examples = []

    field_present = Counter()
    field_empty = Counter()
    field_non_numeric = Counter()
    numeric_stats: Dict[str, RunningStats] = defaultdict(RunningStats)

    target_counts = Counter()
    user_counts = Counter()

    anomaly_counts = Counter()

    for line_num, obj in parse_jsonl(path):
        if "__PARSE_ERROR__" in obj:
            parse_errors += 1
            if len(bad_examples) < max_bad_examples:
                bad_examples.append({"line": line_num, "type": "json_decode_error"})
            continue
        if "__NON_DICT__" in obj:
            non_dict_records += 1
            if len(bad_examples) < max_bad_examples:
                bad_examples.append({"line": line_num, "type": "non_dict"})
            continue

        total_records += 1

        tgt = str(obj.get("target", "<missing>")).strip()
        target_counts[tgt if tgt else "<empty>"] += 1

        usr = str(obj.get("user", "<missing>")).strip()
        user_counts[usr if usr else "<empty>"] += 1

        for k, v in obj.items():
            field_present[k] += 1

            if v is None or (isinstance(v, str) and v.strip() == ""):
                field_empty[k] += 1
                continue

            x = to_float(v)
            if x is None:
                field_non_numeric[k] += 1
                continue

            numeric_stats[k].add(x)

            # Domain sanity checks (simple but useful):
            if "pressure#" in k and not (850.0 <= x <= 1150.0):
                anomaly_counts["pressure_out_of_range"] += 1
            if k.endswith("gravity#mean") and not (8.0 <= x <= 11.5):
                anomaly_counts["gravity_mean_out_of_range"] += 1
            if "accelerometer#" in k and abs(x) > 200.0:
                anomaly_counts["accelerometer_extreme"] += 1
            if "gyroscope#" in k and abs(x) > 1000.0:
                anomaly_counts["gyroscope_extreme"] += 1

    parseable_lines = total_records + parse_errors + non_dict_records
    parse_success_rate = (total_records / parseable_lines) if parseable_lines else 0.0

    always_empty = []
    mostly_empty = []
    for k in field_present:
        ratio = field_empty[k] / field_present[k]
        if ratio == 1.0:
            always_empty.append(k)
        elif ratio >= 0.95:
            mostly_empty.append((k, ratio))

    sorted_targets = target_counts.most_common()
    class_imbalance_ratio = None
    if sorted_targets:
        counts = [c for _, c in sorted_targets if c > 0]
        if counts:
            class_imbalance_ratio = max(counts) / min(counts)

    # Simple usefulness score for tri-class (human/vehicle/noise) setup.
    score = 0
    if total_records >= 50000:
        score += 3
    elif total_records >= 10000:
        score += 2
    elif total_records >= 3000:
        score += 1

    if len([k for k in target_counts if not k.startswith("<")]) >= 3:
        score += 3
    elif len([k for k in target_counts if not k.startswith("<")]) >= 2:
        score += 1

    if parse_success_rate >= 0.995:
        score += 1

    if anomaly_counts["pressure_out_of_range"] > 0:
        score -= 1

    verdict = "weak"
    if score >= 6:
        verdict = "strong"
    elif score >= 3:
        verdict = "moderate"

    key_numeric_preview = {}
    for k in sorted(numeric_stats.keys())[:40]:
        s = numeric_stats[k]
        key_numeric_preview[k] = {
            "count": s.count,
            "min": s.min_val,
            "max": s.max_val,
            "mean": s.mean(),
            "std": s.std(),
        }

    summary = {
        "file": str(path),
        "total_records": total_records,
        "parse_errors": parse_errors,
        "non_dict_records": non_dict_records,
        "parse_success_rate": parse_success_rate,
        "num_unique_fields": len(field_present),
        "target_counts": dict(sorted_targets),
        "num_unique_targets": len(target_counts),
        "user_counts_top10": dict(user_counts.most_common(10)),
        "num_unique_users": len(user_counts),
        "class_imbalance_ratio": class_imbalance_ratio,
        "always_empty_fields": sorted(always_empty),
        "mostly_empty_fields": [{"field": k, "empty_ratio": r} for k, r in sorted(mostly_empty, key=lambda x: x[1], reverse=True)],
        "anomaly_counts": dict(anomaly_counts),
        "usefulness_score": score,
        "usefulness_verdict": verdict,
        "bad_examples": bad_examples,
        "numeric_preview_first_40_fields": key_numeric_preview,
    }

    return summary


def build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# TMD Dataset Audit")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Records: {summary['total_records']:,}")
    lines.append(f"- Parse success: {summary['parse_success_rate']:.4%}")
    lines.append(f"- Unique fields: {summary['num_unique_fields']}")
    lines.append(f"- Unique targets: {summary['num_unique_targets']}")
    lines.append(f"- Unique users: {summary['num_unique_users']}")
    lines.append(f"- Usefulness verdict: **{summary['usefulness_verdict']}** (score={summary['usefulness_score']})")
    lines.append("")

    lines.append("## Target Distribution")
    if summary["target_counts"]:
        for k, v in summary["target_counts"].items():
            lines.append(f"- {k}: {v:,}")
    else:
        lines.append("- No target labels found")
    lines.append("")

    lines.append("## Class Balance")
    ratio = summary["class_imbalance_ratio"]
    if ratio is None:
        lines.append("- Not enough labeled classes to compute imbalance")
    else:
        lines.append(f"- Max/min class ratio: {ratio:.2f}")
    lines.append("")

    lines.append("## Missingness")
    always_empty = summary["always_empty_fields"]
    if always_empty:
        lines.append(f"- Always empty fields ({len(always_empty)}):")
        for f in always_empty[:30]:
            lines.append(f"  - {f}")
        if len(always_empty) > 30:
            lines.append(f"  - ... +{len(always_empty) - 30} more")
    else:
        lines.append("- No always-empty fields")

    mostly_empty = summary["mostly_empty_fields"]
    if mostly_empty:
        lines.append("- Mostly empty fields (>=95% empty):")
        for item in mostly_empty[:20]:
            lines.append(f"  - {item['field']}: {item['empty_ratio']:.1%}")
    lines.append("")

    lines.append("## Anomalies")
    if summary["anomaly_counts"]:
        for k, v in summary["anomaly_counts"].items():
            lines.append(f"- {k}: {v:,}")
    else:
        lines.append("- No rule-based anomalies detected")
    lines.append("")

    lines.append("## Is It Useful For Human/Vehicle/Noise?")
    if summary["num_unique_targets"] < 2:
        lines.append("- Weak for direct supervised training: only one or missing class labels")
    else:
        lines.append("- Potentially useful as a vehicle/human context source, but verify target taxonomy mapping")
    if "Bus" in summary["target_counts"]:
        lines.append("- Contains vehicle-type labels (e.g., Bus) which is useful for the vehicle class")
    if any(f.startswith("sound#") for f in summary["always_empty_fields"]):
        lines.append("- Lacks sound features despite schema fields (sound columns empty)")
    if any(f.startswith("speed#") for f in summary["always_empty_fields"]):
        lines.append("- Lacks speed features despite schema fields (speed columns empty)")
    if summary["anomaly_counts"].get("pressure_out_of_range", 0) > 0:
        lines.append("- Has likely numeric formatting issues in pressure fields (e.g., missing decimal)")

    lines.append("")
    lines.append("## Next Actions")
    lines.append("- Remove or fix malformed numeric outliers")
    lines.append("- Keep only informative sensors for edge deployment (accelerometer/gyro/linear_acceleration)")
    lines.append("- Map targets to your 3 classes: human / vehicle / noise")
    lines.append("- Re-check class balance after mapping")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TMD JSONL dataset quality and usefulness")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--out-json", default="reports/tmd_audit_summary.json", help="Output JSON summary path")
    parser.add_argument("--out-md", default="reports/tmd_audit_summary.md", help="Output Markdown report path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    summary = analyze(in_path)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"Wrote JSON summary: {out_json}")
    print(f"Wrote Markdown report: {out_md}")
    print(f"Records: {summary['total_records']:,}")
    print(f"Targets: {summary['num_unique_targets']}")
    print(f"Verdict: {summary['usefulness_verdict']} (score={summary['usefulness_score']})")


if __name__ == "__main__":
    main()

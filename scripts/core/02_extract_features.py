from __future__ import annotations

"""
Stage 2: Extract compact FPGA-friendly features from canonical records.

Input:
- data/interim/all_sources_canonical.csv

Output:
- data/processed/features_all.csv

Feature set target (10-20 features):
- dominant frequency
- band energies
- spectral entropy
- variance / RMS / crest factor
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "android.sensor.accelerometer#mean",
    "android.sensor.accelerometer#min",
    "android.sensor.accelerometer#max",
    "android.sensor.accelerometer#std",
    "android.sensor.linear_acceleration#mean",
    "android.sensor.linear_acceleration#min",
    "android.sensor.linear_acceleration#max",
    "android.sensor.linear_acceleration#std",
    "android.sensor.gyroscope#mean",
    "android.sensor.gyroscope#min",
    "android.sensor.gyroscope#max",
    "android.sensor.gyroscope#std",
    "android.sensor.gyroscope_uncalibrated#mean",
    "android.sensor.gyroscope_uncalibrated#min",
    "android.sensor.gyroscope_uncalibrated#max",
    "android.sensor.gyroscope_uncalibrated#std",
]

ALLOWED_SOURCES = {"tmd", "uci_har", "bridge"}


def robust_scale_per_source(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    For each source and each feature:
    1) clip to [1st, 99th] percentile
    2) robust-scale using (x - median) / IQR
    """
    scaled = df.copy()
    stats = {}

    for src, idx in scaled.groupby("source").groups.items():
        src_stats = {}
        part = scaled.loc[idx, cols].copy()

        for c in cols:
            s = pd.to_numeric(part[c], errors="coerce")
            q01 = float(s.quantile(0.01))
            q99 = float(s.quantile(0.99))
            med = float(s.median())
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0

            s_clip = s.clip(lower=q01, upper=q99)
            s_scaled = (s_clip - med) / iqr
            scaled.loc[idx, c] = s_scaled

            src_stats[c] = {
                "q01": q01,
                "q99": q99,
                "median": med,
                "iqr": iqr,
            }

        stats[src] = src_stats

    return scaled, stats


def add_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    acc_mean = out["android.sensor.accelerometer#mean"]
    acc_std = out["android.sensor.accelerometer#std"]
    acc_min = out["android.sensor.accelerometer#min"]
    acc_max = out["android.sensor.accelerometer#max"]

    lin_mean = out["android.sensor.linear_acceleration#mean"]
    lin_std = out["android.sensor.linear_acceleration#std"]
    lin_min = out["android.sensor.linear_acceleration#min"]
    lin_max = out["android.sensor.linear_acceleration#max"]

    gyr_mean = out["android.sensor.gyroscope#mean"]
    gyr_std = out["android.sensor.gyroscope#std"]
    gyr_min = out["android.sensor.gyroscope#min"]
    gyr_max = out["android.sensor.gyroscope#max"]

    # Compact set (14) from available canonical summary columns.
    out_feat = pd.DataFrame(
        {
            "source": out["source"],
            "sample_id": out["sample_id"],
            "user": out["user"],
            "raw_target": out["raw_target"],
            "target": out["target"],
            "f_acc_mean": acc_mean,
            "f_acc_std": acc_std,
            "f_acc_range": acc_max - acc_min,
            "f_acc_energy_proxy": acc_mean * acc_mean + acc_std * acc_std,
            "f_lin_mean": lin_mean,
            "f_lin_std": lin_std,
            "f_lin_range": lin_max - lin_min,
            "f_lin_energy_proxy": lin_mean * lin_mean + lin_std * lin_std,
            "f_gyr_mean": gyr_mean,
            "f_gyr_std": gyr_std,
            "f_gyr_range": gyr_max - gyr_min,
            "f_gyr_energy_proxy": gyr_mean * gyr_mean + gyr_std * gyr_std,
            "f_motion_ratio": (lin_std.abs() + 1e-6) / (acc_std.abs() + 1e-6),
            "f_gyro_vs_linear": (gyr_std.abs() + 1e-6) / (lin_std.abs() + 1e-6),
        }
    )

    return out_feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact feature set")
    parser.add_argument("--input", default="data/interim/all_sources_canonical.csv")
    parser.add_argument("--output", default="data/processed/features_all.csv")
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument(
        "--include-sources",
        nargs="*",
        default=["tmd", "uci_har", "bridge"],
        help="Sources to include in feature extraction",
    )
    parser.add_argument(
        "--stats-out",
        default="data/processed/feature_scaler_stats.json",
        help="Where to save per-source robust-scaling stats",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    stats_path = Path(args.stats_out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)
    for c in BASE_FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing required canonical column: {c}")

    include = set(args.include_sources) & ALLOWED_SOURCES
    if not include:
        raise ValueError(f"No valid sources selected. Allowed sources: {sorted(ALLOWED_SOURCES)}")

    df = df[df["source"].isin(include)].copy()
    if df.empty:
        raise ValueError("No rows left after source filtering")

    # Ensure numeric typing for base columns.
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing base features.
    before = len(df)
    df = df.dropna(subset=BASE_FEATURES)
    dropped = before - len(df)

    scaled, scaler_stats = robust_scale_per_source(df, BASE_FEATURES)
    feat_df = add_compact_features(scaled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")

    print(f"Feature extraction complete: {out_path}")
    print(f"Rows kept={len(feat_df)} dropped_missing={dropped}")
    print(f"Sources used={sorted(include)}")
    print(f"Scaler stats saved: {stats_path}")


if __name__ == "__main__":
    main()

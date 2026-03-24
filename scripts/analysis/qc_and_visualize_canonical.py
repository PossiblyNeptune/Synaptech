from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="QC + visualization for canonical dataset")
    parser.add_argument("--input", default="data/interim/all_sources_canonical.csv")
    parser.add_argument("--out-dir", default="docs/reports")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["source", "sample_id", "user", "raw_target", "target", "time_sec"]
    ]

    # Force numeric conversion for feature diagnostics.
    num = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 1) Class distribution
    cls = df["target"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    cls.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    plt.close()

    # 2) Source x Class heatmap (as table plot)
    pivot = pd.crosstab(df["source"], df["target"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=pivot.values,
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.2)
    plt.title("Source vs Class Counts", pad=10)
    plt.tight_layout()
    plt.savefig(out / "source_class_counts.png", dpi=150)
    plt.close()

    # 3) Boxplots for key features by source to detect domain/outliers
    key_feats = [
        "android.sensor.accelerometer#mean",
        "android.sensor.accelerometer#std",
        "android.sensor.linear_acceleration#mean",
        "android.sensor.gyroscope#mean",
    ]

    for feat in key_feats:
        if feat not in num.columns:
            continue
        d = pd.DataFrame({"source": df["source"], feat: num[feat]})
        d = d.dropna()
        if d.empty:
            continue

        plt.figure(figsize=(7, 4))
        order = sorted(d["source"].unique())
        data = [d.loc[d["source"] == s, feat].values for s in order]
        plt.boxplot(data, tick_labels=order, showfliers=False)
        plt.title(f"{feat} by source (fliers hidden)")
        plt.xlabel("source")
        plt.ylabel(feat)
        plt.tight_layout()
        safe = feat.replace("#", "_").replace(".", "_")
        plt.savefig(out / f"box_{safe}.png", dpi=150)
        plt.close()

    # 4) Save compact QC summary
    summary = {
        "rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in cls.to_dict().items()},
        "missing_fraction_top": {
            k: float(v)
            for k, v in num.isna().mean().sort_values(ascending=False).head(10).to_dict().items()
        },
    }
    (out / "canonical_qc_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

    print(f"Wrote plots and summary to: {out}")


if __name__ == "__main__":
    main()

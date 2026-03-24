from __future__ import annotations

"""
Stage 3: Build balanced, source-aware train/val/test splits.

Input:
- data/processed/features_all.csv

Outputs:
- data/processed/features_train.csv
- data/processed/features_val.csv
- data/processed/features_test.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED_SOURCES = {"tmd", "uci_har", "bridge"}


def stratified_source_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Source-aware stratified split: split independently inside each (source, target) bucket.
    This keeps source/class representation across splits.
    """
    train_parts = []
    val_parts = []
    test_parts = []

    for (source, target), g in df.groupby(["source", "target"], dropna=False):
        g = g.sample(frac=1.0, random_state=seed)
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val

        # Keep at least one sample in test where feasible.
        if n >= 3 and n_test == 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(0, n_train - 1)

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val : n_train + n_val + n_test])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    return train_df, val_df, test_df


def balance_by_class(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df["target"].value_counts()
    min_count = int(counts.min())
    parts = []
    for cls, g in df.groupby("target", dropna=False):
        parts.append(g.sample(n=min_count, random_state=seed) if len(g) > min_count else g)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def split_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["target"].value_counts().to_dict().items()},
        "source_counts": {k: int(v) for k, v in df["source"].value_counts().to_dict().items()},
        "source_class": {
            f"{src}|{cls}": int(cnt)
            for (src, cls), cnt in df.groupby(["source", "target"]).size().to_dict().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create balanced source-aware splits")
    parser.add_argument("--input", default="data/processed/features_all.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        choices=["none", "train", "all"],
        default="train",
        help="Balance class counts in train only (default), all splits, or none",
    )
    args = parser.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0 or (args.train_frac + args.val_frac) >= 1:
        raise ValueError("Invalid fractions: require train_frac > 0, val_frac >= 0, and train_frac + val_frac < 1")

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)
    required = {"source", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["source", "target"]).copy()
    before_source_filter = len(df)
    df = df[df["source"].isin(ALLOWED_SOURCES)].copy()
    removed_source_rows = before_source_filter - len(df)
    if df.empty:
        raise ValueError("No rows left after relevant-source filtering")
    train_df, val_df, test_df = stratified_source_split(df, args.train_frac, args.val_frac, args.seed)

    if args.balance == "train":
        train_df = balance_by_class(train_df, args.seed)
    elif args.balance == "all":
        train_df = balance_by_class(train_df, args.seed)
        val_df = balance_by_class(val_df, args.seed)
        test_df = balance_by_class(test_df, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "features_train.csv"
    val_path = out_dir / "features_val.csv"
    test_path = out_dir / "features_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "input": str(in_path),
        "allowed_sources": sorted(ALLOWED_SOURCES),
        "removed_rows_non_relevant_sources": int(removed_source_rows),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "balance": args.balance,
        "seed": args.seed,
        "train": split_stats(train_df),
        "val": split_stats(val_df),
        "test": split_stats(test_df),
    }
    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")
    print(f"Wrote: {summary_path}")
    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()

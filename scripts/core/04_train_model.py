from __future__ import annotations

"""
Stage 4: Train tiny MLP model.

Inputs:
- data/processed/features_train.csv
- data/processed/features_val.csv

Outputs:
- experiments/<run_id>/models/tiny_mlp.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


META_COLUMNS = {"source", "sample_id", "user", "raw_target", "target"}


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    pr, rc, f1_cls, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1_cls[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def _train_tiny_mlp(x_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny MLP model")
    parser.add_argument("--train", default="data/processed/features_train.csv")
    parser.add_argument("--val", default="data/processed/features_val.csv")
    parser.add_argument("--run-id", default="run_local")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    run_dir = Path("experiments") / args.run_id
    models_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading train split: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"[INFO] Loading val split: {val_path}")
    df_val = pd.read_csv(val_path)

    feat_cols = _get_feature_columns(df_train)
    if not feat_cols:
        raise ValueError("No feature columns found in training file.")

    missing_in_val = [c for c in feat_cols if c not in df_val.columns]
    if missing_in_val:
        raise ValueError(f"Validation file is missing feature columns: {missing_in_val[:10]}")

    x_train = df_train[feat_cols].fillna(0.0)
    y_train = df_train["target"].astype(str)
    x_val = df_val[feat_cols].fillna(0.0)
    y_val = df_val["target"].astype(str)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    results = {
        "run_id": args.run_id,
        "num_train": int(len(df_train)),
        "num_val": int(len(df_val)),
        "num_features": int(len(feat_cols)),
        "feature_columns": feat_cols,
        "models": {},
    }

    print("[INFO] Training tiny MLP...")
    tiny_mlp = _train_tiny_mlp(x_train, y_train_enc)
    tiny_mlp_pred_enc = tiny_mlp.predict(x_val)
    tiny_mlp_pred = pd.Series(label_encoder.inverse_transform(tiny_mlp_pred_enc))
    tiny_mlp_metrics = _compute_metrics(y_val, tiny_mlp_pred)
    with (models_dir / "tiny_mlp.pkl").open("wb") as f:
        pickle.dump(tiny_mlp, f)
    results["models"]["tiny_mlp"] = tiny_mlp_metrics
    print(
        f"[OK] tiny_mlp | val_acc={tiny_mlp_metrics['accuracy']:.4f} "
        f"val_f1_macro={tiny_mlp_metrics['f1_macro']:.4f}"
    )

    metrics_path = metrics_dir / "val_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved models to: {models_dir}")
    print(f"[DONE] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Stage 5: Evaluate best model and export FPGA artifact.

Inputs:
- experiments/<run_id>/models/*
- data/processed/features_test.csv

Outputs:
- experiments/<run_id>/metrics.json
- experiments/<run_id>/confusion_matrix.csv
- experiments/<run_id>/models/best_int8.onnx
"""

import argparse
import importlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


META_COLUMNS = {"source", "sample_id", "user", "raw_target", "target"}
ALLOWED_SOURCES = {"tmd", "uci_har", "bridge"}


def _select_model_file(run_dir: Path, explicit_model_file: str | None) -> Path:
    models_dir = run_dir / "models"
    if explicit_model_file:
        model_path = Path(explicit_model_file)
        if not model_path.is_absolute():
            model_path = models_dir / explicit_model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    val_metrics_path = run_dir / "metrics" / "val_metrics.json"
    if val_metrics_path.exists():
        with val_metrics_path.open("r", encoding="utf-8") as f:
            val = json.load(f)
        models = val.get("models", {})
        if models:
            best_name = max(models.keys(), key=lambda name: models[name].get("f1_macro", -1.0))
            candidate = models_dir / f"{best_name}.pkl"
            if candidate.exists():
                return candidate

    pkl_files = sorted(models_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No model .pkl files found in: {models_dir}")
    return pkl_files[0]


def _human_table_from_report(report: dict, labels: list[str]) -> str:
    lines = [
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = report.get(label, {})
        lines.append(
            f"| {label} | {row.get('precision', 0.0):.4f} | {row.get('recall', 0.0):.4f} | "
            f"{row.get('f1-score', 0.0):.4f} | {int(row.get('support', 0))} |"
        )
    return "\n".join(lines)


def _save_confusion_plots(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if cm.dtype.kind == "f" else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_per_class_plot(report: dict, labels: list[str], out_path: Path) -> None:
    precision = [report.get(lbl, {}).get("precision", 0.0) for lbl in labels]
    recall = [report.get(lbl, {}).get("recall", 0.0) for lbl in labels]
    f1 = [report.get(lbl, {}).get("f1-score", 0.0) for lbl in labels]

    x = np.arange(len(labels))
    w = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, precision, width=w, label="precision")
    ax.bar(x, recall, width=w, label="recall")
    ax.bar(x + w, f1, width=w, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_source_metrics_plot(source_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    source_df.plot.bar(x="source", y="accuracy", ax=axes[0], legend=False, color="#1f77b4")
    source_df.plot.bar(x="source", y="f1_macro", ax=axes[1], legend=False, color="#ff7f0e")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].set_title("Accuracy by Source")
    axes[1].set_title("Macro F1 by Source")
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_support_plot(y_true: pd.Series, labels: list[str], out_path: Path) -> None:
    counts = y_true.value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="#2ca02c")
    ax.set_title("Test Support by Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_confidence_hist(max_prob: np.ndarray, correct: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_prob[correct], bins=25, alpha=0.7, label="correct", color="#2ca02c")
    ax.hist(max_prob[~correct], bins=25, alpha=0.7, label="incorrect", color="#d62728")
    ax.set_title("Prediction Confidence Histogram")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_roc_pr_curves(y_true: pd.Series, y_proba: np.ndarray, labels: list[str], plots_dir: Path) -> dict:
    y_bin = label_binarize(y_true, classes=labels)
    roc_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    pr_auc = average_precision_score(y_bin, y_proba, average="macro")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        cls_auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        cls_ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={cls_auc:.3f})")
        ax_pr.plot(r, p, label=f"{label} (AP={cls_ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("One-vs-Rest ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(plots_dir / "roc_curves.png", dpi=160)
    plt.close(fig_roc)

    ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)
    fig_pr.tight_layout()
    fig_pr.savefig(plots_dir / "pr_curves.png", dpi=160)
    plt.close(fig_pr)

    return {"roc_auc_macro_ovr": float(roc_auc), "average_precision_macro": float(pr_auc)}


def _save_feature_importance_plot(model, feature_cols: list[str], out_path: Path) -> bool:
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            score = np.abs(coef)
        else:
            score = np.mean(np.abs(coef), axis=0)
        order = np.argsort(score)[::-1]
        sorted_scores = score[order]
        sorted_names = [feature_cols[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(sorted_names[::-1], sorted_scores[::-1], color="#9467bd")
        ax.set_title("Feature Importance (|coef|)")
        ax.set_xlabel("Mean absolute coefficient")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return True
    return False


def _infer_label_mapping(model_classes, labels: list[str]) -> dict | None:
    classes_arr = np.asarray(model_classes)
    if classes_arr.size != len(labels):
        return None
    if classes_arr.dtype.kind in {"i", "u"}:
        expected = np.arange(len(labels))
        if np.array_equal(np.sort(classes_arr), expected):
            return {int(i): labels[int(i)] for i in expected}
    return None


def _export_onnx_int8_if_possible(model, feature_cols: list[str], model_dir: Path, model_name: str) -> dict:
    result = {"onnx_exported": False, "int8_quantized": False, "message": ""}
    onnx_path = model_dir / f"{model_name}.onnx"
    int8_path = model_dir / "best_int8.onnx"

    try:
        skl2onnx = importlib.import_module("skl2onnx")
        data_types = importlib.import_module("skl2onnx.common.data_types")
        convert_sklearn = getattr(skl2onnx, "convert_sklearn")
        FloatTensorType = getattr(data_types, "FloatTensorType")

        initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with onnx_path.open("wb") as f:
            f.write(onnx_model.SerializeToString())
        result["onnx_exported"] = True
    except Exception as e:  # pragma: no cover
        result["message"] = f"ONNX export skipped: {e}"
        return result

    try:
        quant_mod = importlib.import_module("onnxruntime.quantization")
        QuantType = getattr(quant_mod, "QuantType")
        quantize_dynamic = getattr(quant_mod, "quantize_dynamic")

        quantize_dynamic(
            str(onnx_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        result["int8_quantized"] = True
        result["message"] = "ONNX and INT8 export successful"
    except Exception as e:  # pragma: no cover
        result["message"] = f"ONNX exported, INT8 quantization skipped: {e}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and export quantized model")
    parser.add_argument("--test", default="data/processed/features_test.csv")
    parser.add_argument("--run-id", default="run_local")
    parser.add_argument(
        "--model-file",
        default=None,
        help="Optional model file name/path. If omitted, best val model is selected.",
    )
    args = parser.parse_args()

    run_dir = Path("experiments") / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    test_path = Path(args.test)
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    model_path = _select_model_file(run_dir, args.model_file)
    model_name = model_path.stem

    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading test split: {test_path}")
    df_test = pd.read_csv(test_path)
    pre_filter_rows = len(df_test)
    df_test = df_test[df_test["source"].isin(ALLOWED_SOURCES)].copy()
    if df_test.empty:
        raise ValueError("No relevant-source rows in test split after filtering")
    removed_rows = pre_filter_rows - len(df_test)

    feature_cols = [c for c in df_test.columns if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found in test file.")

    x_test = df_test[feature_cols].fillna(0.0)
    y_true = df_test["target"].astype(str)
    labels = sorted(y_true.unique().tolist())

    print(f"[INFO] Loading model: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    raw_pred = pd.Series(model.predict(x_test), index=df_test.index)
    proba_available = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(x_test) if proba_available else None

    class_mapping = _infer_label_mapping(getattr(model, "classes_", []), labels)
    if class_mapping is not None:
        y_pred = raw_pred.astype(int).map(class_mapping).astype(str)
    else:
        y_pred = raw_pred.astype(str)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    base_metrics = {
        "run_id": args.run_id,
        "selected_model": model_name,
        "allowed_sources": sorted(ALLOWED_SOURCES),
        "removed_rows_non_relevant_sources": int(removed_rows),
        "num_samples_test": int(len(df_test)),
        "num_features": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }

    per_class_df = pd.DataFrame(
        [
            {
                "class": label,
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        ]
    )
    per_class_df.to_csv(metrics_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    cm_norm = cm_norm / row_sum

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_df.to_csv(metrics_dir / "confusion_matrix.csv")
    cm_norm_df.to_csv(metrics_dir / "confusion_matrix_normalized.csv")

    _save_confusion_plots(cm, labels, plots_dir / "confusion_matrix_raw.png", "Confusion Matrix (Counts)")
    _save_confusion_plots(
        cm_norm,
        labels,
        plots_dir / "confusion_matrix_normalized.png",
        "Confusion Matrix (Row-Normalized)",
    )
    _save_per_class_plot(report, labels, plots_dir / "per_class_metrics.png")
    _save_support_plot(y_true, labels, plots_dir / "support_by_class.png")

    by_source_rows = []
    for source_name, g in df_test.assign(pred=y_pred.values).groupby("source"):
        src_true = g["target"].astype(str)
        src_pred = g["pred"].astype(str)
        by_source_rows.append(
            {
                "source": source_name,
                "accuracy": float(accuracy_score(src_true, src_pred)),
                "f1_macro": float(
                    f1_score(src_true, src_pred, labels=labels, average="macro", zero_division=0)
                ),
            }
        )
    source_df = pd.DataFrame(by_source_rows).sort_values("source")
    source_df = source_df[source_df["source"].isin(ALLOWED_SOURCES)].copy()
    source_df.to_csv(metrics_dir / "source_metrics.csv", index=False)
    if not source_df.empty:
        _save_source_metrics_plot(source_df, plots_dir / "source_metrics.png")

    if y_proba is not None:
        proba_cols = list(getattr(model, "classes_", labels))
        if class_mapping is not None:
            proba_cols = [class_mapping.get(int(c), str(c)) for c in proba_cols]
        proba_df = pd.DataFrame(y_proba, columns=proba_cols)
        aligned_proba = proba_df.reindex(columns=labels, fill_value=0.0).to_numpy()

        max_prob = aligned_proba.max(axis=1)
        correct = y_pred.to_numpy() == y_true.to_numpy()
        _save_confidence_hist(max_prob, correct, plots_dir / "confidence_histogram.png")
        roc_pr_metrics = _save_roc_pr_curves(y_true, aligned_proba, labels, plots_dir)
        base_metrics.update(roc_pr_metrics)

        errors_df = df_test[["source", "sample_id", "target"]].copy()
        errors_df["pred"] = y_pred.values
        errors_df["confidence"] = max_prob
        errors_df = errors_df[errors_df["target"] != errors_df["pred"]]
        errors_df = errors_df.sort_values("confidence", ascending=False)
        errors_df.head(300).to_csv(metrics_dir / "top_confident_errors.csv", index=False)

    if _save_feature_importance_plot(model, feature_cols, plots_dir / "feature_importance.png"):
        base_metrics["feature_importance_plot"] = "created"
    else:
        base_metrics["feature_importance_plot"] = "not_available_for_model_type"

    export_info = _export_onnx_int8_if_possible(model, feature_cols, run_dir / "models", model_name)
    base_metrics["export"] = export_info

    metrics_json_path = metrics_dir / "test_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(base_metrics, f, indent=2)

    report_md_path = metrics_dir / "report_human_readable.md"
    table_md = _human_table_from_report(report, labels)
    report_text = "\n".join(
        [
            f"# Evaluation Report ({args.run_id})",
            "",
            f"- selected_model: {model_name}",
            f"- test_samples: {len(df_test)}",
            f"- features: {len(feature_cols)}",
            f"- accuracy: {base_metrics['accuracy']:.4f}",
            f"- balanced_accuracy: {base_metrics['balanced_accuracy']:.4f}",
            f"- macro_f1: {base_metrics['f1_macro']:.4f}",
            f"- weighted_f1: {base_metrics['f1_weighted']:.4f}",
            f"- matthews_corrcoef: {base_metrics['matthews_corrcoef']:.4f}",
            f"- cohen_kappa: {base_metrics['cohen_kappa']:.4f}",
            "",
            "## Per-class metrics",
            "",
            table_md,
            "",
            "## Generated artifacts",
            "",
            "- metrics/test_metrics.json",
            "- metrics/confusion_matrix.csv",
            "- metrics/confusion_matrix_normalized.csv",
            "- metrics/source_metrics.csv",
            "- metrics/per_class_metrics.csv",
            "- metrics/top_confident_errors.csv (if predict_proba available)",
            "- plots/confusion_matrix_raw.png",
            "- plots/confusion_matrix_normalized.png",
            "- plots/per_class_metrics.png",
            "- plots/support_by_class.png",
            "- plots/source_metrics.png",
            "- plots/confidence_histogram.png (if predict_proba available)",
            "- plots/roc_curves.png (if predict_proba available)",
            "- plots/pr_curves.png (if predict_proba available)",
            "- plots/feature_importance.png (if supported by model)",
            "",
            f"## Export status\n\n- {export_info.get('message', 'no export status')}",
        ]
    )
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[DONE] Selected model: {model_name}")
    print(f"[DONE] Metrics JSON: {metrics_json_path}")
    print(f"[DONE] Human report: {report_md_path}")
    print(f"[DONE] Plots directory: {plots_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _quantize_to_int(weights: np.ndarray, frac_bits: int) -> tuple[np.ndarray, float]:
    max_int = (2**15) - 1
    scale = float(2**frac_bits)
    q = np.clip(np.round(weights * scale), -max_int - 1, max_int).astype(np.int16)
    return q, scale


def _to_c_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace("/", "_")


def _array_to_c_initializer(arr: np.ndarray) -> str:
    flat = arr.flatten(order="C")
    return "{" + ", ".join(str(int(v)) for v in flat) + "}"


def _write_header(
    out_path: Path,
    model_name: str,
    classes: list[str],
    qfrac: int,
    payload: dict[str, np.ndarray],
) -> None:
    guard = _to_c_name(f"FPGA_WEIGHTS_{model_name}_H").upper()
    lines: list[str] = []
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define MODEL_Q_FRAC_BITS {qfrac}")
    lines.append(f"#define MODEL_NUM_CLASSES {len(classes)}")
    lines.append("")
    for i, c in enumerate(classes):
        lines.append(f"#define CLASS_{_to_c_name(c).upper()} {i}")
    lines.append("")

    for key, arr in payload.items():
        c_key = _to_c_name(key).upper()
        lines.append(f"#define {c_key}_SIZE {arr.size}")
        if arr.ndim == 2:
            lines.append(f"#define {c_key}_ROWS {arr.shape[0]}")
            lines.append(f"#define {c_key}_COLS {arr.shape[1]}")
        elif arr.ndim == 1:
            lines.append(f"#define {c_key}_LEN {arr.shape[0]}")
        init = _array_to_c_initializer(arr)
        lines.append(f"static const int16_t {key}[{arr.size}] = {init};")
        lines.append("")

    lines.append(f"#endif /* {guard} */")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_tiny_mlp(model, out_dir: Path, frac_bits: int, classes: list[str]) -> dict:
    w1 = np.asarray(model.coefs_[0], dtype=np.float32)
    b1 = np.asarray(model.intercepts_[0], dtype=np.float32)
    w2 = np.asarray(model.coefs_[1], dtype=np.float32)
    b2 = np.asarray(model.intercepts_[1], dtype=np.float32)

    w1_q, scale = _quantize_to_int(w1, frac_bits)
    b1_q, _ = _quantize_to_int(b1, frac_bits)
    w2_q, _ = _quantize_to_int(w2, frac_bits)
    b2_q, _ = _quantize_to_int(b2, frac_bits)

    np.savez(out_dir / "weights_float.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    np.savez(out_dir / "weights_q.npz", w1_q=w1_q, b1_q=b1_q, w2_q=w2_q, b2_q=b2_q)
    _write_header(
        out_dir / "weights_q15.h",
        "tiny_mlp",
        classes,
        frac_bits,
        {
            "mlp_w1_q": w1_q,
            "mlp_b1_q": b1_q,
            "mlp_w2_q": w2_q,
            "mlp_b2_q": b2_q,
        },
    )

    return {
        "model_type": "tiny_mlp",
        "w1_shape": list(w1.shape),
        "b1_shape": list(b1.shape),
        "w2_shape": list(w2.shape),
        "b2_shape": list(b2.shape),
        "q_scale": scale,
        "activation_hidden": "relu",
        "activation_output": "softmax",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model weights for FPGA")
    parser.add_argument("--run-id", default="run_local")
    parser.add_argument("--model-file", default="tiny_mlp.pkl", help="Model file name/path")
    parser.add_argument("--q-frac-bits", type=int, default=15, help="Fractional bits for fixed-point quantization")
    args = parser.parse_args()

    run_dir = Path("experiments") / args.run_id
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_path: Path
    model_path = Path(args.model_file)
    if not model_path.is_absolute():
        model_path = models_dir / args.model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    classes_raw = list(getattr(model, "classes_", ["human", "noise", "vehicle"]))
    if classes_raw and isinstance(classes_raw[0], (int, np.integer)):
        classes = ["human", "noise", "vehicle"]
    else:
        classes = [str(c) for c in classes_raw]

    model_name = model_path.stem
    out_dir = run_dir / "fpga" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "coefs_") and hasattr(model, "intercepts_"):
        export_info = _export_tiny_mlp(model, out_dir, args.q_frac_bits, classes)
    else:
        raise ValueError(f"Unsupported model type for FPGA export: {type(model).__name__}")

    metadata = {
        "run_id": args.run_id,
        "model_file": str(model_path),
        "classes": classes,
        "quantization": {"format": "Qm.n", "frac_bits": args.q_frac_bits},
        "artifacts": {
            "float_npz": str(out_dir / "weights_float.npz"),
            "quant_npz": str(out_dir / "weights_q.npz"),
            "c_header": str(out_dir / "weights_q15.h"),
        },
        "model": export_info,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Exported FPGA artifacts to: {out_dir}")
    print(f"[DONE] Metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

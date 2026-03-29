# FPGA Implementation Plan

## 0. FPGA Implementation Pipeline (Structured)
Use this as the execution order for the FPGA engineer.

1. Confirm model contract files are present.
2. Validate dimensions/classes/quantization from metadata.json.
3. Implement tiny MLP fixed-point core in HLS/RTL.
4. Build software golden generator and test vectors from features_test.csv.
5. Run csim and fix arithmetic/shape mismatches.
6. Run csynth and check timing/resources.
7. Run cosim and compare predictions/logits against golden outputs.
8. Package reports + artifacts for board integration.

## 1. Files Required for FPGA Work
These are the only repo files you need for implementation and validation.

### Mandatory model contract files
- [experiments/run_local/fpga/tiny_mlp/metadata.json](experiments/run_local/fpga/tiny_mlp/metadata.json)
- [experiments/run_local/fpga/tiny_mlp/weights_q15.h](experiments/run_local/fpga/tiny_mlp/weights_q15.h)

### Mandatory verification files
- [data/processed/features_test.csv](data/processed/features_test.csv)
- [experiments/run_local/metrics/test_metrics.json](experiments/run_local/metrics/test_metrics.json)

### Optional helper files
- [experiments/run_local/fpga/tiny_mlp/weights_q.npz](experiments/run_local/fpga/tiny_mlp/weights_q.npz)
- [experiments/run_local/fpga/tiny_mlp/weights_float.npz](experiments/run_local/fpga/tiny_mlp/weights_float.npz)

## 1.1 Important Repo Files and What They Do
These are the most important files for implementation handoff.

1. [experiments/run_local/fpga/tiny_mlp/metadata.json](experiments/run_local/fpga/tiny_mlp/metadata.json)
- Why it matters: single source of truth for model shape, class order, and quantization settings.
- Use it for: validating tensor dimensions in hardware and class-index mapping.

2. [experiments/run_local/fpga/tiny_mlp/weights_q15.h](experiments/run_local/fpga/tiny_mlp/weights_q15.h)
- Why it matters: direct C/HLS consumable fixed-point arrays.
- Use it for: embedding weights/biases into accelerator implementation.

3. [data/processed/features_test.csv](data/processed/features_test.csv)
- Why it matters: consistent test input distribution for verification.
- Use it for: generating deterministic Q15 test vectors and expected labels.

4. [experiments/run_local/metrics/test_metrics.json](experiments/run_local/metrics/test_metrics.json)
- Why it matters: baseline model quality reference.
- Use it for: parity sanity check at class-level and source-level.

5. [scripts/core/06_export_fpga_weights.py](scripts/core/06_export_fpga_weights.py)
- Why it matters: defines exact quantization/export logic.
- Use it for: re-exporting artifacts or auditing Q15 conversion behavior.

6. [scripts/core/04_train_model.py](scripts/core/04_train_model.py)
- Why it matters: defines tiny MLP architecture and training setup.
- Use it for: understanding layer sizes and activation flow in hardware implementation.

## 2. If FPGA Files Are Missing: Regenerate Minimal Artifacts
Run only these steps:

```bash
python scripts/core/04_train_model.py --train data/processed/features_train.csv --val data/processed/features_val.csv --run-id run_local
python scripts/core/05_evaluate_export.py --test data/processed/features_test.csv --run-id run_local
python scripts/core/06_export_fpga_weights.py --run-id run_local --model-file tiny_mlp.pkl --q-frac-bits 15
```

## 3. Hardware Contract (Read First)
Source of truth:
- [experiments/run_local/fpga/tiny_mlp/metadata.json](experiments/run_local/fpga/tiny_mlp/metadata.json)

Current contract:
- model: tiny_mlp
- classes: human, noise, vehicle
- input features: 14
- hidden layer: 16 (ReLU)
- output logits: 3
- quantization format: Qm.n
- fractional bits: 15
- scale: 32768

Tensor shapes:
- w1: 14 x 16
- b1: 16
- w2: 16 x 3
- b2: 3

## 4. What the Header File Contains
Header used by firmware/HLS:
- [experiments/run_local/fpga/tiny_mlp/weights_q15.h](experiments/run_local/fpga/tiny_mlp/weights_q15.h)

Contains:
- class index defines
- quantization define
- flattened int16 arrays:
1. mlp_w1_q
2. mlp_b1_q
3. mlp_w2_q
4. mlp_b2_q

Interpretation:
- arrays are flattened row-major

## 5. Fixed-Point Inference Equation
For one input vector x of length 14, run these steps in order:

1. Compute hidden pre-activation (16 values):
- h_pre[j] = b1[j] + sum over i=0..13 of (w1[i][j] * x[i])

2. Apply ReLU activation:
- h[j] = max(0, h_pre[j])

3. Compute output logits (3 values):
- y[k] = b2[k] + sum over j=0..15 of (w2[j][k] * h[j])

4. Predict class index:
- pred_class = index of largest value in y[0], y[1], y[2]

5. Class index mapping from metadata:
- 0 = human
- 1 = noise
- 2 = vehicle

Implementation notes:
1. Use int32 or wider accumulators.
2. Apply controlled scaling shifts consistent with Q15.
3. Saturate when narrowing back to int16.
4. Keep arithmetic order exactly the same in software golden and RTL/HLS.

## 6. Recommended FPGA Workspace
Add these directories for implementation flow:
- fpga/hls_tiny_mlp
- fpga/tb
- fpga/scripts
- fpga/reports

## 7. Suggested Top-Level Accelerator API

```c
void tiny_mlp_accel(
    const int16_t x_q15[14],
    int16_t logits_q15[3],
    uint8_t *pred_class
);
```

## 8. Testbench Inputs and Golden Outputs
Use [data/processed/features_test.csv](data/processed/features_test.csv) to generate deterministic vectors.

Recommended testbench files:
- fpga/tb/input_q15.txt
- fpga/tb/expected_logits_q15.txt
- fpga/tb/expected_pred.txt

Golden generation rule:
1. Quantize test features with same Q15 policy.
2. Compute software integer reference with exact same arithmetic path.
3. Compare FPGA output against reference.

Practical recommendation:
1. Start with 100 samples smoke test.
2. Then run full features_test.csv regression.
3. Save mismatch report by sample_id.

## 9. Implementation Pipeline (Clear Sequence)
1. Load metadata and confirm dimensions/class order.
2. Integrate weights_q15.h arrays into HLS/RTL core.
3. Implement forward pass with ReLU hidden layer.
4. Build C testbench using generated input/golden vectors.
5. Run csim.
6. Run csynth.
7. Run cosim.
8. Archive timing/resource/parity reports.

Deliverable per step:
1. Contract-check log (dims/classes/q format).
2. Compilable accelerator source.
3. Unit test for layer-1 and layer-2 outputs.
4. Deterministic test vector files.
5. csim pass report.
6. csynth utilization and timing report.
7. cosim parity report.
8. Final handoff package.

## 10. Acceptance Gates
Must pass before handoff complete:
1. Label parity vs software reference >= 99.5%.
2. No class-order mismatch vs metadata.
3. No overflow-induced instability on representative vectors.
4. Timing/resource fits selected target board.
5. Deterministic rerun gives identical predicted labels.

## 11. Common Failure Modes
1. Wrong class order hardcoded in FPGA path.
2. Wrong row/column interpretation of flattened arrays.
3. Missing ReLU between layers.
4. Incorrect Q15 shift/saturation policy.

## 12. Final Handoff Bundle
Share these files with FPGA implementer:
1. [experiments/run_local/fpga/tiny_mlp/metadata.json](experiments/run_local/fpga/tiny_mlp/metadata.json)
2. [experiments/run_local/fpga/tiny_mlp/weights_q15.h](experiments/run_local/fpga/tiny_mlp/weights_q15.h)
3. [data/processed/features_test.csv](data/processed/features_test.csv)
4. [experiments/run_local/metrics/test_metrics.json](experiments/run_local/metrics/test_metrics.json)

This is the minimal complete package needed to implement and validate the FPGA inference core.

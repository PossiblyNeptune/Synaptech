# Implementation Plan

## 1. Planning Objective
Define the forward execution roadmap for the final edge intrusion classifier program, covering:
1. data quality and model quality improvements,
2. export and deployment reliability,
3. FPGA implementation path from software artifacts to hardware-ready inference.

## 2. Program Baseline
### 2.1 Current quantitative baseline
Dataset and split baseline:
- canonical rows: 50,037
- train rows: 30,837
- validation rows: 7,502
- test rows: 7,513

Model baseline (tiny_mlp):
- validation accuracy: 0.8392
- validation macro_f1: 0.8428
- test accuracy: 0.8312
- test balanced_accuracy: 0.8390
- test macro_f1: 0.8350
- test weighted_f1: 0.8303
- test MCC: 0.7479
- test Cohen kappa: 0.7461
- test macro ROC-AUC: 0.9442
- test macro average precision: 0.9048

Per-class recall baseline:
- human: 0.8876
- noise: 0.8834
- vehicle: 0.7461

### 2.2 Source-level performance baseline
- bridge: accuracy 0.5000, f1_macro 0.3505
- tmd: accuracy 0.8104, f1_macro 0.8138
- uci_har: accuracy 0.9425, f1_macro 0.6480

## 3. Strategic Priorities
Priority 1:
- Raise bridge-domain performance and reduce source gap.

Priority 2:
- Improve vehicle recall while preserving human/noise strength.

Priority 3:
- Harden export reproducibility and quantized parity checks.

Priority 4:
- Deliver FPGA virtual deployment package and board transition readiness.

## 4. Workstream A: Data Quality and Labeling
Goal:
- Increase label quality and coverage for difficult and underrepresented event patterns.

Tasks:
1. Complete and verify bridge label coverage in configs/bridge_event_labels.csv.
2. Add annotation policy for ambiguous events and noise taxonomy.
3. Add automated stage-1 drop-reason counters by source and class.
4. Add per-run bridge class coverage report in split summary.

Deliverables:
- curated bridge labels with review notes
- automated data quality summary in pipeline outputs
- updated split-level class/source diagnostics

Acceptance gate:
- zero unlabeled bridge events in active training pipeline

## 5. Workstream B: Feature and Model Optimization
Goal:
- Improve vehicle detection and maintain robust macro performance.

Tasks:
1. Run feature ablations focused on vehicle separability.
2. Add confidence calibration and reject-option threshold tuning.
3. Use top_confident_errors.csv for hard-example retraining strategy.
4. Evaluate compact architecture variants that remain FPGA-friendly:
   - 14->16->3
   - 14->20->3
   - 14->16->8->3

Deliverables:
- benchmark table with architecture and feature variant outcomes
- selected architecture decision note with rationale

Acceptance gates:
- test macro_f1 >= 0.84
- test vehicle recall >= 0.78
- test balanced_accuracy >= 0.84

## 6. Workstream C: Export and Quantization Reliability
Goal:
- Make model export paths deterministic, testable, and deployment-safe.

Tasks:
1. Enable ONNX toolchain and integrate export checks.
2. Add float-versus-quantized parity tests.
3. Validate metadata consistency (class order, tensor shapes, q format).
4. Add export smoke test for clean environments.

Deliverables:
- export validation script
- parity report artifact per run
- reproducible export command checklist

Acceptance gate:
- quantized macro_f1 degradation <= 1.0 percentage point from float baseline

## 7. Workstream D: FPGA Virtual Deployment
Goal:
- Move from software artifacts to verified hardware simulation path.

Scope:
- tiny MLP fixed-point inference core
- C simulation and C/RTL co-simulation
- synthesis resource and timing reports

Tasks:
1. Implement tiny MLP HLS/RTL inference core with Q15 arithmetic contract.
2. Generate deterministic fixed-point test vectors from features_test.csv.
3. Run csim, csynth, cosim and archive report bundle.
4. Define board mapping for Zynq-class targets (clock, interfaces, transfer strategy).

Deliverables:
- HLS project sources and run scripts
- simulation parity report
- timing/resource report package

Acceptance gates:
- label agreement >= 99.5% vs software Q15 reference
- timing/resource within board budget envelope

## 8. Milestones
M1: Data quality stabilization
- label completion and stage-1 diagnostics finalized

M2: Model quality uplift
- vehicle recall and macro_f1 targets achieved

M3: Export reliability completion
- ONNX and quantized parity validation operational

M4: FPGA virtual deployment package
- HLS/RTL simulation and synthesis reports completed

## 9. Risks and Mitigation Plan
Risk 1: Limited bridge event diversity
- Mitigation: prioritize bridge event labeling and targeted collection

Risk 2: Vehicle improvements reduce other class performance
- Mitigation: enforce per-class guardrails at model selection

Risk 3: Toolchain dependency drift for export
- Mitigation: lock dependencies and add CI smoke checks

Risk 4: FPGA implementation resource pressure
- Mitigation: architecture fallback and pragma tuning iterations

## 10. Reporting and Control
At every milestone:
1. Update README.md with refreshed measured results and artifact pointers.
2. Update plan.md with completed tasks, outstanding blockers, and next milestone gates.

Control metrics tracked each cycle:
- macro_f1
- per-class recall
- source-wise metrics
- parity checks for export formats
- FPGA simulation pass/fail and resource margins

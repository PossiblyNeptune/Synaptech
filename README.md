# Edge AI Seismic Intrusion Detection

## 1. Project Summary
This project implements an end-to-end edge AI pipeline for classifying seismic and vibration events into three operational classes:
- human
- vehicle
- noise

The solution is designed for practical software validation and FPGA-oriented deployment using compact features and a tiny multilayer perceptron model.

## 2. Final System Scope
### Data sources
- data/raw/tmd
- data/raw/bridge
- data/raw/uci_har

### Core model
- tiny_mlp

### Deployment target
- fixed-point Q15 artifacts for FPGA integration

## 3. End-to-End Pipeline
Core scripts in execution order:
1. scripts/core/01_prepare_data.py
2. scripts/core/02_extract_features.py
3. scripts/core/03_build_splits.py
4. scripts/core/04_train_model.py
5. scripts/core/05_evaluate_export.py
6. scripts/core/06_export_fpga_weights.py

Pipeline stages:
1. Raw source harmonization into a canonical table.
2. Compact feature extraction with source-aware scaling.
3. Source-aware train/validation/test split generation and train balancing.
4. Tiny MLP training and validation metrics generation.
5. Test evaluation with detailed metrics and diagnostic plots.
6. FPGA artifact export for fixed-point inference integration.

## 4. Canonical and Feature Design
### 4.1 Canonical table
Canonical rows include metadata and sensor summary statistics:
- metadata: source, sample_id, user, raw_target, target, time_sec
- motion summaries from accelerometer, linear acceleration, and gyroscope channels

Main canonical artifact:
- data/interim/all_sources_canonical.csv

### 4.2 Final model feature set (14)
- f_acc_mean
- f_acc_std
- f_acc_range
- f_acc_energy_proxy
- f_lin_mean
- f_lin_std
- f_lin_range
- f_lin_energy_proxy
- f_gyr_mean
- f_gyr_std
- f_gyr_range
- f_gyr_energy_proxy
- f_motion_ratio
- f_gyro_vs_linear

## 5. Current Dataset Volumes
### 5.1 Canonical rows by source
- tmd: 38,723
- uci_har: 10,299
- bridge: 1,015

Total canonical rows: 50,037

### 5.2 Split configuration
- train_frac: 0.70
- val_frac: 0.15
- test_frac: 0.15
- balance: train

Split sizes:
- train: 30,837
- val: 7,502
- test: 7,513

Balanced train class counts:
- human: 10,279
- vehicle: 10,279
- noise: 10,279

Validation class counts:
- human: 2,201
- vehicle: 2,919
- noise: 2,382

Test class counts:
- human: 2,206
- vehicle: 2,922
- noise: 2,385

## 6. Model Configuration
Tiny MLP architecture:
- input: 14
- hidden: 16 (ReLU)
- output: 3

Training setup:
- solver: adam
- alpha: 1e-4
- batch_size: 128
- learning_rate_init: 1e-3
- max_iter: 120
- early_stopping: true
- validation_fraction: 0.1

## 7. Performance Snapshot
### 7.1 Validation metrics
From experiments/run_local/metrics/val_metrics.json:
- accuracy: 0.8392
- macro_f1: 0.8428

Per-class validation metrics:
- human: precision 0.8863, recall 0.8923, f1 0.8893
- noise: precision 0.8022, recall 0.8854, f1 0.8417
- vehicle: precision 0.8367, recall 0.7616, f1 0.7973

### 7.2 Test metrics
From experiments/run_local/metrics/test_metrics.json:
- selected_model: tiny_mlp
- test_samples: 7,513
- num_features: 14
- accuracy: 0.8312
- balanced_accuracy: 0.8390
- macro_f1: 0.8350
- weighted_f1: 0.8303
- matthews_corrcoef: 0.7479
- cohen_kappa: 0.7461
- roc_auc_macro_ovr: 0.9442
- average_precision_macro: 0.9048

Per-class test metrics:
- human: precision 0.8868, recall 0.8876, f1 0.8872, support 2206
- noise: precision 0.7844, recall 0.8834, f1 0.8310, support 2385
- vehicle: precision 0.8324, recall 0.7461, f1 0.7869, support 2922

Source-wise test metrics:
- bridge: accuracy 0.5000, f1_macro 0.3505
- tmd: accuracy 0.8104, f1_macro 0.8138
- uci_har: accuracy 0.9425, f1_macro 0.6480

## 8. Deliverables
### Data
- data/interim/all_sources_canonical.csv
- data/processed/features_all.csv
- data/processed/features_train.csv
- data/processed/features_val.csv
- data/processed/features_test.csv
- data/processed/split_summary.json

### Model
- experiments/run_local/models/tiny_mlp.pkl

### Metrics and reports
- experiments/run_local/metrics/val_metrics.json
- experiments/run_local/metrics/test_metrics.json
- experiments/run_local/metrics/per_class_metrics.csv
- experiments/run_local/metrics/source_metrics.csv
- experiments/run_local/metrics/confusion_matrix.csv
- experiments/run_local/metrics/confusion_matrix_normalized.csv
- experiments/run_local/metrics/top_confident_errors.csv

### Evaluation plots
- experiments/run_local/plots/confusion_matrix_raw.png
- experiments/run_local/plots/confusion_matrix_normalized.png
- experiments/run_local/plots/per_class_metrics.png
- experiments/run_local/plots/support_by_class.png
- experiments/run_local/plots/source_metrics.png
- experiments/run_local/plots/confidence_histogram.png
- experiments/run_local/plots/roc_curves.png
- experiments/run_local/plots/pr_curves.png

### FPGA artifacts
- experiments/run_local/fpga/tiny_mlp/weights_float.npz
- experiments/run_local/fpga/tiny_mlp/weights_q.npz
- experiments/run_local/fpga/tiny_mlp/weights_q15.h
- experiments/run_local/fpga/tiny_mlp/metadata.json

FPGA metadata highlights:
- quantization format: Qm.n
- frac_bits: 15
- q_scale: 32768
- class order: human, noise, vehicle
- tensor shapes: w1 (14x16), b1 (16), w2 (16x3), b2 (3)

## 9. Reproducible Run Commands
Run from repository root:

```bash
python scripts/core/01_prepare_data.py --data-root data
python scripts/core/02_extract_features.py --input data/interim/all_sources_canonical.csv --output data/processed/features_all.csv --sample-rate 100
python scripts/core/03_build_splits.py --input data/processed/features_all.csv --out-dir data/processed --balance train
python scripts/core/04_train_model.py --train data/processed/features_train.csv --val data/processed/features_val.csv --run-id run_local
python scripts/core/05_evaluate_export.py --test data/processed/features_test.csv --run-id run_local
python scripts/core/06_export_fpga_weights.py --run-id run_local --model-file tiny_mlp.pkl
```

## 10. Project Outcome
The repository currently provides a complete and reproducible software-to-FPGA-prep workflow for intrusion classification with:
1. Source-harmonized dataset preparation.
2. Compact feature engineering.
3. Source-aware evaluation.
4. Strong baseline tiny-MLP performance.
5. Ready-to-integrate Q15 FPGA artifacts.

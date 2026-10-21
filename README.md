# Post-Training Weight Quantization of DistilBERT on SST-2

A systematic comparison of 38 weight-quantization configurations on
`distilbert/distilbert-base-uncased-finetuned-sst-2-english` (GLUE SST-2), with
the goal of finding the configuration that shrinks the model the most **without
noticeably changing its behavior**.

**Result:** `INT8_AFFINE_per_channel_axis0` is the selected configuration. It cuts
the serialized model size by **74.64%** (267.82 MB to 67.92 MB) with **0.00 pp
accuracy loss**, a logit RMSE of **0.040** against the FP32 model, and **no
changed predictions** on the 872 validation examples.

---

## Targets

A configuration passes only if it meets all three conditions:

| Metric | Requirement |
|---|---|
| Size reduction | at least 70% versus FP32 (measured on the serialized artifact) |
| Accuracy loss | at most 1.00 percentage point versus FP32 |
| Logit RMSE | below 0.2 versus FP32 logits |

The logit limit was tightened from 0.3 to 0.2 during the project, which removed
three configurations that passed only the looser limit (INT4_AFFINE_g64,
INT4_AFFINE_g32 and INT8_AFFINE_per_tensor).

## Setup

- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
  (about 67M parameters, 267.82 MB in FP32)
- **Data:** GLUE SST-2 validation split, 872 labeled examples, shuffled with seed 42
- **FP32 baseline accuracy:** 91.06% (794 of 872 correct)
- **Device:** CPU
- **What is quantized:** weights only. Weights are stored as packed integer codes
  with scales (and zero-points for affine schemes). At load time they are
  dequantized back to FP32, so accuracy reflects quantization error, not integer
  kernels.

## Configurations tested (38)

| Family | Variants |
|---|---|
| Floating-point references (2) | FP16, BF16 (both 133.94 MB, 49.99% reduction) |
| Integer (36) | bit width **INT8 / INT4 / INT2** x scheme **symmetric / affine** x granularity **per-tensor / per-channel (axis 0 and 1) / per-group (g32, g64, g128)** |

## Results

Sizes and reductions below are measured from the serialized artifacts
(`artifact_comparison.csv`), not estimated.

### Configurations that pass all three targets (9)

All nine are INT8. Sorted by size reduction:

| Config | Size (MB) | Reduction | Accuracy loss | Logit RMSE |
|---|---|---|---|---|
| INT8_SYM_per_channel_axis1 | 67.21 | 74.91% | 0.23 pp | 0.066 |
| INT8_SYM_per_channel_axis0 | 67.32 | 74.87% | 0.23 pp | 0.044 |
| INT8_AFFINE_per_channel_axis1 | 67.58 | 74.77% | 0.34 pp | 0.042 |
| **INT8_AFFINE_per_channel_axis0** | **67.92** | **74.64%** | **0.00 pp** | **0.040** |
| INT8_SYM_g128 | 69.12 | 74.19% | 0.23 pp | 0.034 |
| INT8_SYM_g64 | 71.21 | 73.41% | 0.00 pp | 0.027 |
| INT8_AFFINE_g128 | 73.34 | 72.62% | 0.23 pp | 0.025 |
| INT8_SYM_g32 | 75.40 | 71.85% | 0.00 pp | 0.021 |
| INT8_AFFINE_g64 | 79.61 | 70.28% | 0.00 pp | 0.018 |

INT8_AFFINE_g64 passes the size target by only 0.28 points.

### Near misses

- **INT8_SYM_per_tensor:** 74.98% reduction and 0.46 pp loss, but logit RMSE
  0.327 is over the limit.
- **INT8_AFFINE_per_tensor:** 74.98% reduction, passes the original 0.3 limit
  (0.233) but not 0.2.
- **INT8_AFFINE_g32:** excellent accuracy and logits, but only 65.59% reduction
  (92.17 MB) because of the extra zero-point storage per group.
- **INT4_AFFINE_g64 / g32:** 82.78% and 78.09% reduction, passing at 0.3 but not
  at 0.2 (logit RMSE 0.290 and 0.259). They are the best choice if the logit
  limit is relaxed.
- **All INT2 configurations fail.** Most collapse to roughly chance-level
  accuracy; the best (INT2_AFFINE_g32) still loses 5.39 pp.

### Why `INT8_AFFINE_per_channel_axis0` was selected

The three targets are pass/fail, so the choice among the nine rests on safety
margin and practicality:

1. **Zero accuracy loss and the best compression among the zero-loss configs.**
   Four configs lose 0.00 pp (INT8_AFFINE_per_channel_axis0, INT8_SYM_g64,
   INT8_SYM_g32, INT8_AFFINE_g64); this one is the smallest at 67.92 MB.
2. **Comfortable margin on every limit:** 4.6 points above the size target and
   about 5x below the logit limit.
3. **Identical predictions to FP32** on the validation set (see below).
4. **Per-channel quantization is widely supported** by inference kernels and is
   simpler to deploy than group-wise schemes.

The grouped configs (INT8_SYM_g64, INT8_AFFINE_g64) have lower logit RMSE, but
give up 1 to 4 points of size reduction. Accuracy differences between the nine
are within a couple of examples (see Limitations).

## Prediction-level consistency

Accuracy is an average, so a model can keep the same accuracy while changing
individual predictions. `prediction_consistency.py` compares each quantized model
with FP32 example by example and reports:

- **flips:** predictions that differ from FP32
- **harmful flips:** FP32 correct, quantized wrong
- **helpful flips:** FP32 wrong, quantized correct
- **masked flips:** flips that cancel in the accuracy number
- **FP32 margin:** how confident FP32 was on the examples that flipped

It runs two groups separately, and never mixes them:

- the **passing** configs (`prediction_consistency_passing/`)
- the **5 worst** configs by accuracy loss (`prediction_consistency_worst/`)

The worst group shows what failure looks like: collapsed models predict a single
class for nearly every example, producing hundreds of flips instead of a few
cancelling ones.

### Selected configuration

`quantized_deviation_analysis.py` writes a per-example comparison
(`prediction_comparison.csv`) for `INT8_AFFINE_per_channel_axis0`. On all 872
examples the quantized model gives the same prediction as FP32: no right-to-wrong
flips, no wrong-to-right flips, and the 78 examples FP32 gets wrong stay wrong.
The largest per-example logit shift is about 0.5 (RMSE over the two logits), on an
example where FP32 was barely confident, and it still did not cross the decision
boundary.

## Repository layout

```
.
├── build_all_artifact.py            # build and size every quantized artifact
├── load_quantized_model.py          # load a quantized artifact for inference
├── quantized_deviation_analysis.py  # weight + prediction deviation for the winner
├── prediction_consistency.py        # flip analysis: passing group and worst group
└── experiments/
    └── results/
        ├── granularity_comparison.csv       # all 38 configs (estimated sizes)
        ├── artifact_comparison.csv          # built artifacts (measured sizes)
        ├── prediction_consistency_passing/  # CSVs + plots, passing configs
        ├── prediction_consistency_worst/    # CSVs + plots, worst configs
        ├── best_performers/                 # deviation analysis per config
        └── plots/                           # error grids and other figures
```

Model artifacts (`*.pt`) are not committed because of their size. Rebuild them
with `build_all_artifact.py`.

## Reproducing

Run from the project root. Adjust script names and paths to match your layout.

```bash
pip install torch transformers datasets matplotlib numpy

python build_all_artifact.py             # build artifacts, write artifact_comparison.csv
python quantized_deviation_analysis.py   # deviation analysis of the selected config
python prediction_consistency.py         # flip analysis, passing and worst groups
```

Thresholds, the number of worst configs, and the candidate lists are constants at
the top of each script.

## Limitations

- **Small validation set.** With 872 examples, one example is about 0.115 pp.
  Differences such as 0.00 vs 0.23 pp are two examples and are not strong
  evidence of a real accuracy difference. Logit RMSE and flip counts are more
  reliable signals.
- **Weight-only, simulated quantization.** Weights are dequantized to FP32 at load
  time, so there is no integer-kernel speedup to measure. Reported timings are
  single CPU passes and are **not** a latency benchmark.
- **Sizes** in `granularity_comparison.csv` are estimates and differ slightly from
  the measured sizes in `artifact_comparison.csv` (for example 74.78% estimated vs
  74.64% measured for the selected config). The tables above use the measured values..
- **No calibration or fine-tuning.** Everything is plain post-training
  quantization with no data-dependent calibration.

"""
Weight-deviation and prediction analysis for the BEST PERFORMING quantized
configurations.

Best performers = configs in artifact_comparison.csv that meet all three limits

    actual_reduction_pct >= MIN_REDUCTION_PCT   (70 %)
    accuracy_loss_pp     <= MAX_ACC_LOSS_PP     (1 pp)
    logit_RMSE           <  MAX_LOGIT_RMSE      (0.2)

ranked by accuracy loss (lowest first), then logit RMSE (lowest first).
The TOP_N best are analyzed, each in its own folder:

    experiments/results/best_performers/<candidate>/
        quantized_deviation_summary.json
        per_tensor_weight_deviation.csv
        weight_error_examples.csv
        prediction_comparison.csv
        plots/weight_reconstruction_dashboard.png
        plots/top_tensor_rmse_detailed.png
        plots/sample_weight_comparison_20.png

    experiments/results/best_performers/best_performers_summary.csv
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.patches import FancyArrowPatch, Rectangle
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    MODEL_ID,
    MAX_LENGTH,
    BATCH_SIZE,
    RESULTS_PATH,
    decode_tensor,
)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_ROOT = RESULTS_DIR / "best_performers"

# Passing conditions (applied to artifact_comparison.csv)
MIN_REDUCTION_PCT = 70.0   # actual_reduction_pct >= this
MAX_ACC_LOSS_PP = 1.0      # accuracy_loss_pp     <= this
MAX_LOGIT_RMSE = 0.2       # logit_RMSE           <  this

# How many of the passing configs to analyze (best first).
# None -> analyze every passing config.
TOP_N = None

# None -> choose automatically from the CSV (all configs that pass).
# Or list candidate names explicitly to override the filter, e.g. the 9 below:
#   CANDIDATES = [
#       "INT8_SYM_per_channel_axis1",
#       "INT8_SYM_per_channel_axis0",
#       "INT8_AFFINE_per_channel_axis1",
#       "INT8_AFFINE_per_channel_axis0",
#       "INT8_SYM_per_group_g128",
#       "INT8_SYM_per_group_g64",
#       "INT8_AFFINE_per_group_g128",
#       "INT8_SYM_per_group_g32",
#       "INT8_AFFINE_per_group_g64",
#   ]
CANDIDATES = None

TOP_K_ERRORS_PER_TENSOR = 10


# ----------------------------------------------------------------------
# Candidate selection
# ----------------------------------------------------------------------

def select_best_performers():
    with RESULTS_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if CANDIDATES:
        known = {row["candidate"] for row in rows}
        for name in CANDIDATES:
            if name not in known:
                print(f"WARNING: {name} not found in {RESULTS_PATH.name}")
        return list(CANDIDATES)

    passing = []
    for row in rows:
        reduction = float(row["actual_reduction_pct"])
        acc_loss = float(row["accuracy_loss_pp"])
        logit = float(row["logit_RMSE"])

        if (
            reduction >= MIN_REDUCTION_PCT
            and acc_loss <= MAX_ACC_LOSS_PP
            and logit < MAX_LOGIT_RMSE
        ):
            passing.append((acc_loss, logit, reduction, row["candidate"]))

    if not passing:
        raise RuntimeError(
            f"No configurations in {RESULTS_PATH} meet the thresholds."
        )

    passing.sort(key=lambda item: (item[0], item[1], -item[2], item[3]))

    print(
        f"{len(passing)} of {len(rows)} configurations pass "
        f"(reduction >= {MIN_REDUCTION_PCT}%, accuracy loss <= "
        f"{MAX_ACC_LOSS_PP} pp, logit RMSE < {MAX_LOGIT_RMSE})."
    )
    print("Ranking (accuracy loss, then logit RMSE):")
    for rank, (loss, logit, reduction, name) in enumerate(passing, start=1):
        marker = "*" if TOP_N is None or rank <= TOP_N else " "
        print(
            f" {marker} {rank:>2}. {name:<32} loss {loss:>5.2f} pp   "
            f"logit RMSE {logit:.3f}   reduction {reduction:.2f}%"
        )

    chosen = passing if TOP_N is None else passing[:TOP_N]
    return [item[3] for item in chosen]


# ----------------------------------------------------------------------
# Evaluation and weight comparison
# ----------------------------------------------------------------------

def evaluate(model, inputs, labels):
    """Evaluate the entire labeled dataset and retain logits."""
    model.eval()
    all_logits = []

    with torch.inference_mode():
        for start in range(0, len(labels), BATCH_SIZE):
            batch = {
                name: tensor[start:start + BATCH_SIZE]
                for name, tensor in inputs.items()
            }
            logits = model(**batch).logits
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    predictions = logits.argmax(dim=-1)
    accuracy = predictions.eq(labels_tensor).float().mean().item()

    return accuracy, logits, predictions, labels_tensor


def compare_weights(original_state, restored_state):
    """
    Measure reconstruction error between the original FP32 weights and the
    weights reconstructed after quantization + dequantization.

    Returns: global_metrics, per_tensor_rows, error_examples
    """
    if original_state.keys() != restored_state.keys():
        raise AssertionError("Original and reconstructed state dicts differ.")

    total_squared_error = 0.0
    total_squared_original = 0.0
    total_absolute_error = 0.0
    total_elements = 0
    maximum_absolute_error = 0.0

    per_tensor_rows = []
    error_examples = []

    for name, original in original_state.items():
        restored = restored_state[name]

        if original.shape != restored.shape:
            raise AssertionError(f"Shape mismatch in {name}")

        if not original.is_floating_point():
            if not torch.equal(original, restored):
                raise AssertionError(f"Non-floating tensor changed: {name}")
            continue

        original = original.detach().cpu().to(torch.float64)
        restored = restored.detach().cpu().to(torch.float64)

        difference = restored - original
        absolute_difference = difference.abs()

        squared_error = difference.square().sum().item()
        absolute_error = absolute_difference.sum().item()
        squared_original = original.square().sum().item()
        count = original.numel()

        total_squared_error += squared_error
        total_absolute_error += absolute_error
        total_squared_original += squared_original
        total_elements += count

        tensor_max_error = absolute_difference.max().item()
        maximum_absolute_error = max(maximum_absolute_error, tensor_max_error)

        per_tensor_rows.append({
            "tensor": name,
            "num_elements": count,
            "original_RMS": (squared_original / count) ** 0.5,
            "MAE": absolute_error / count,
            "RMSE": (squared_error / count) ** 0.5,
            "relative_RMSE": (
                (squared_error / squared_original) ** 0.5
                if squared_original > 0 else 0.0
            ),
            "max_absolute_error": tensor_max_error,
        })

        flat_original = original.flatten()
        flat_restored = restored.flatten()
        flat_difference = difference.flatten()
        flat_absolute = absolute_difference.flatten()

        k = min(TOP_K_ERRORS_PER_TENSOR, flat_absolute.numel())
        top_values, top_indices = torch.topk(flat_absolute, k=k)

        for rank in range(k):
            index = top_indices[rank].item()
            error_examples.append({
                "tensor": name,
                "flat_index": index,
                "original_weight": flat_original[index].item(),
                "reconstructed_weight": flat_restored[index].item(),
                "error": flat_difference[index].item(),
                "absolute_error": top_values[rank].item(),
            })

    global_metrics = {
        "floating_point_parameter_count": total_elements,
        "weight_MAE": total_absolute_error / total_elements,
        "weight_RMSE": (total_squared_error / total_elements) ** 0.5,
        "relative_weight_RMSE": (
            (total_squared_error / total_squared_original) ** 0.5
            if total_squared_original > 0 else 0.0
        ),
        "maximum_absolute_weight_error": maximum_absolute_error,
    }

    return global_metrics, per_tensor_rows, error_examples


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_comprehensive_dashboard(candidate, original_state, restored_state,
                                 per_tensor_rows, plots_dir):
    """4-panel dashboard of weight reconstruction errors."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(candidate, fontsize=15, fontweight="bold")

    # Panel 1: global error distribution
    errors = []
    for name, original in original_state.items():
        if not original.is_floating_point():
            continue
        diff = (
            restored_state[name].detach().cpu().to(torch.float64)
            - original.detach().cpu().to(torch.float64)
        )
        errors.append(diff.flatten().numpy())

    flat_errors = np.concatenate(errors)

    axes[0, 0].hist(flat_errors, bins=120, color="royalblue",
                    edgecolor="none", alpha=0.85)
    axes[0, 0].axvline(0, color="crimson", linestyle="--", linewidth=1.5,
                       label="Zero Error")
    axes[0, 0].set_title("Global Weight Reconstruction Error Distribution",
                         fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Reconstructed Weight - Original FP32 Weight")
    axes[0, 0].set_ylabel("Parameter Count")
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: original vs reconstructed scatter
    selected_name = max(
        [n for n, t in original_state.items() if t.is_floating_point()],
        key=lambda n: original_state[n].numel(),
    )
    orig_flat = original_state[selected_name].detach().cpu().to(
        torch.float64).flatten().numpy()
    rest_flat = restored_state[selected_name].detach().cpu().to(
        torch.float64).flatten().numpy()

    axes[0, 1].scatter(orig_flat, rest_flat, s=3, alpha=0.35,
                       color="darkorange")
    lo = min(orig_flat.min(), rest_flat.min())
    hi = max(orig_flat.max(), rest_flat.max())
    axes[0, 1].plot([lo, hi], [lo, hi], color="black", linestyle="--",
                    linewidth=1.2, label="Ideal (y=x)")
    axes[0, 1].set_title(f"Original vs Reconstructed Weights\n({selected_name})",
                         fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Original FP32 Weight")
    axes[0, 1].set_ylabel("Reconstructed Weight")
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: RMSE by module type
    module_groups = {
        "embeddings": [], "attention": [],
        "intermediate/output": [], "classifier/other": [],
    }
    for row in per_tensor_rows:
        t_name = row["tensor"]
        if "embeddings" in t_name:
            module_groups["embeddings"].append(row["RMSE"])
        elif "attention" in t_name or "self" in t_name:
            module_groups["attention"].append(row["RMSE"])
        elif "intermediate" in t_name or "output" in t_name:
            module_groups["intermediate/output"].append(row["RMSE"])
        else:
            module_groups["classifier/other"].append(row["RMSE"])

    group_names = list(module_groups.keys())
    group_avg_rmse = [
        np.mean(vals) if vals else 0.0 for vals in module_groups.values()
    ]

    bars = axes[1, 0].bar(
        group_names, group_avg_rmse,
        color=["forestgreen", "mediumpurple", "indianred", "goldenrod"],
        alpha=0.85,
    )
    axes[1, 0].set_title("Average Weight Reconstruction RMSE by Module",
                         fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Mean RMSE")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        axes[1, 0].annotate(
            f"{height:.5f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    # Panel 4: tensor size vs relative RMSE
    num_elements = [row["num_elements"] for row in per_tensor_rows]
    rel_rmses = [row["relative_RMSE"] for row in per_tensor_rows]

    scatter = axes[1, 1].scatter(
        num_elements, rel_rmses, c=rel_rmses, cmap="viridis", s=40,
        edgecolors="k", linewidth=0.5, alpha=0.8,
    )
    axes[1, 1].set_title("Tensor Element Count vs Relative RMSE",
                         fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Number of Elements in Tensor")
    axes[1, 1].set_ylabel("Relative RMSE")
    axes[1, 1].set_xscale("log")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    fig.colorbar(scatter, ax=axes[1, 1], label="Relative RMSE")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = plots_dir / "weight_reconstruction_dashboard.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_top_tensor_rmse_detailed(candidate, per_tensor_rows, plots_dir,
                                  top_n=15):
    """Horizontal breakdown of the top tensors by RMSE."""
    sorted_rows = sorted(
        per_tensor_rows, key=lambda row: row["RMSE"], reverse=True,
    )[:top_n]
    sorted_rows.reverse()

    names = [
        row["tensor"].replace("transformer.encoder.layer.", "L.")
        for row in sorted_rows
    ]
    values = [row["RMSE"] for row in sorted_rows]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, values, color="teal", alpha=0.85)

    plt.xlabel("Reconstruction RMSE", fontsize=11, fontweight="bold")
    plt.ylabel("Tensor Name", fontsize=11, fontweight="bold")
    plt.title(
        f"Top {top_n} Tensors with Highest Weight Reconstruction RMSE\n"
        f"{candidate}",
        fontsize=13, fontweight="bold",
    )
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + (max(values) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.5f}", va="center", ha="left",
            fontsize=9, fontweight="semibold",
        )

    plt.tight_layout()
    output_path = plots_dir / "top_tensor_rmse_detailed.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def _extract_int_block(candidate, entry, original_2d):
    """
    Find the stored integer tensor in the packed artifact entry (any integer
    tensor with the same number of elements as the weight).

    The recompute fallback exists only for INT8 affine per-channel axis 0,
    which is the one scheme it reproduces exactly. For every other config it
    returns None rather than showing misleading numbers.
    """
    if isinstance(entry, dict):
        for value in entry.values():
            if (
                torch.is_tensor(value)
                and value.dtype in (torch.int8, torch.uint8, torch.int16)
                and value.numel() == original_2d.numel()
            ):
                return value.reshape(original_2d.shape).to(torch.int64)

    if candidate == "INT8_AFFINE_per_channel_axis0":
        w = original_2d.to(torch.float64)
        w_min = w.min(dim=1, keepdim=True).values.clamp(max=0)
        w_max = w.max(dim=1, keepdim=True).values.clamp(min=0)
        scale = ((w_max - w_min) / 255).clamp(min=1e-12)
        zero_point = torch.round(-128 - w_min / scale)
        q = torch.clamp(torch.round(w / scale) + zero_point, -128, 127)
        return q.to(torch.int64)

    return None


def _draw_value_grid(ax, values, colors, fmt, title):
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 1, 1,
                facecolor=colors[r][c], edgecolor="white", linewidth=2.5,
            ))
            ax.text(
                c + 0.5, rows - 1 - r + 0.5, fmt(values[r, c]),
                ha="center", va="center",
                color="white", fontsize=12, fontweight="bold",
            )
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)


def plot_sample_weight_comparison(candidate, original_state, restored_state,
                                  packed_artifact, plots_dir,
                                  rows=4, cols=5):
    """
    FP32 -> quantize -> integer -> dequantize -> FP32 grid comparison for
    rows*cols selected weights. Returns None if the integer values are not
    available for this config.
    """
    selected_name = max(
        [
            name for name, t in original_state.items()
            if t.is_floating_point() and t.ndim >= 2
        ],
        key=lambda n: original_state[n].numel(),
    )

    original = original_state[selected_name].detach().cpu().to(torch.float64)
    restored = restored_state[selected_name].detach().cpu().to(torch.float64)

    original = original.reshape(original.shape[0], -1)
    restored = restored.reshape(restored.shape[0], -1)

    int_full = _extract_int_block(
        candidate, packed_artifact[selected_name], original,
    )
    if int_full is None:
        print(
            "  Sample-weight plot skipped: no stored integer tensor found "
            f"for {selected_name} in this artifact."
        )
        return None

    bit_label = candidate.split("_")[0]  # e.g. INT8

    r0 = original.shape[0] // 2
    c0 = original.shape[1] // 2
    window = (slice(r0, r0 + rows), slice(c0, c0 + cols))

    orig_block = original[window].numpy()
    rest_block = restored[window].numpy()
    int_block = int_full[window].numpy()

    palette = ["#4a6cf7", "#e3b05a", "#4cae7a", "#c46bb5", "#e07a5f", "#5bc0de"]
    unique, counts = np.unique(int_block, return_counts=True)
    repeated = [int(u) for u, n in zip(unique, counts) if n > 1 and u != 0]
    value_color = {v: palette[i % len(palette)] for i, v in enumerate(repeated)}

    colors = [
        [
            "black" if int_block[r, c] == 0
            else value_color.get(int(int_block[r, c]), "#8c8c8c")
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    _draw_value_grid(axes[0], orig_block, colors, lambda v: f"{v:.4f}", "FP32")
    _draw_value_grid(axes[1], int_block, colors, lambda v: f"{int(v)}", bit_label)
    _draw_value_grid(axes[2], rest_block, colors, lambda v: f"{v:.4f}",
                     "FP32 (reconstructed)")

    for x_start, x_end, label in [(0.335, 0.385, "quantize"),
                                  (0.645, 0.695, "dequantize")]:
        fig.add_artist(
            FancyArrowPatch(
                (x_start, 0.48), (x_end, 0.48),
                transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=22,
                color="#444444", linewidth=2,
            )
        )
        fig.text((x_start + x_end) / 2, 0.53, label,
                 ha="center", va="bottom", fontsize=15)

    abs_err = np.abs(rest_block - orig_block)
    fig.text(
        0.5, 0.04,
        f"{candidate}  |  {selected_name}  |  {rows * cols} weights  |  "
        f"mean |error| = {abs_err.mean():.6f}, "
        f"max |error| = {abs_err.max():.6f}",
        ha="center", fontsize=11, color="#555555",
    )

    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.35,
                        top=0.82, bottom=0.12)

    output_path = plots_dir / f"sample_weight_comparison_{rows * cols}.png"
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)

    return output_path


# ----------------------------------------------------------------------
# One candidate, end to end
# ----------------------------------------------------------------------

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_candidate(candidate, quantized, original_state, inputs, labels,
                      baseline_accuracy, baseline_logits, baseline_preds,
                      true_labels):
    artifact_path = ARTIFACT_DIR / f"{candidate}.pt"
    if not artifact_path.is_file():
        print(f"  SKIPPED: artifact not found: {artifact_path}")
        return None

    out_dir = OUTPUT_ROOT / candidate
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    packed_artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=True,
    )
    restored_state = {
        name: decode_tensor(entry) for name, entry in packed_artifact.items()
    }

    # Weight reconstruction
    weight_metrics, per_tensor_rows, error_examples = compare_weights(
        original_state, restored_state,
    )
    write_csv(out_dir / "per_tensor_weight_deviation.csv", per_tensor_rows)
    write_csv(out_dir / "weight_error_examples.csv", error_examples)

    # Evaluate the quantized model
    quantized.load_state_dict(restored_state, strict=True)
    quantized_accuracy, quantized_logits, quantized_preds, _ = evaluate(
        quantized, inputs, labels,
    )

    difference = quantized_logits.to(torch.float64) - baseline_logits.to(
        torch.float64)
    logit_rmse = difference.square().mean().sqrt().item()
    logit_mae = difference.abs().mean().item()
    max_logit_error = difference.abs().max().item()

    changed = baseline_preds.ne(quantized_preds)
    baseline_correct = baseline_preds.eq(true_labels)
    quantized_correct = quantized_preds.eq(true_labels)

    prediction_rows = []
    for i in range(len(labels)):
        prediction_rows.append({
            "example_index": i,
            "true_label": labels[i],
            "fp32_prediction": baseline_preds[i].item(),
            "quantized_prediction": quantized_preds[i].item(),
            "prediction_changed": bool(changed[i].item()),
            "fp32_correct": bool(baseline_correct[i].item()),
            "quantized_correct": bool(quantized_correct[i].item()),
            "fp32_negative_logit": baseline_logits[i, 0].item(),
            "fp32_positive_logit": baseline_logits[i, 1].item(),
            "quantized_negative_logit": quantized_logits[i, 0].item(),
            "quantized_positive_logit": quantized_logits[i, 1].item(),
            "logit_RMSE_for_example":
                difference[i].square().mean().sqrt().item(),
        })
    write_csv(out_dir / "prediction_comparison.csv", prediction_rows)

    # Plots
    dashboard_path = plot_comprehensive_dashboard(
        candidate, original_state, restored_state, per_tensor_rows, plots_dir,
    )
    detailed_rmse_path = plot_top_tensor_rmse_detailed(
        candidate, per_tensor_rows, plots_dir, top_n=15,
    )
    sample_path = plot_sample_weight_comparison(
        candidate, original_state, restored_state, packed_artifact,
        plots_dir, rows=4, cols=5,
    )

    summary = {
        "candidate": candidate,
        "dataset": "GLUE SST-2 validation",
        "examples": len(labels),
        "baseline_accuracy_pct": baseline_accuracy * 100,
        "quantized_accuracy_pct": quantized_accuracy * 100,
        "accuracy_change_pp": (quantized_accuracy - baseline_accuracy) * 100,
        "changed_predictions": int(changed.sum().item()),
        "changed_prediction_pct": changed.float().mean().item() * 100,
        "both_correct": int((baseline_correct & quantized_correct).sum().item()),
        "fp32_correct_quantized_wrong": int(
            (baseline_correct & ~quantized_correct).sum().item()),
        "fp32_wrong_quantized_correct": int(
            (~baseline_correct & quantized_correct).sum().item()),
        "both_wrong": int((~baseline_correct & ~quantized_correct).sum().item()),
        "logit_RMSE": logit_rmse,
        "logit_MAE": logit_mae,
        "maximum_absolute_logit_error": max_logit_error,
        **weight_metrics,
    }

    with (out_dir / "quantized_deviation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        f"  accuracy {quantized_accuracy * 100:.4f}% "
        f"({summary['accuracy_change_pp']:+.4f} pp) | "
        f"changed predictions {summary['changed_predictions']}/{len(labels)} | "
        f"logit RMSE {logit_rmse:.6f} | "
        f"weight RMSE {weight_metrics['weight_RMSE']:.6f}"
    )
    print(f"  saved to {out_dir}")
    for path in (dashboard_path, detailed_rmse_path, sample_path):
        if path is not None:
            print(f"    {path.name}")

    del packed_artifact, restored_state
    return summary


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = select_best_performers()
    print(f"\nAnalyzing {len(candidates)} best-performing configuration(s):")
    for name in candidates:
        print(f"  {name}")

    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    print("\nLoading FP32 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    baseline = AutoModelForSequenceClassification.from_pretrained(source)
    baseline.to("cpu")
    baseline.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in baseline.state_dict().items()
    }

    # Separate model that receives each candidate's decoded weights
    quantized = AutoModelForSequenceClassification.from_pretrained(source)
    quantized.to("cpu")
    quantized.eval()

    print("Loading the labeled SST-2 validation split...")
    dataset = load_dataset(
        "nyu-mll/glue", "sst2", split="validation",
    ).shuffle(seed=42)

    texts = [str(text) for text in dataset["sentence"]]
    labels = [int(label) for label in dataset["label"]]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print(f"Evaluating FP32 baseline on all {len(labels)} examples...")
    baseline_accuracy, baseline_logits, baseline_preds, true_labels = evaluate(
        baseline, inputs, labels,
    )
    print(f"FP32 accuracy: {baseline_accuracy * 100:.4f}%")

    summaries = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)}] {candidate}")
        summary = analyze_candidate(
            candidate, quantized, original_state, inputs, labels,
            baseline_accuracy, baseline_logits, baseline_preds, true_labels,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No artifacts could be analyzed.")

    write_csv(OUTPUT_ROOT / "best_performers_summary.csv", summaries)

    print("\nComparison")
    header = (
        f"{'config':<32}{'acc %':>9}{'chg pp':>9}{'changed':>9}"
        f"{'logit RMSE':>12}{'weight RMSE':>13}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['candidate']:<32}"
            f"{s['quantized_accuracy_pct']:>9.3f}"
            f"{s['accuracy_change_pp']:>+9.3f}"
            f"{s['changed_predictions']:>9d}"
            f"{s['logit_RMSE']:>12.6f}"
            f"{s['weight_RMSE']:>13.6f}"
        )

    print(f"\nAll results saved under: {OUTPUT_ROOT}")
    print("  best_performers_summary.csv")


if __name__ == "__main__":
    main()
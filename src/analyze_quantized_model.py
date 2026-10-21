import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    MODEL_ID,
    MAX_LENGTH,
    BATCH_SIZE,
    decode_tensor,
)
from load_quantized_model import load_quantized_model


CANDIDATE = "INT8_AFFINE_per_channel_axis0"
ARTIFACT_PATH = ARTIFACT_DIR / f"{CANDIDATE}.pt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SUMMARY_PATH = RESULTS_DIR / "quantized_deviation_summary.json"
WEIGHTS_CSV_PATH = RESULTS_DIR / "per_tensor_weight_deviation.csv"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "prediction_comparison.csv"
ERROR_EXAMPLES_CSV_PATH = RESULTS_DIR / "weight_error_examples.csv"

def evaluate(model, inputs, labels):
    """
    Evaluate the entire labeled dataset and retain logits.
    """
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

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    predictions = logits.argmax(dim=-1)

    accuracy = (
        predictions.eq(labels_tensor)
        .float()
        .mean()
        .item()
    )

    return (
        accuracy,
        logits,
        predictions,
        labels_tensor,
    )

def compare_weights(original_state, restored_state):
    """
    Measure reconstruction error between the original FP32
    weights and the reconstructed weights obtained after
    quantization + dequantization.

    Returns:
        global_metrics
        per_tensor_rows
        error_examples
    """

    if original_state.keys() != restored_state.keys():
        raise AssertionError(
            "Original and reconstructed state dicts differ."
        )

    total_squared_error = 0.0
    total_squared_original = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    maximum_absolute_error = 0.0

    per_tensor_rows = []
    error_examples = []

    TOP_K_ERRORS_PER_TENSOR = 10

    for name, original in original_state.items():
        restored = restored_state[name]

        if original.shape != restored.shape:
            raise AssertionError(
                f"Shape mismatch in {name}"
            )

        if not original.is_floating_point():
            if not torch.equal(original, restored):
                raise AssertionError(
                    f"Non-floating tensor changed: {name}"
                )
            continue

        original = (
            original
            .detach()
            .cpu()
            .to(torch.float64)
        )

        restored = (
            restored
            .detach()
            .cpu()
            .to(torch.float64)
        )

        difference = restored - original
        absolute_difference = difference.abs()

        squared_error = (
            difference.square()
            .sum()
            .item()
        )

        absolute_error = (
            absolute_difference
            .sum()
            .item()
        )

        squared_original = (
            original.square()
            .sum()
            .item()
        )

        count = original.numel()

        total_squared_error += squared_error
        total_absolute_error += absolute_error
        total_squared_original += squared_original
        total_elements += count

        tensor_rmse = (
            squared_error / count
        ) ** 0.5

        tensor_mae = (
            absolute_error / count
        )

        tensor_relative_rmse = (
            (squared_error / squared_original) ** 0.5
            if squared_original > 0
            else 0.0
        )

        tensor_max_error = (
            absolute_difference.max().item()
        )

        maximum_absolute_error = max(
            maximum_absolute_error,
            tensor_max_error,
        )

        per_tensor_rows.append({
            "tensor": name,
            "num_elements": count,
            "original_RMS": (
                squared_original / count
            ) ** 0.5,
            "MAE": tensor_mae,
            "RMSE": tensor_rmse,
            "relative_RMSE": tensor_relative_rmse,
            "max_absolute_error": tensor_max_error,
        })

        flat_original = original.flatten()
        flat_restored = restored.flatten()
        flat_difference = difference.flatten()
        flat_absolute = absolute_difference.flatten()

        k = min(
            TOP_K_ERRORS_PER_TENSOR,
            flat_absolute.numel(),
        )

        top_values, top_indices = torch.topk(
            flat_absolute,
            k=k,
        )

        for rank in range(k):
            index = top_indices[rank].item()

            error_examples.append({
                "tensor": name,
                "flat_index": index,
                "original_weight": (
                    flat_original[index].item()
                ),
                "reconstructed_weight": (
                    flat_restored[index].item()
                ),
                "error": (
                    flat_difference[index].item()
                ),
                "absolute_error": (
                    top_values[rank].item()
                ),
            })

    global_rmse = (
        total_squared_error / total_elements
    ) ** 0.5

    global_mae = (
        total_absolute_error / total_elements
    )

    global_relative_rmse = (
        (
            total_squared_error
            / total_squared_original
        ) ** 0.5
        if total_squared_original > 0
        else 0.0
    )

    global_metrics = {
        "floating_point_parameter_count": total_elements,
        "weight_MAE": global_mae,
        "weight_RMSE": global_rmse,
        "relative_weight_RMSE": global_relative_rmse,
        "maximum_absolute_weight_error": maximum_absolute_error,
    }

    return (
        global_metrics,
        per_tensor_rows,
        error_examples,
    )

def plot_comprehensive_dashboard(original_state, restored_state, per_tensor_rows):
    """
    Generate a comprehensive 4-panel analytical dashboard detailing
    weight reconstruction errors, correlations, layer groupings, and scaling behavior.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Global Error Distribution (Top-Left)
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
    
    axes[0, 0].hist(flat_errors, bins=120, color='royalblue', edgecolor='none', alpha=0.85)
    axes[0, 0].axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='Zero Error')
    axes[0, 0].set_title("Global Weight Reconstruction Error Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Reconstructed Weight - Original FP32 Weight")
    axes[0, 0].set_ylabel("Parameter Count")
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Original vs Reconstructed Scatter (Top-Right)
    selected_name = max(
        [name for name, t in original_state.items() if t.is_floating_point()],
        key=lambda n: original_state[n].numel()
    )
    
    orig_flat = original_state[selected_name].detach().cpu().to(torch.float64).flatten().numpy()
    rest_flat = restored_state[selected_name].detach().cpu().to(torch.float64).flatten().numpy()
    
    axes[0, 1].scatter(orig_flat, rest_flat, s=3, alpha=0.35, color='darkorange')
    min_val = min(orig_flat.min(), rest_flat.min())
    max_val = max(orig_flat.max(), rest_flat.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.2, label='Ideal (y=x)')
    axes[0, 1].set_title(f"Original vs Reconstructed Weights\n({selected_name})", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Original FP32 Weight")
    axes[0, 1].set_ylabel("Reconstructed Weight")
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: RMSE by Functional Module Type (Bottom-Left)
    module_groups = {"embeddings": [], "attention": [], "intermediate/output": [], "classifier/other": []}
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
    group_avg_rmse = [np.mean(vals) if vals else 0.0 for vals in module_groups.values()]
    
    bars = axes[1, 0].bar(group_names, group_avg_rmse, color=['forestgreen', 'mediumpurple', 'indianred', 'goldenrod'], alpha=0.85)
    axes[1, 0].set_title("Average Weight Reconstruction RMSE by Module", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Mean RMSE")
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.5f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # Panel 4: Tensor Size vs Relative RMSE (Bottom-Right)
    num_elements = [row["num_elements"] for row in per_tensor_rows]
    rel_rmses = [row["relative_RMSE"] for row in per_tensor_rows]
    
    scatter = axes[1, 1].scatter(num_elements, rel_rmses, c=rel_rmses, cmap='viridis', s=40, edgecolors='k', linewidth=0.5, alpha=0.8)
    axes[1, 1].set_title("Tensor Element Count vs Relative RMSE", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Number of Elements in Tensor")
    axes[1, 1].set_ylabel("Relative RMSE")
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(scatter, ax=axes[1, 1], label='Relative RMSE')

    plt.tight_layout()
    output_path = PLOTS_DIR / "weight_reconstruction_dashboard.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_top_tensor_rmse_detailed(per_tensor_rows, top_n=15):
    """
    Generate an expanded, highly readable horizontal breakdown of top tensors by RMSE
    with value annotations.
    """
    sorted_rows = sorted(
        per_tensor_rows,
        key=lambda row: row["RMSE"],
        reverse=True,
    )[:top_n]

    sorted_rows.reverse()

    names = [row["tensor"].replace("transformer.encoder.layer.", "L.") for row in sorted_rows]
    values = [row["RMSE"] for row in sorted_rows]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, values, color='teal', alpha=0.85)
    
    plt.xlabel("Reconstruction RMSE", fontsize=11, fontweight='bold')
    plt.ylabel("Tensor Name", fontsize=11, fontweight='bold')
    plt.title(f"Top {top_n} Tensors with Highest Weight Reconstruction RMSE", fontsize=13, fontweight='bold')
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(values) * 0.01), bar.get_y() + bar.get_height()/2,
                 f'{width:.5f}', 
                 va='center', ha='left', fontsize=9, fontweight='semibold')

    plt.tight_layout()
    output_path = PLOTS_DIR / "top_tensor_rmse_detailed.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_sample_weight_comparison(original_state, restored_state, num_samples=50):
    """
    Compare a subset of specific weight values (e.g., 50 sample weights)
    between the original FP32 matrix and the reconstructed INT8 matrix.
    """
    selected_name = max(
        [name for name, t in original_state.items() if t.is_floating_point()],
        key=lambda n: original_state[n].numel()
    )

    orig_flat = original_state[selected_name].detach().cpu().to(torch.float64).flatten().numpy()
    rest_flat = restored_state[selected_name].detach().cpu().to(torch.float64).flatten().numpy()

    # Take a evenly spaced sample of weights to visualize comparison clearly
    indices = np.linspace(0, len(orig_flat) - 1, num_samples, dtype=int)
    sampled_orig = orig_flat[indices]
    sampled_rest = rest_flat[indices]

    plt.figure(figsize=(14, 6))
    x_coords = np.arange(num_samples)

    plt.scatter(x_coords, sampled_orig, color='dodgerblue', label='Original FP32', s=40, zorder=3)
    plt.scatter(x_coords, sampled_rest, color='crimson', marker='x', label='Reconstructed INT8', s=40, zorder=3)

    # Draw vertical connecting lines showing error delta magnitude
    for x, o_val, r_val in zip(x_coords, sampled_orig, sampled_rest):
        plt.plot([x, x], [o_val, r_val], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.xlabel("Sample Weight Index", fontsize=11, fontweight='bold')
    plt.ylabel("Weight Value", fontsize=11, fontweight='bold')
    plt.title(f"Sample Weight Comparison ({num_samples} Weights from {selected_name})", fontsize=13, fontweight='bold')
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_path = PLOTS_DIR / f"sample_weight_comparison_{num_samples}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main():

    torch.manual_seed(42)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Selected artifact not found: "
            f"{ARTIFACT_PATH}"
        )
    print("Loading FP32 baseline...")

    source = (
        str(BASELINE_DIR)
        if BASELINE_DIR.exists()
        else MODEL_ID
    )

    baseline = (
        AutoModelForSequenceClassification
        .from_pretrained(source)
    )

    baseline.to("cpu")
    baseline.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in baseline.state_dict().items()
    }
    print(
        "Loading the selected artifact "
        "through the real loader..."
    )

    tokenizer, quantized = load_quantized_model()

    quantized.to("cpu")
    quantized.eval()

    packed_artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    restored_state = {
        name: decode_tensor(entry)
        for name, entry in packed_artifact.items()
    }
    print(
        "Comparing original and reconstructed weights..."
    )

    (
        weight_metrics,
        per_tensor_rows,
        error_examples,
    ) = compare_weights(
        original_state,
        restored_state,
    )

    with WEIGHTS_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                per_tensor_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(per_tensor_rows)

    with ERROR_EXAMPLES_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                error_examples[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(error_examples)

    print(
        "Loading the labeled SST-2 validation split..."
    )

    dataset = load_dataset(
        "nyu-mll/glue",
        "sst2",
        split="validation",
    ).shuffle(seed=42)

    texts = [
        str(text)
        for text in dataset["sentence"]
    ]

    labels = [
        int(label)
        for label in dataset["label"]
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print(
        f"Evaluating all {len(labels)} examples..."
    )

    (
        baseline_accuracy,
        baseline_logits,
        baseline_preds,
        true_labels,
    ) = evaluate(
        baseline,
        inputs,
        labels,
    )

    (
        quantized_accuracy,
        quantized_logits,
        quantized_preds,
        _,
    ) = evaluate(
        quantized,
        inputs,
        labels,
    )

    difference = (
        quantized_logits.to(torch.float64)
        - baseline_logits.to(torch.float64)
    )

    logit_rmse = (
        difference.square()
        .mean()
        .sqrt()
        .item()
    )

    logit_mae = (
        difference.abs()
        .mean()
        .item()
    )

    max_logit_error = (
        difference.abs()
        .max()
        .item()
    )
    changed = baseline_preds.ne(
        quantized_preds
    )

    baseline_correct = baseline_preds.eq(
        true_labels
    )

    quantized_correct = quantized_preds.eq(
        true_labels
    )
    prediction_rows = []

    for i in range(len(labels)):

        prediction_rows.append({
            "example_index": i,
            "true_label": labels[i],

            "fp32_prediction":
                baseline_preds[i].item(),

            "int8_prediction":
                quantized_preds[i].item(),

            "prediction_changed":
                bool(changed[i].item()),

            "fp32_correct":
                bool(baseline_correct[i].item()),

            "int8_correct":
                bool(quantized_correct[i].item()),

            "fp32_negative_logit":
                baseline_logits[i, 0].item(),

            "fp32_positive_logit":
                baseline_logits[i, 1].item(),

            "int8_negative_logit":
                quantized_logits[i, 0].item(),

            "int8_positive_logit":
                quantized_logits[i, 1].item(),

            "logit_RMSE_for_example":
                difference[i]
                .square()
                .mean()
                .sqrt()
                .item(),
        })

    with PREDICTIONS_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                prediction_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(prediction_rows)

    print("\nGenerating detailed descriptive visualizations...")

    dashboard_path = plot_comprehensive_dashboard(
        original_state,
        restored_state,
        per_tensor_rows,
    )

    detailed_rmse_path = plot_top_tensor_rmse_detailed(
        per_tensor_rows,
        top_n=15,
    )

    sample_comparison_path = plot_sample_weight_comparison(
        original_state,
        restored_state,
        num_samples=50,
    )

    summary = {
        "candidate": CANDIDATE,

        "dataset":
            "GLUE SST-2 validation",

        "examples":
            len(labels),

        "baseline_accuracy_pct":
            baseline_accuracy * 100,

        "quantized_accuracy_pct":
            quantized_accuracy * 100,

        "accuracy_change_pp":
            (
                quantized_accuracy
                - baseline_accuracy
            ) * 100,

        "changed_predictions":
            int(changed.sum().item()),

        "changed_prediction_pct":
            changed.float().mean().item() * 100,

        "both_correct":
            int(
                (
                    baseline_correct
                    & quantized_correct
                ).sum().item()
            ),

        "fp32_correct_int8_wrong":
            int(
                (
                    baseline_correct
                    & ~quantized_correct
                ).sum().item()
            ),

        "fp32_wrong_int8_correct":
            int(
                (
                    ~baseline_correct
                    & quantized_correct
                ).sum().item()
            ),

        "both_wrong":
            int(
                (
                    ~baseline_correct
                    & ~quantized_correct
                ).sum().item()
            ),

        "logit_RMSE":
            logit_rmse,

        "logit_MAE":
            logit_mae,

        "maximum_absolute_logit_error":
            max_logit_error,

        **weight_metrics,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nWeight reconstruction error")

    for key, value in weight_metrics.items():
        print(
            f"{key}: {value}"
        )

    print("\nFull validation evaluation")

    print(
        f"Examples: {len(labels)}"
    )

    print(
        f"FP32 accuracy: "
        f"{baseline_accuracy * 100:.4f}%"
    )

    print(
        f"INT8 accuracy: "
        f"{quantized_accuracy * 100:.4f}%"
    )

    print(
        "Accuracy change: "
        f"{(quantized_accuracy - baseline_accuracy) * 100:+.4f} pp"
    )

    print(
        f"Changed predictions: "
        f"{changed.sum().item()} / {len(labels)}"
    )

    print(
        f"Logit RMSE: "
        f"{logit_rmse:.8f}"
    )

    print(
        f"Logit MAE: "
        f"{logit_mae:.8f}"
    )

    print(
        f"Maximum absolute logit error: "
        f"{max_logit_error:.8f}"
    )

    print("\nSaved reports")

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Per-tensor weight errors: "
        f"{WEIGHTS_CSV_PATH}"
    )

    print(
        f"Top individual weight errors: "
        f"{ERROR_EXAMPLES_CSV_PATH}"
    )

    print(
        f"Per-example predictions: "
        f"{PREDICTIONS_CSV_PATH}"
    )

    print("\nSaved visualizations")

    print(
        f"Comprehensive Analytical Dashboard: {dashboard_path}"
    )

    print(
        f"Detailed Top Tensor RMSE Breakdown: {detailed_rmse_path}"
    )

    print(
        f"Sample Weight Comparison (50 weights): {sample_comparison_path}"
    )


if __name__ == "__main__":
    main()
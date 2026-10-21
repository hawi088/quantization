import csv
import gc
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from model_quantization import quantize_state_dict


MODEL_ID = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
BASELINE_DIR = Path("baseline_model")
RESULTS_DIR = Path("experiments/results")
OUTPUT_PATH = RESULTS_DIR / "granularity_comparison.csv"

BATCH_SIZE = 16
MAX_LENGTH = 128

# Passing conditions
SIZE_REDUCTION_TARGET = 70.0     # was 60.0
MAX_ACCURACY_LOSS_PP = 1.0
MAX_LOGIT_RMSE = 0.2            # new: maximum allowed logit RMSE vs FP32

BIT_WIDTHS = (8, 4, 2)
METHODS = ("SYM", "AFFINE")
GROUP_SIZES = (32, 64, 128)
CHANNEL_AXES = (0, 1)

CSV_FIELDS = [
    "candidate",
    "representation",
    "granularity",
    "channel_axis",
    "group_size",
    "accuracy_pct",
    "accuracy_loss_pp",
    "estimated_weight_MB",
    "estimated_reduction_pct",
    "weight_MAE",
    "weight_RMSE",
    "relative_weight_RMSE",
    "logit_RMSE",
    "evaluation_seconds",
    "ms_per_example",
    "quantized_tensor_count",
    "parameter_set_count",
    "size_pass",
    "accuracy_pass",
    "logit_pass",
    "meets_targets",
]


def build_candidates():
    """Build 2 floating-point references and 36 integer experiments."""
    candidates = [
        {
            "candidate": "FP16",
            "representation": "FP16",
            "granularity": "per_tensor",
            "channel_axis": None,
            "group_size": None,
        },
        {
            "candidate": "BF16",
            "representation": "BF16",
            "granularity": "per_tensor",
            "channel_axis": None,
            "group_size": None,
        },
    ]

    for bits in BIT_WIDTHS:
        for method in METHODS:
            representation = f"INT{bits}_{method}"

            # One global scale/zero-point pair per tensor.
            candidates.append({
                "candidate": f"{representation}_per_tensor",
                "representation": representation,
                "granularity": "per_tensor",
                "channel_axis": None,
                "group_size": None,
            })

            # Independently quantize rows and columns.
            for axis in CHANNEL_AXES:
                candidates.append({
                    "candidate": (
                        f"{representation}_per_channel_axis{axis}"
                    ),
                    "representation": representation,
                    "granularity": "per_channel",
                    "channel_axis": axis,
                    "group_size": None,
                })

            # Independently quantize consecutive flattened groups.
            for group_size in GROUP_SIZES:
                candidates.append({
                    "candidate": (
                        f"{representation}_per_group_g{group_size}"
                    ),
                    "representation": representation,
                    "granularity": "per_group",
                    "channel_axis": None,
                    "group_size": group_size,
                })

    assert len(candidates) == 38, (
        f"Expected 38 configurations, got {len(candidates)}"
    )
    assert len({c["candidate"] for c in candidates}) == 38

    return candidates


def evaluate(model, inputs, labels):
    """Evaluate accuracy, logits, and wall-clock evaluation time."""
    predictions = []
    all_logits = []

    model.eval()
    start_time = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(labels), BATCH_SIZE):
            batch = {
                key: value[start:start + BATCH_SIZE]
                for key, value in inputs.items()
            }

            logits = model(**batch).logits
            all_logits.append(logits.cpu())
            predictions.extend(logits.argmax(dim=-1).tolist())

    elapsed_seconds = time.perf_counter() - start_time

    accuracy = sum(
        prediction == label
        for prediction, label in zip(predictions, labels)
    ) / len(labels)

    return accuracy, torch.cat(all_logits), elapsed_seconds


def save_results(rows):
    """Persist completed experiments, including after each candidate."""
    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()

    print(f"Planned configurations: {len(candidates)}")
    print("  2 floating-point references")
    print("  36 integer quantization experiments")
    print("Device: CPU")
    print("\nPassing conditions:")
    print(f"  Size reduction   >= {SIZE_REDUCTION_TARGET:.1f}%")
    print(f"  Accuracy loss    <= {MAX_ACCURACY_LOSS_PP:.2f} pp")
    print(f"  Logit RMSE       <= {MAX_LOGIT_RMSE:.3f}")

    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    print("\nLoading tokenizer and FP32 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(torch.device("cpu"))
    model.eval()

    # Preserve an untouched FP32 state for every experiment.
    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    baseline_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in original_state.values()
    )

    print("Loading SST-2 validation set...")
    dataset = load_dataset(
        "nyu-mll/glue",
        "sst2",
        split="validation",
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

    print("\nEvaluating FP32 baseline...")
    model.load_state_dict(original_state, strict=True)

    baseline_accuracy, baseline_logits, baseline_seconds = evaluate(
        model, inputs, labels
    )

    print(f"Examples: {len(labels)}")
    print(f"FP32 accuracy: {baseline_accuracy * 100:.2f}%")
    print(
        f"FP32 state-dict storage: "
        f"{baseline_weight_bytes / 1_000_000:.2f} MB"
    )
    print(f"FP32 evaluation time: {baseline_seconds:.2f} seconds")

    rows = []
    save_results(rows)

    for index, config in enumerate(candidates, start=1):
        candidate = config["candidate"]
        representation = config["representation"]
        granularity = config["granularity"]
        channel_axis = config["channel_axis"]
        group_size = config["group_size"]

        print(f"\n[{index}/{len(candidates)}] {candidate}")

        # Supply only the parameter relevant to the chosen granularity.
        quantization_kwargs = {
            "granularity": granularity,
        }

        if granularity == "per_channel":
            quantization_kwargs["channel_axis"] = channel_axis
        elif granularity == "per_group":
            quantization_kwargs["group_size"] = group_size

        # Every candidate starts from the same FP32 weights.
        candidate_state, metrics = quantize_state_dict(
            original_state,
            representation,
            **quantization_kwargs,
        )

        model.load_state_dict(candidate_state, strict=True)

        accuracy, logits, evaluation_seconds = evaluate(
            model, inputs, labels
        )

        accuracy_loss_pp = (
            baseline_accuracy - accuracy
        ) * 100

        size_reduction_pct = (
            1
            - metrics["estimated_weight_bytes"]
            / baseline_weight_bytes
        ) * 100

        logit_difference = (
            baseline_logits.to(torch.float64)
            - logits.to(torch.float64)
        )

        logit_rmse = (
            logit_difference.square().mean().item()
        ) ** 0.5

        size_pass = size_reduction_pct >= SIZE_REDUCTION_TARGET
        accuracy_pass = accuracy_loss_pp <= MAX_ACCURACY_LOSS_PP
        logit_pass = logit_rmse <= MAX_LOGIT_RMSE

        qualifies = size_pass and accuracy_pass and logit_pass

        row = {
            "candidate": candidate,
            "representation": representation,
            "granularity": granularity,
            "channel_axis": channel_axis,
            "group_size": group_size,
            "accuracy_pct": round(accuracy * 100, 4),
            "accuracy_loss_pp": round(accuracy_loss_pp, 4),
            "estimated_weight_MB": round(
                metrics["estimated_weight_bytes"] / 1_000_000, 4
            ),
            "estimated_reduction_pct": round(
                size_reduction_pct, 4
            ),
            "weight_MAE": metrics["weight_mae"],
            "weight_RMSE": metrics["weight_rmse"],
            "relative_weight_RMSE": metrics["relative_weight_rmse"],
            "logit_RMSE": logit_rmse,
            "evaluation_seconds": round(evaluation_seconds, 4),
            "ms_per_example": round(
                evaluation_seconds / len(labels) * 1000, 4
            ),
            "quantized_tensor_count": (
                metrics["quantized_tensor_count"]
            ),
            "parameter_set_count": metrics["parameter_set_count"],
            "size_pass": size_pass,
            "accuracy_pass": accuracy_pass,
            "logit_pass": logit_pass,
            "meets_targets": qualifies,
        }

        rows.append(row)

        # Save every completed result so interruption doesn't erase work.
        save_results(rows)

        print(f"Accuracy: {row['accuracy_pct']:.2f}%")
        print(f"Accuracy loss: {row['accuracy_loss_pp']:.2f} pp")
        print(
            f"Estimated reduction: "
            f"{row['estimated_reduction_pct']:.2f}%"
        )
        print(
            f"Relative weight RMSE: "
            f"{row['relative_weight_RMSE']:.6f}"
        )
        print(f"Logit RMSE: {row['logit_RMSE']:.6f}")
        print(
            f"Evaluation time: "
            f"{row['evaluation_seconds']:.2f} seconds"
        )
        print(
            f"Size pass: {size_pass} | "
            f"Accuracy pass: {accuracy_pass} | "
            f"Logit pass: {logit_pass}"
        )
        print(f"Meets targets: {qualifies}")

        del candidate_state, logits
        gc.collect()

    print("\n--- Experiment matrix complete ---")
    print(f"Completed configurations: {len(rows)}/{len(candidates)}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(
        "Storage reductions are estimates, not actual serialized "
        "artifact measurements."
    )
    print(
        "Evaluation timing is a single CPU evaluation pass, "
        "not a controlled latency benchmark."
    )
    print("No winner is selected automatically.")


if __name__ == "__main__":
    main()
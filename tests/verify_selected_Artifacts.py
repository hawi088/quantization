
import csv
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    BASELINE_PATH,
    BATCH_SIZE,
    MAX_LENGTH,
    MODEL_ID,
    decode_tensor,
    evaluate,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE = "INT8_AFFINE_per_channel_axis0"

ARTIFACT_PATH = ARTIFACT_DIR / f"{CANDIDATE}.pt"
CSV_PATH = PROJECT_ROOT / "experiments" / "results" / "artifact_comparison.csv"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "results" / "selected_int8_config.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv_record():
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        matches = [
            row for row in csv.DictReader(file)
            if row["candidate"] == CANDIDATE
        ]

    if len(matches) != 1:
        raise AssertionError(
            f"Expected one CSV record for {CANDIDATE}; found {len(matches)}"
        )

    return matches[0]


def calculate_weight_errors(original_state, restored_state):
    if original_state.keys() != restored_state.keys():
        raise AssertionError("State-dict parameter names differ.")

    squared_error = 0.0
    squared_original = 0.0
    element_count = 0

    for name, original in original_state.items():
        restored = restored_state[name]

        if original.shape != restored.shape:
            raise AssertionError(
                f"Shape mismatch for {name}: "
                f"{original.shape} vs {restored.shape}"
            )

        if original.is_floating_point():
            original64 = original.to(torch.float64)
            restored64 = restored.to(torch.float64)
            difference = original64 - restored64

            squared_error += difference.square().sum().item()
            squared_original += original64.square().sum().item()
            element_count += original.numel()
        elif not torch.equal(original, restored):
            raise AssertionError(
                f"Non-floating tensor changed: {name}"
            )

    rmse = (squared_error / element_count) ** 0.5
    relative_rmse = (
        (squared_error / squared_original) ** 0.5
        if squared_original else 0.0
    )

    return rmse, relative_rmse


def main():
    for path in (ARTIFACT_PATH, BASELINE_PATH, CSV_PATH, CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)

    csv_record = load_csv_record()

    # 1. Verify artifact identity and recorded file size.
    actual_bytes = ARTIFACT_PATH.stat().st_size
    recorded_bytes = int(csv_record["artifact_bytes"])
    actual_hash = sha256_file(ARTIFACT_PATH)

    assert actual_bytes == recorded_bytes, (
        f"Artifact size mismatch: {actual_bytes} != {recorded_bytes}"
    )
    assert actual_hash == config["artifact"]["sha256"], (
        "Artifact SHA-256 differs from the saved configuration."
    )

    baseline_bytes = BASELINE_PATH.stat().st_size
    reduction_pct = (1 - actual_bytes / baseline_bytes) * 100

    print("=== Artifact integrity ===")
    print(f"Artifact: {ARTIFACT_PATH.name}")
    print(f"Actual size: {actual_bytes:,} bytes")
    print(f"SHA-256: {actual_hash}")
    print(f"Size reduction vs FP32 reference: {reduction_pct:.4f}%")

    # 2. Load the original FP32 reference and packed artifact.
    original_state = torch.load(
        BASELINE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    packed_artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # 3. Reconstruct every tensor using the existing decoder.
    restored_state = {
        name: decode_tensor(entry)
        for name, entry in packed_artifact.items()
    }

    rmse, relative_rmse = calculate_weight_errors(
        original_state, restored_state
    )

    print("\n=== Reconstruction ===")
    print(f"Tensor count: {len(restored_state)}")
    print(f"Weight RMSE: {rmse:.8f}")
    print(f"Relative weight RMSE: {relative_rmse:.8f}")
    print("All tensor names, shapes, and non-floating values verified.")

    # 4. Independently evaluate the baseline and reconstructed model.
    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to("cpu")
    model.eval()

    model.load_state_dict(original_state, strict=True)

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

    baseline_accuracy, baseline_logits, _ = evaluate(
        model, inputs, labels
    )

    model.load_state_dict(restored_state, strict=True)

    restored_accuracy, restored_logits, _ = evaluate(
        model, inputs, labels
    )

    accuracy_loss_pp = (
        baseline_accuracy - restored_accuracy
    ) * 100

    logit_difference = (
        baseline_logits.to(torch.float64)
        - restored_logits.to(torch.float64)
    )
    logit_rmse = logit_difference.square().mean().sqrt().item()
        # Compare actual class predictions and correctness.
    baseline_predictions = baseline_logits.argmax(dim=1)
    restored_predictions = restored_logits.argmax(dim=1)
    true_labels = torch.tensor(labels)

    baseline_correct = baseline_predictions.eq(true_labels)
    restored_correct = restored_predictions.eq(true_labels)

    prediction_changes = baseline_predictions.ne(restored_predictions).sum().item()

    both_correct = (baseline_correct & restored_correct).sum().item()
    fp32_only_correct = (baseline_correct & ~restored_correct).sum().item()
    int8_only_correct = (~baseline_correct & restored_correct).sum().item()
    both_wrong = (~baseline_correct & ~restored_correct).sum().item()

    print("\n=== Prediction-level audit ===")
    print(f"FP32 correct: {baseline_correct.sum().item()} / {len(labels)}")
    print(f"INT8 correct: {restored_correct.sum().item()} / {len(labels)}")
    print(f"Predictions that changed: {prediction_changes} / {len(labels)}")
    print(f"Both correct: {both_correct}")
    print(f"FP32 correct, INT8 wrong: {fp32_only_correct}")
    print(f"FP32 wrong, INT8 correct: {int8_only_correct}")
    print(f"Both wrong: {both_wrong}")
    print("\n=== Independent evaluation ===")
    print(f"Examples: {len(labels)}")
    print(f"FP32 accuracy: {baseline_accuracy * 100:.4f}%")
    print(f"Reconstructed accuracy: {restored_accuracy * 100:.4f}%")
    print(f"Accuracy loss: {accuracy_loss_pp:.4f} percentage points")
    print(f"Logit RMSE: {logit_rmse:.8f}")

    # 5. Check the project's stated acceptance criteria.
    assert reduction_pct >= 60.0, "Storage reduction requirement failed."
    assert accuracy_loss_pp <= 1.0, "Accuracy-loss requirement failed."

    print("\nPASS: Artifact integrity, reconstruction, and")
    print("      the two acceptance criteria have been verified.")


if __name__ == "__main__":
    main()

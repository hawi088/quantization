
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_all_artifact import (
    BASELINE_DIR,
    BATCH_SIZE,
    MAX_LENGTH,
)
from load_quantized_model import load_quantized_model


def get_predictions(model, inputs):
    predictions = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs["input_ids"]), BATCH_SIZE):
            batch = {
                name: tensor[start : start + BATCH_SIZE]
                for name, tensor in inputs.items()
            }
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).tolist())

    return predictions


def main():
    torch.manual_seed(42)

    # Load the exact validation split and shuffle used in the experiment.
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(872))
    print("Dataset columns:", dataset.column_names)
    print("First example:", dataset[0])
    # Use the same tokenizer and tokenization settings as the experiment.
    tokenizer = AutoTokenizer.from_pretrained(str(BASELINE_DIR))
    inputs = tokenizer(
        list(dataset["sentence"]),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels = dataset["label"]

    # Load the FP32 baseline.
    baseline = AutoModelForSequenceClassification.from_pretrained(
        str(BASELINE_DIR)
    )
    baseline.eval()

    # Load the saved quantized artifact through the actual loader.
    _, quantized = load_quantized_model()

    baseline_predictions = get_predictions(baseline, inputs)
    quantized_predictions = get_predictions(quantized, inputs)

    baseline_correct = sum(
        pred == label
        for pred, label in zip(baseline_predictions, labels)
    )
    quantized_correct = sum(
        pred == label
        for pred, label in zip(quantized_predictions, labels)
    )
    changed = sum(
        fp32 != quantized_pred
        for fp32, quantized_pred in zip(
            baseline_predictions, quantized_predictions
        )
    )

    total = len(labels)

    print("\n=== Loader Reproducibility Check ===")
    print(f"Validation examples: {total}")
    print(f"FP32 accuracy: {baseline_correct / total * 100:.4f}%")
    print(f"Loader artifact accuracy: {quantized_correct / total * 100:.4f}%")
    print(f"Changed predictions: {changed} / {total}")

    if changed == 0 and baseline_correct == quantized_correct:
        print("\nPASS: Loader reproduces the verified predictions.")
    else:
        raise AssertionError(
            "Loader predictions differ from the FP32 baseline."
        )


if __name__ == "__main__":
    main()

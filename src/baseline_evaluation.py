
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Keep evaluation on CPU for consistent comparisons
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
# Load the official GLUE SST-2 validation split
dataset = load_dataset(
    "nyu-mll/glue",
    "sst2",
    split="validation"
)

# SST-2 validation has 872 labeled examples.
# Use all of them so we don't accidentally request more than exist.
dataset = dataset.shuffle(seed=42)

texts = [str(x) for x in dataset["sentence"]]
labels = [int(x) for x in dataset["label"]]

# Tokenize once, outside the timed inference loop
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# Warm-up
with torch.inference_mode():
    for start in range(0, len(texts), 16):
        batch = {
            key: value[start:start + 16].to(device)
            for key, value in inputs.items()
        }
        model(**batch)

# Timed inference
predictions = []

start_time = time.perf_counter()

with torch.inference_mode():
    for start in range(0, len(texts), 16):
        batch = {
            key: value[start:start + 16].to(device)
            for key, value in inputs.items()
        }

        logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).tolist())

elapsed = time.perf_counter() - start_time

correct = sum(
    prediction == label
    for prediction, label in zip(predictions, labels)
)

accuracy = correct / len(labels)

print("\n FP32 Baseline Evaluation")
print(f"Examples: {len(texts)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Inference time: {elapsed:.3f} seconds")
print(f"Throughput: {len(texts) / elapsed:.2f} examples/second")
print(f"Latency per example: {elapsed / len(texts) * 1000:.2f} ms")
print("Device:", device)
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_ID = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# 1. Load the tokenizer and pretrained model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
for name, parameter in model.named_parameters():
    print(name, parameter.dtype)
    break
# 2. Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")

# 3. Prepare the input sentence
text = "This movie was surprisingly good!"
inputs = tokenizer(text, return_tensors="pt")

# 4. Run inference
with torch.no_grad():
    outputs = model(**inputs)

# 5. Get the predicted sentiment
predicted_id = outputs.logits.argmax(dim=-1).item()
predicted_label = model.config.id2label[predicted_id]

# 6. Display the result
print("Text:", text)
print("Logits:", outputs.logits)
print("Prediction:", predicted_label)



# Baseline inspection

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
)

print("\n FP32 Baseline")
print(f"Total parameters: {total_parameters:,}")
print(f"Parameter storage: {parameter_bytes / (1024**2):.2f} MiB")

print("\nParameter data types:")
for dtype in sorted({
    str(parameter.dtype)
    for parameter in model.parameters()
}):
    print(dtype)




output_dir = Path("baseline_model")
output_dir.mkdir(exist_ok=True)

# Save the model and tokenizer locally
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Measure all saved files
total_bytes = sum(
    file.stat().st_size
    for file in output_dir.rglob("*")
    if file.is_file()
)

print("\n Saved Model Size ")
print(f"Directory: {output_dir.resolve()}")
print(f"Total size: {total_bytes / 1_000_000:.2f} MB")
print(f"Total size: {total_bytes / (1024**2):.2f} MiB")


from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    MODEL_ID,
    decode_tensor,
)

CANDIDATE = "INT8_AFFINE_per_channel_axis0"
ARTIFACT_PATH = ARTIFACT_DIR / f"{CANDIDATE}.pt"


def load_quantized_model():
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Quantized artifact not found: {ARTIFACT_PATH}"
        )

    # Load the model architecture and tokenizer.
    # Prefer the saved local baseline to avoid unnecessary downloads.
    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)

    # Load the actual packed INT8 artifact.
    packed_artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # Reconstruct the stored tensors and load them into the model.
    restored_state = {
        name: decode_tensor(entry)
        for name, entry in packed_artifact.items()
    }

    model.load_state_dict(restored_state, strict=True)
    model.eval()

    print(f"Loaded artifact: {ARTIFACT_PATH.resolve()}")
    print(f"Model source for architecture: {source}")
    print("Artifact weights loaded and model set to evaluation mode.")

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_quantized_model()

    text = "This movie was surprisingly good!"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = outputs.logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]

    print("Text:", text)
    print("Logits:", outputs.logits)
    print("Prediction:", predicted_label)

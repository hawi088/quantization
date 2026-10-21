import csv
import shutil
from pathlib import Path

from huggingface_hub import whoami

CANDIDATE = "INT8_AFFINE_per_channel_axis0"
REPO_NAME = "distilbert-sst2-int8-affine-per-channel"
OUT = Path("hf_release")
CSV_PATH = Path("experiments/results/artifact_comparison.csv")
MODULES = [
    "quantizers.py",
    "quantization_math.py",
    "symmetric_quantization.py",
    "representations.py",
]

UTILS = '''import numpy as np
import torch

from quantizers import QuantizationResult, dequantize_tensor


def unpack_codes(packed, count, bits):
    """Unpack bytes and restore signed integer codes."""
    data = packed.detach().cpu().numpy().astype(np.uint8)

    if bits == 8:
        unsigned = data.astype(np.int16)
    else:
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        shifts = np.arange(per_byte, dtype=np.uint8) * bits
        values = ((data[:, None] >> shifts[None, :]) & mask)
        unsigned = values.reshape(-1)[:count].astype(np.int16)

    sign_threshold = 1 << (bits - 1)
    signed = np.where(
        unsigned >= sign_threshold,
        unsigned - (1 << bits),
        unsigned,
    ).astype(np.int16)

    return torch.from_numpy(signed.copy())


def dtype_from_name(name):
    mapping = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported original dtype: {name}")
    return mapping[name]


def decode_tensor(entry):
    """Reconstruct one tensor from a reloaded artifact entry."""
    kind = entry["kind"]

    if kind == "raw":
        return entry["tensor"]

    if kind == "float":
        return entry["values"].to(
            dtype=dtype_from_name(entry["original_dtype"])
        )

    metadata = entry["metadata"]
    codes = unpack_codes(
        entry["packed_codes"],
        entry["code_count"],
        metadata["bits"],
    ).reshape(metadata["original_shape"])

    result = QuantizationResult(
        codes=codes,
        scale=entry["scale"],
        zero_point=entry["zero_point"],
        qmin=metadata["qmin"],
        qmax=metadata["qmax"],
        bits=metadata["bits"],
        method=metadata["method"],
        granularity=metadata["granularity"],
        axis=metadata["axis"],
        group_size=metadata["group_size"],
        original_shape=metadata["original_shape"],
    )

    return dequantize_tensor(result).to(
        dtype=dtype_from_name(entry["original_dtype"])
    )
'''

README = """---
license: apache-2.0
base_model: distilbert/distilbert-base-uncased-finetuned-sst-2-english
tags:
  - quantization
  - int8
  - text-classification
  - distilbert
---

# DistilBERT SST-2 INT8 (affine, per-channel)

INT8 affine per-channel (axis 0) quantized weights of
`distilbert/distilbert-base-uncased-finetuned-sst-2-english`, stored as a
compact PyTorch artifact. Weights are reconstructed to floating point at
load time, so this reduces **storage size**, not inference speed.

## Results (SST-2 validation set)

| Model | Size (MB) | Accuracy (%) |
|---|---|---|
| FP32 original | @@FP32_MB@@ | @@FP32_ACC@@ |
| INT8 affine per-channel | @@INT8_MB@@ | @@INT8_ACC@@ |

Size reduction: @@REDUCTION@@%. Accuracy loss: @@LOSS@@ percentage points.

## Limitations

- Evaluated only on the SST-2 validation set (max length 128).
- Weights are dequantized to float on load; there is no native integer
  inference, so no speedup is expected.
- English sentiment classification only (positive/negative).

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

repo = "@@REPO@@"

for name in [
    "quantization_utils.py",
    "quantizers.py",
    "quantization_math.py",
    "symmetric_quantization.py",
    "representations.py",
]:
    hf_hub_download(repo, name, local_dir=".")

from quantization_utils import decode_tensor

path = hf_hub_download(repo, "@@CANDIDATE@@.pt")
artifact = torch.load(path, map_location="cpu", weights_only=True)
state = {name: decode_tensor(entry) for name, entry in artifact.items()}

model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(repo))
model.load_state_dict(state, strict=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(repo)
inputs = tokenizer("This movie was great!", return_tensors="pt")
print(model(**inputs).logits.argmax(-1).item())  # 1 = positive, 0 = negative
```
"""


def find_module(name):
    for path in Path(".").rglob(name):
        if OUT not in path.parents and "__pycache__" not in path.parts:
            return path
    raise SystemExit(f"Cannot find {name} anywhere in this folder.")


def main():
    if not OUT.exists():
        raise SystemExit("hf_release folder not found. Run this from the quantization folder.")

    try:
        username = whoami()["name"]
    except Exception:
        raise SystemExit("Not logged in to Hugging Face. Log in first.")
    repo = f"{username}/{REPO_NAME}"

    for name in MODULES:
        src = find_module(name)
        shutil.copy(src, OUT / name)
        print(f"Copied {src} -> {OUT / name}")

    (OUT / "quantization_utils.py").write_text(UTILS, encoding="utf-8")
    print("Wrote quantization_utils.py")

    row = None
    with CSV_PATH.open(encoding="utf-8") as file:
        for r in csv.DictReader(file):
            if r["candidate"] == CANDIDATE:
                row = r
    if row is None:
        raise SystemExit(f"{CANDIDATE} not found in {CSV_PATH}")

    int8_bytes = float(row["artifact_bytes"])
    reduction = float(row["actual_reduction_pct"])
    int8_acc = float(row["accuracy_pct"])
    loss = float(row["accuracy_loss_pp"])
    fp32_bytes = int8_bytes / (1 - reduction / 100)

    values = {
        "@@FP32_MB@@": f"{fp32_bytes / 1e6:.2f}",
        "@@FP32_ACC@@": f"{int8_acc + loss:.2f}",
        "@@INT8_MB@@": f"{int8_bytes / 1e6:.2f}",
        "@@INT8_ACC@@": f"{int8_acc:.2f}",
        "@@REDUCTION@@": f"{reduction:.2f}",
        "@@LOSS@@": f"{loss:.2f}",
        "@@REPO@@": repo,
        "@@CANDIDATE@@": CANDIDATE,
    }
    text = README
    for key, value in values.items():
        text = text.replace(key, value)
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print("Wrote README.md")

    shutil.rmtree(OUT / "__pycache__", ignore_errors=True)

    print("\nFiles in hf_release:")
    for f in sorted(OUT.iterdir()):
        print(" -", f.name)


if __name__ == "__main__":
    main()

import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from run_experiment import build_candidates
from quantizers import (
    QuantizationResult,
    quantize_tensor,
    dequantize_tensor,
)


MODEL_ID = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
BASELINE_DIR = Path("baseline_model")
ARTIFACT_DIR = Path("experiments/artifacts/all_candidates")
RESULTS_DIR = Path("experiments/results")

BASELINE_PATH = ARTIFACT_DIR / "fp32_reference.pt"
RESULTS_PATH = RESULTS_DIR / "artifact_comparison.csv"

BATCH_SIZE = 16
MAX_LENGTH = 128
GROUP_SIZE_DEFAULT = 64


def pack_codes(codes, bits):
    """Pack signed integer codes into 8-bit storage bytes."""
    values = codes.detach().cpu().numpy().astype(np.int16).reshape(-1)
    mask = (1 << bits) - 1
    unsigned = np.bitwise_and(values, mask).astype(np.uint8)

    if bits == 8:
        packed = unsigned
    else:
        per_byte = 8 // bits
        padding = (-len(unsigned)) % per_byte

        if padding:
            unsigned = np.pad(unsigned, (0, padding))

        chunks = unsigned.reshape(-1, per_byte)
        packed = np.zeros(len(chunks), dtype=np.uint8)

        for index in range(per_byte):
            packed |= chunks[:, index] << (index * bits)

    return torch.from_numpy(packed.copy()), len(values)


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


def encode_tensor(tensor, config):
    """Encode one state-dict tensor for compact serialization."""
    original_dtype = str(tensor.dtype)

    if not tensor.is_floating_point():
        return {
            "kind": "raw",
            "tensor": tensor.detach().cpu(),
            "original_dtype": original_dtype,
        }

    representation = config["representation"]

    if representation in ("FP16", "BF16"):
        target_dtype = (
            torch.float16 if representation == "FP16"
            else torch.bfloat16
        )
        return {
            "kind": "float",
            "values": tensor.detach().cpu().to(target_dtype),
            "original_dtype": original_dtype,
        }

    method = (
        "symmetric"
        if representation.endswith("_SYM")
        else "affine"
    )
    bits = int(representation[3])

    granularity = config["granularity"]
    kwargs = {
        "bits": bits,
        "method": method,
        "granularity": granularity,
    }

    if granularity == "per_channel":
        kwargs["axis"] = config["channel_axis"]
    elif granularity == "per_group":
        kwargs["group_size"] = config["group_size"]

    result = quantize_tensor(tensor.detach().cpu(), **kwargs)
    packed, code_count = pack_codes(result.codes, bits)

    zero_point = result.zero_point
    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.detach().cpu()

    return {
        "kind": "quantized",
        "packed_codes": packed,
        "code_count": code_count,
        "scale": result.scale.detach().cpu().to(torch.float32),
        "zero_point": zero_point,
        "original_dtype": original_dtype,
        "metadata": {
            "qmin": result.qmin,
            "qmax": result.qmax,
            "bits": result.bits,
            "method": result.method,
            "granularity": result.granularity,
            "axis": result.axis,
            "group_size": result.group_size,
            "original_shape": tuple(result.original_shape),
        },
    }


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


def evaluate(model, inputs, labels):
    predictions = []
    all_logits = []

    model.eval()
    start = time.perf_counter()

    with torch.inference_mode():
        for offset in range(0, len(labels), BATCH_SIZE):
            batch = {
                key: value[offset:offset + BATCH_SIZE]
                for key, value in inputs.items()
            }
            logits = model(**batch).logits
            all_logits.append(logits.cpu())
            predictions.extend(logits.argmax(dim=-1).tolist())

    accuracy = sum(
        prediction == label
        for prediction, label in zip(predictions, labels)
    ) / len(labels)

    return accuracy, torch.cat(all_logits), time.perf_counter() - start


def save_csv(rows):
    fields = [
        "candidate",
        "representation",
        "granularity",
        "channel_axis",
        "group_size",
        "artifact_path",
        "artifact_bytes",
        "artifact_MB",
        "actual_reduction_pct",
        "accuracy_pct",
        "accuracy_loss_pp",
        "weight_RMSE",
        "relative_weight_RMSE",
        "logit_RMSE",
        "size_pass",
        "accuracy_pass",
        "passes_both",
    ]

    with RESULTS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(42)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    print("Loading FP32 model...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to("cpu")
    model.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    # Serialize the baseline using the same PyTorch serializer.
    torch.save(original_state, BASELINE_PATH)
    baseline_bytes = BASELINE_PATH.stat().st_size

    print(f"FP32 checkpoint size: {baseline_bytes:,} bytes")
    print(f"FP32 checkpoint size: {baseline_bytes / 1_000_000:.2f} MB")

    print("\nLoading SST-2 validation set...")
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

    model.load_state_dict(original_state, strict=True)
    baseline_accuracy, baseline_logits, _ = evaluate(
        model, inputs, labels
    )

    print(f"FP32 validation accuracy: {baseline_accuracy * 100:.2f}%")

    candidates = build_candidates()
    rows = []

    for index, config in enumerate(candidates, start=1):
        name = config["candidate"]
        artifact_path = ARTIFACT_DIR / f"{name}.pt"
        print(f"\n[{index}/{len(candidates)}] {name}")

        if not artifact_path.exists():
            print("Encoding tensors...")
            artifact = {
                tensor_name: encode_tensor(tensor, config)
                for tensor_name, tensor in original_state.items()
            }

            torch.save(artifact, artifact_path)
            del artifact
            gc.collect()
        else:
            print("Artifact already exists; reusing saved file.")

        artifact_bytes = artifact_path.stat().st_size
        reduction_pct = (
            1 - artifact_bytes / baseline_bytes
        ) * 100

        print(
            f"Serialized size: {artifact_bytes:,} bytes "
            f"({artifact_bytes / 1_000_000:.2f} MB)"
        )
        print(f"Actual size reduction: {reduction_pct:.2f}%")

        print("Reloading and reconstructing...")
        loaded_artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )

        reconstructed_state = {
            tensor_name: decode_tensor(entry)
            for tensor_name, entry in loaded_artifact.items()
        }

        squared_error = 0.0
        squared_weight = 0.0
        element_count = 0

        for tensor_name, original in original_state.items():
            restored = reconstructed_state[tensor_name]

            assert restored.shape == original.shape, (
                f"Shape mismatch: {tensor_name}"
            )

            if original.is_floating_point():
                diff = (
                    original.to(torch.float64)
                    - restored.to(torch.float64)
                )
                squared_error += diff.square().sum().item()
                squared_weight += (
                    original.to(torch.float64).square().sum().item()
                )
                element_count += original.numel()
            else:
                assert torch.equal(original, restored), (
                    f"Non-floating tensor changed: {tensor_name}"
                )

        weight_rmse = (
            squared_error / max(element_count, 1)
        ) ** 0.5

        relative_weight_rmse = (
            (squared_error / squared_weight) ** 0.5
            if squared_weight > 0 else 0.0
        )

        model.load_state_dict(reconstructed_state, strict=True)
        accuracy, logits, _ = evaluate(model, inputs, labels)

        accuracy_loss_pp = (
            baseline_accuracy - accuracy
        ) * 100

        logit_diff = (
            baseline_logits.to(torch.float64)
            - logits.to(torch.float64)
        )
        logit_rmse = (
            logit_diff.square().mean().item()
        ) ** 0.5

        size_pass = reduction_pct >= 60.0
        accuracy_pass = accuracy_loss_pp <= 1.0

        row = {
            "candidate": name,
            "representation": config["representation"],
            "granularity": config["granularity"],
            "channel_axis": config["channel_axis"],
            "group_size": config["group_size"],
            "artifact_path": str(artifact_path),
            "artifact_bytes": artifact_bytes,
            "artifact_MB": round(artifact_bytes / 1_000_000, 4),
            "actual_reduction_pct": round(reduction_pct, 4),
            "accuracy_pct": round(accuracy * 100, 4),
            "accuracy_loss_pp": round(accuracy_loss_pp, 4),
            "weight_RMSE": weight_rmse,
            "relative_weight_RMSE": relative_weight_rmse,
            "logit_RMSE": logit_rmse,
            "size_pass": size_pass,
            "accuracy_pass": accuracy_pass,
            "passes_both": size_pass and accuracy_pass,
        }

        rows.append(row)
        save_csv(rows)

        print(f"Reloaded accuracy: {row['accuracy_pct']:.2f}%")
        print(f"Accuracy loss: {accuracy_loss_pp:.2f} pp")
        print(f"Weight RMSE: {weight_rmse:.6f}")
        print(f"Logit RMSE: {logit_rmse:.6f}")
        print(f"Passes both targets: {row['passes_both']}")

        del loaded_artifact, reconstructed_state, logits
        gc.collect()

    print("\n--- Actual artifact comparison complete ---")
    print(f"Artifacts directory: {ARTIFACT_DIR}")
    print(f"Comparison CSV: {RESULTS_PATH}")
    print(f"Configurations measured: {len(rows)}")
    print(
        "These are actual serialized sizes of the saved representations. "
        "Inference reconstructs floating-point weights, so this does not "
        "demonstrate native integer inference speed."
    )


if __name__ == "__main__":
    main()

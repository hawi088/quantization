
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT / "experiments" / "results" / "artifact_comparison.csv"
)
BASELINE_FILE = (
    PROJECT_ROOT / "experiments" / "artifacts"
    / "all_candidates" / "fp32_reference.pt"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "experiments" / "results"
    / "selected_int8_config.json"
)

SELECTED_CANDIDATE = "INT8_AFFINE_per_channel_axis0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_FILE}")

    if not BASELINE_FILE.is_file():
        raise FileNotFoundError(f"FP32 baseline not found: {BASELINE_FILE}")

    with RESULTS_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        matches = [
            row for row in rows
            if row["candidate"] == SELECTED_CANDIDATE
        ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one CSV row for {SELECTED_CANDIDATE}; "
            f"found {len(matches)}."
        )

    row = matches[0]

    if row["passes_both"].strip().lower() != "true":
        raise ValueError("The selected candidate does not pass both requirements.")

    artifact_path = PROJECT_ROOT / row["artifact_path"]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Selected artifact not found: {artifact_path}")

    actual_artifact_bytes = artifact_path.stat().st_size
    recorded_artifact_bytes = int(row["artifact_bytes"])

    if actual_artifact_bytes != recorded_artifact_bytes:
        raise ValueError(
            "Artifact size mismatch: "
            f"CSV={recorded_artifact_bytes}, "
            f"disk={actual_artifact_bytes}"
        )

    baseline_bytes = BASELINE_FILE.stat().st_size

    # Accuracy loss = baseline accuracy - candidate accuracy.
    baseline_accuracy = (
        float(row["accuracy_pct"]) + float(row["accuracy_loss_pp"])
    )

    config = {
        "project": "Build Your Own Model Quantizer",
        "model_id": (
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        ),
        "task": "binary sentiment classification",
        "evaluation": {
            "dataset": "SST-2 validation split",
            "examples": 872,
            "random_seed": 42,
            "batch_size": 16,
            "max_sequence_length": 128,
            "baseline_accuracy_pct": baseline_accuracy,
            "quantized_accuracy_pct": float(row["accuracy_pct"]),
            "accuracy_loss_percentage_points": float(
                row["accuracy_loss_pp"]
            ),
        },
        "quantization": {
            "candidate": row["candidate"],
            "representation": row["representation"],
            "bits": 8,
            "granularity": row["granularity"],
            "channel_axis": int(row["channel_axis"]),
            "signed_code_range": [-128, 127],
            "method": "affine",
        },
        "artifact": {
            "path": str(artifact_path.relative_to(PROJECT_ROOT)),
            "size_bytes": actual_artifact_bytes,
            "size_MB_decimal": actual_artifact_bytes / 1_000_000,
            "sha256": sha256_file(artifact_path),
        },
        "baseline_artifact": {
            "path": str(BASELINE_FILE.relative_to(PROJECT_ROOT)),
            "size_bytes": baseline_bytes,
            "size_MB_decimal": baseline_bytes / 1_000_000,
        },
        "measured_results": {
            "actual_reduction_pct": float(row["actual_reduction_pct"]),
            "weight_RMSE": float(row["weight_RMSE"]),
            "relative_weight_RMSE": float(row["relative_weight_RMSE"]),
            "logit_RMSE": float(row["logit_RMSE"]),
            "size_requirement_passed": (
                row["size_pass"].strip().lower() == "true"
            ),
            "accuracy_requirement_passed": (
                row["accuracy_pass"].strip().lower() == "true"
            ),
            "passes_both": True,
        },
        "artifact_format_note": (
            "Custom packed checkpoint. This artifact demonstrates storage "
            "compression and reconstruction, not native INT8 inference."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Selected configuration saved.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Artifact size: {actual_artifact_bytes:,} bytes")
    print(f"SHA-256: {config['artifact']['sha256']}")
    print(f"Accuracy: {config['evaluation']['quantized_accuracy_pct']:.3f}%")
    print(
        "Storage reduction: "
        f"{config['measured_results']['actual_reduction_pct']:.4f}%"
    )
    print(f"Passes both requirements: {config['measured_results']['passes_both']}")


if __name__ == "__main__":
    main()

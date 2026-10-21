"""
Visualize quantization error for ALL passing configurations.

For every config it draws the same FP32 -> quantize -> INT -> dequantize -> FP32
grid (4x5 = 20 weights) and saves one image per config, labelled with the
configuration name. Images are ranked best-first (lowest logit RMSE first)
and the rank is part of the filename, so they sort in order in a file browser.

The same 20 weights (same tensor, same window) are used for every config,
so the images are directly comparable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from transformers import AutoModelForSequenceClassification

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    MODEL_ID,
    decode_tensor,
    unpack_codes,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR = RESULTS_DIR / "plots" / "config_error_grids"

ROWS, COLS = 4, 5  # 20 weights

# stem = artifact filename without ".pt"  (same naming as INT8_AFFINE_per_channel_axis0)
# label = what is printed on the image
PASSING_CONFIGS = [
    # stem,                              label,                            size_MB, loss_pp, logit_RMSE
    ("INT4_AFFINE_per_channel_axis1", "INT4_AFFINE per-channel ax1", 33.83, 0.573, 0.541),
    ("INT4_AFFINE_per_channel_axis0", "INT4_AFFINE per-channel ax0", 34.06, 0.459, 0.612),
    ("INT4_SYM_g128",                "INT4_SYM g128",               35.57, -0.115, 0.602),
    ("INT4_SYM_g64",                 "INT4_SYM g64",                37.66, 0.688, 0.456),
    ("INT4_AFFINE_g128",             "INT4_AFFINE g128",            37.66, 0.459, 0.377),
    ("INT4_SYM_g32",                 "INT4_SYM g32",                41.85, 0.459, 0.363),
    ("INT4_AFFINE_g64",              "INT4_AFFINE g64",             41.85, 0.229, 0.290),
    ("INT4_AFFINE_g32",              "INT4_AFFINE g32",             50.22, 0.917, 0.259),
    ("INT8_SYM_per_tensor",          "INT8_SYM per-tensor",         66.96, 0.573, 0.327),
    ("INT8_AFFINE_per_tensor",       "INT8_AFFINE per-tensor",      66.96, 0.688, 0.233),
    ("INT8_SYM_per_channel_axis1",   "INT8_SYM per-channel ax1",    67.13, 0.229, 0.066),
    ("INT8_SYM_per_channel_axis0",   "INT8_SYM per-channel ax0",    67.25, 0.229, 0.044),
    ("INT8_AFFINE_per_channel_axis1", "INT8_AFFINE per-channel ax1", 67.31, 0.344, 0.042),
    ("INT8_AFFINE_per_channel_axis0", "INT8_AFFINE per-channel ax0", 67.54, 0.000, 0.040),
    ("INT8_SYM_g128",                "INT8_SYM g128",               69.05, 0.229, 0.034),
    ("INT8_SYM_g64",                 "INT8_SYM g64",                71.14, 0.000, 0.027),
    ("INT8_AFFINE_g128",             "INT8_AFFINE g128",            71.14, 0.229, 0.025),
    ("INT8_SYM_g32",                 "INT8_SYM g32",                75.32, 0.000, 0.021),
    ("INT8_AFFINE_g64",              "INT8_AFFINE g64",             75.32, 0.000, 0.018),
]


def rank_key(config):
    """
    'Performed better' = closest to the FP32 model:
    lowest logit RMSE first, then smaller accuracy loss, then smaller size.
    Edit this if you want a different ordering.
    """
    _, _, size_mb, loss_pp, logit_rmse = config
    return (logit_rmse, abs(loss_pp), size_mb)


# ----------------------------------------------------------------------
# Real integer codes from the artifact
# ----------------------------------------------------------------------

def get_integer_block(entry, original_shape, window):
    """
    Unpack the REAL signed integer codes stored in the artifact
    (same unpack_codes used by decode_tensor), for INT4 and INT8 alike.
    """
    if entry.get("kind") != "quantized":
        raise ValueError("Selected tensor is not stored as quantized codes.")

    metadata = entry["metadata"]
    codes = unpack_codes(
        entry["packed_codes"],
        entry["code_count"],
        metadata["bits"],
    ).reshape(original_shape)

    return (
        codes[window].numpy().astype(np.int64),
        f"INT{metadata['bits']} codes read from artifact",
    )


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def draw_value_grid(ax, values, colors, fmt, title):
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 1, 1,
                facecolor=colors[r][c], edgecolor="white", linewidth=2.5,
            ))
            ax.text(
                c + 0.5, rows - 1 - r + 0.5, fmt(values[r, c]),
                ha="center", va="center",
                color="white", fontsize=12, fontweight="bold",
            )
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)


def build_colors(int_block):
    """Black = integer 0, shared color = repeated integer value, gray = unique."""
    palette = ["#4a6cf7", "#e3b05a", "#4cae7a", "#c46bb5", "#e07a5f", "#5bc0de"]
    unique, counts = np.unique(int_block, return_counts=True)
    repeated = [int(u) for u, n in zip(unique, counts) if n > 1 and u != 0]
    value_color = {v: palette[i % len(palette)] for i, v in enumerate(repeated)}

    rows, cols = int_block.shape
    return [
        [
            "black" if int_block[r, c] == 0
            else value_color.get(int(int_block[r, c]), "#8c8c8c")
            for c in range(cols)
        ]
        for r in range(rows)
    ]


def plot_config(
    rank,
    total,
    label,
    tensor_name,
    orig_block,
    int_block,
    rest_block,
    int_source,
    size_mb,
    loss_pp,
    logit_rmse,
    output_path,
):
    colors = build_colors(int_block)
    int_name = label.split("_")[0]  # "INT4" or "INT8"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))

    draw_value_grid(axes[0], orig_block, colors, lambda v: f"{v:.5f}", "FP32")
    draw_value_grid(axes[1], int_block, colors, lambda v: f"{int(v)}", int_name)
    draw_value_grid(axes[2], rest_block, colors, lambda v: f"{v:.5f}", "FP32 (reconstructed)")

    for x_start, x_end, text in [(0.335, 0.385, "quantize"), (0.645, 0.695, "dequantize")]:
        fig.add_artist(
            FancyArrowPatch(
                (x_start, 0.47), (x_end, 0.47),
                transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=22,
                color="#444444", linewidth=2,
            )
        )
        fig.text(
            (x_start + x_end) / 2, 0.52, text,
            ha="center", va="bottom", fontsize=15,
        )

    # Configuration label (the main identifier of the image)
    fig.text(
        0.5, 0.965, f"#{rank:02d} of {total}   |   {label}",
        ha="center", va="top", fontsize=24, fontweight="bold",
    )
    fig.text(
        0.5, 0.895,
        f"size {size_mb:.2f} MB   |   accuracy loss {loss_pp:+.3f} pp   |   "
        f"logit RMSE {logit_rmse:.3f}",
        ha="center", va="top", fontsize=14, color="#333333",
    )

    abs_err = np.abs(rest_block - orig_block)
    fig.text(
        0.5, 0.04,
        f"{tensor_name}  |  {ROWS * COLS} weights  |  "
        f"mean |error| = {abs_err.mean():.6f}, max |error| = {abs_err.max():.6f}  |  "
        f"{int_source}",
        ha="center", fontsize=11, color="#555555",
    )

    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.35, top=0.78, bottom=0.12)
    plt.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FP32 baseline...")
    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID
    baseline = AutoModelForSequenceClassification.from_pretrained(source)
    baseline.to("cpu")
    baseline.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in baseline.state_dict().items()
    }
    del baseline

    # Same tensor + same window for every config -> comparable images
    tensor_name = max(
        [
            name for name, t in original_state.items()
            if t.is_floating_point() and t.ndim >= 2
        ],
        key=lambda n: original_state[n].numel(),
    )
    original = original_state[tensor_name].to(torch.float64)
    original = original.reshape(original.shape[0], -1)

    r0 = original.shape[0] // 2
    c0 = original.shape[1] // 2
    window = (slice(r0, r0 + ROWS), slice(c0, c0 + COLS))
    orig_block = original[window].numpy()

    print(f"Selected tensor: {tensor_name}  shape={tuple(original.shape)}")

    ranked = sorted(PASSING_CONFIGS, key=rank_key)
    total = len(ranked)
    saved = []

    for rank, (stem, label, size_mb, loss_pp, logit_rmse) in enumerate(ranked, start=1):
        artifact_path = ARTIFACT_DIR / f"{stem}.pt"

        if not artifact_path.is_file():
            available = sorted(p.stem for p in Path(ARTIFACT_DIR).glob("*.pt"))
            print(f"[{rank:02d}/{total}] SKIPPED {stem}: artifact not found.")
            print(f"        Available artifacts: {available}")
            continue

        print(f"[{rank:02d}/{total}] {label}")

        packed = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )

        entry = packed[tensor_name]
        restored = decode_tensor(entry).detach().cpu().to(torch.float64)
        restored = restored.reshape(restored.shape[0], -1)
        rest_block = restored[window].numpy()

        int_block, int_source = get_integer_block(
            entry, tuple(original.shape), window
        )

        output_path = OUTPUT_DIR / f"{rank:02d}_{stem}.png"

        plot_config(
            rank=rank,
            total=total,
            label=label,
            tensor_name=tensor_name,
            orig_block=orig_block,
            int_block=int_block,
            rest_block=rest_block,
            int_source=int_source,
            size_mb=size_mb,
            loss_pp=loss_pp,
            logit_rmse=logit_rmse,
            output_path=output_path,
        )

        saved.append(output_path)
        del packed, restored

    print(f"\nSaved {len(saved)} images to: {OUTPUT_DIR}")
    for path in saved:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
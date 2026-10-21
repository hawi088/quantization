"""
Prediction-level consistency check, run separately for two groups:

  1. PASSING groups : configs that meet all constraints
                      (size reduction, accuracy loss, logit RMSE)
  2. WORST group    : the N worst configs by accuracy loss

The two groups are NEVER mixed: each one has its own CSVs, its own plots, its
own heatmap and its own console report, stored in separate folders.

For every configuration it compares the quantized model with the FP32 model
example by example and reports:

  * flips            : predictions that differ from FP32
  * harmful flips    : FP32 correct  -> quantized wrong
  * helpful flips    : FP32 wrong    -> quantized correct
  * masked flips     : flips cancelled out in the accuracy number
                       (2 * min(harmful, helpful))
  * FP32 margin      : how confident FP32 was on the examples that flipped
  * predicted +%     : share of examples predicted as class 1 (collapse check)

Outputs:
  experiments/results/prediction_consistency_passing/
  experiments/results/prediction_consistency_worst/
    prediction_consistency_summary.csv
    flipped_examples.csv
    fragile_examples.csv
    plots/flip_balance.png
    plots/margin_vs_fragility.png
    plots/flip_heatmap.png

Run it from the project root, like your other scripts.
"""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from build_all_artifact import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    MODEL_ID,
    MAX_LENGTH,
    RESULTS_DIR,
    RESULTS_PATH,
    decode_tensor,
    evaluate,
)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# Output folders (one per group, never shared)
PASSING_DIR = RESULTS_DIR / "prediction_consistency_passing"
WORST_DIR = RESULTS_DIR / "prediction_consistency_worst"

# Which groups to run
RUN_PASSING = True
RUN_WORST = True

# ---- Passing group: thresholds applied to artifact_comparison.csv ----
MIN_REDUCTION_PCT = 70.0   # actual_reduction_pct >= this
MAX_ACC_LOSS_PP = 1.0      # accuracy_loss_pp     <= this
MAX_LOGIT_RMSE = 0.2       # logit_RMSE           <  this

# None -> filter by the thresholds above.
# Or list candidate names explicitly to override the filter.
PASSING_CONFIGS = None

# ---- Worst group: ranked by accuracy loss (ties broken by logit RMSE) ----
WORST_N = 5

# None -> pick the worst N automatically.
# Or list candidate names explicitly to override.
WORST_CONFIGS = None

MAX_HEATMAP_ROWS = 40  # most fragile examples shown in the heatmap

HARM_COLOR = "#d1495b"
HELP_COLOR = "#2e8b57"


# ----------------------------------------------------------------------
# Config selection
# ----------------------------------------------------------------------

def read_rows():
    with RESULTS_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_passing(rows):
    """Configs that meet all three thresholds."""
    if PASSING_CONFIGS:
        names = {row["candidate"] for row in rows}
        for name in PASSING_CONFIGS:
            if name not in names:
                print(f"WARNING: {name} not found in {RESULTS_PATH.name}")
        return list(PASSING_CONFIGS)

    passing = []
    rejected = []

    for row in rows:
        reduction = float(row["actual_reduction_pct"])
        acc_loss = float(row["accuracy_loss_pp"])
        logit = float(row["logit_RMSE"])

        reasons = []
        if reduction < MIN_REDUCTION_PCT:
            reasons.append(f"reduction {reduction:.2f}% < {MIN_REDUCTION_PCT}%")
        if acc_loss > MAX_ACC_LOSS_PP:
            reasons.append(f"accuracy loss {acc_loss:.2f} pp > {MAX_ACC_LOSS_PP} pp")
        if not logit < MAX_LOGIT_RMSE:
            reasons.append(f"logit RMSE {logit:.3f} >= {MAX_LOGIT_RMSE}")

        if reasons:
            rejected.append((row["candidate"], reasons))
        else:
            passing.append(row["candidate"])

    print(
        f"Thresholds: reduction >= {MIN_REDUCTION_PCT}%, "
        f"accuracy loss <= {MAX_ACC_LOSS_PP} pp, "
        f"logit RMSE < {MAX_LOGIT_RMSE}"
    )
    print(f"{len(passing)} of {len(rows)} configurations pass.")
    for name, reasons in rejected:
        print(f"  rejected {name}: {'; '.join(reasons)}")

    if not passing:
        raise RuntimeError(
            f"No configurations in {RESULTS_PATH} meet the thresholds."
        )

    return passing


def select_worst(rows, exclude=()):
    """The WORST_N configs by accuracy loss (ties -> higher logit RMSE)."""
    if WORST_CONFIGS:
        names = {row["candidate"] for row in rows}
        for name in WORST_CONFIGS:
            if name not in names:
                print(f"WARNING: {name} not found in {RESULTS_PATH.name}")
        return list(WORST_CONFIGS)

    pool = [row for row in rows if row["candidate"] not in set(exclude)]
    ranked = sorted(
        pool,
        key=lambda r: (-float(r["accuracy_loss_pp"]), -float(r["logit_RMSE"])),
    )
    worst = ranked[:WORST_N]

    print(f"Worst {WORST_N} of {len(rows)} configurations in {RESULTS_PATH.name}:")
    for row in worst:
        print(
            f"  {row['candidate']:<32} "
            f"acc loss {float(row['accuracy_loss_pp']):>6.2f} pp   "
            f"logit RMSE {float(row['logit_RMSE']):.3f}"
        )

    if len(ranked) > WORST_N:
        cutoff = float(worst[-1]["accuracy_loss_pp"])
        next_loss = float(ranked[WORST_N]["accuracy_loss_pp"])
        if cutoff == next_loss:
            print(
                f"  NOTE: {ranked[WORST_N]['candidate']} ties with the last "
                f"selected config at {cutoff:.2f} pp and was left out."
            )

    return [row["candidate"] for row in worst]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def pretty(name):
    name = name.replace("_per_channel_axis", " ch-ax")
    name = name.replace("_per_group_g", " g")
    name = name.replace("_per_tensor", " per-tensor")
    return name


def safe_text(text, limit=55):
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("$", r"\$")


def top_margin(logits):
    """Top logit minus runner-up logit (the model's confidence margin)."""
    ordered = np.sort(logits, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def analyze_config(candidate, logits, base_pred, base_correct, labels, margin):
    """Compare one quantized model with the FP32 model, example by example."""
    pred = logits.argmax(axis=1)
    correct = pred == labels
    flipped = pred != base_pred

    harmful = base_correct & ~correct
    helpful = ~base_correct & correct

    n = len(labels)
    harmful_count = int(harmful.sum())
    helpful_count = int(helpful.sum())

    direction = np.zeros(n, dtype=np.int8)
    direction[harmful] = -1
    direction[helpful] = 1

    stats = {
        "candidate": candidate,
        "accuracy_pct": float(correct.mean() * 100),
        "net_change_pp": (helpful_count - harmful_count) / n * 100,
        "flips": int(flipped.sum()),
        "flip_rate_pct": float(flipped.mean() * 100),
        "harmful_flips": harmful_count,
        "helpful_flips": helpful_count,
        "masked_flips": 2 * min(harmful_count, helpful_count),
        "predicted_positive_pct": float((pred == 1).mean() * 100),
        "mean_margin_flipped": (
            float(margin[flipped].mean()) if flipped.any() else 0.0
        ),
        "max_margin_flipped": (
            float(margin[flipped].max()) if flipped.any() else 0.0
        ),
        "mean_margin_all": float(margin.mean()),
    }

    return stats, flipped, direction, pred


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_flip_balance(summary, n_examples, group_label, path):
    names = [pretty(row["candidate"]) for row in summary]
    harmful = np.array([row["harmful_flips"] for row in summary])
    helpful = np.array([row["helpful_flips"] for row in summary])
    net = helpful - harmful
    y = np.arange(len(summary))

    limit = max(1, int(harmful.max()), int(helpful.max()))

    fig, ax = plt.subplots(figsize=(13, max(5, 0.5 * len(summary) + 2.5)))

    ax.barh(
        y, -harmful, color=HARM_COLOR,
        label="Right → wrong  (FP32 correct, quantized wrong)",
    )
    ax.barh(
        y, helpful, color=HELP_COLOR,
        label="Wrong → right  (FP32 wrong, quantized correct)",
    )
    ax.scatter(
        net, y, marker="D", s=45, color="black", zorder=5,
        label="Net change (what accuracy reports)",
    )
    ax.axvline(0, color="#333333", linewidth=1)

    ax.set_xlim(-limit * 1.15, limit * 1.15)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of examples")
    ax.set_title(
        f"Prediction flips vs FP32 - {group_label} configs "
        f"({n_examples} SST-2 validation examples)",
        fontsize=14, fontweight="bold", pad=24,
    )
    ax.text(
        0.5, 1.01,
        "Bars: individual predictions that changed. "
        "Diamond: the net effect that the accuracy number sees.",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color="#555555",
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    for i, row in enumerate(summary):
        ax.text(
            1.02, i,
            f"{row['flips']} flips ({row['flip_rate_pct']:.2f}%)  |  "
            f"net {row['net_change_pp']:+.2f} pp",
            transform=ax.get_yaxis_transform(),
            va="center", fontsize=9,
        )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
    )

    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_margin_vs_fragility(margin, flip_counts, n_configs, group_label, path):
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, size=len(margin))

    stable = flip_counts == 0
    fragile = ~stable

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        margin[stable], flip_counts[stable] + jitter[stable],
        s=14, color="#9aa0a6", alpha=0.35,
        label=f"Never flips ({int(stable.sum())})",
    )
    ax.scatter(
        margin[fragile], flip_counts[fragile] + jitter[fragile],
        s=28, color=HARM_COLOR, alpha=0.8, edgecolors="white", linewidth=0.4,
        label=f"Flips in at least one config ({int(fragile.sum())})",
    )

    if fragile.any():
        largest = float(margin[fragile].max())
        ax.axvline(largest, color="black", linestyle="--", linewidth=1)
        ax.text(
            largest, n_configs + 0.3,
            f" largest FP32 margin of any flipped example: {largest:.2f}",
            ha="left", va="center", fontsize=9,
        )

    ax.set_ylim(-0.5, n_configs + 0.7)
    ax.set_xlabel("FP32 confidence margin  (top logit − runner-up logit)")
    ax.set_ylabel(
        f"Number of {group_label} configs that flip the example (of {n_configs})"
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(
        f"Which examples flip? FP32 confidence vs flip count ({group_label} configs)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="center right")

    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_flip_heatmap(config_names, direction_matrix, margin, texts,
                      group_label, path):
    """
    direction_matrix: shape [n_configs, n_examples];
    -1 harmful, 0 same, +1 helpful.
    """
    counts = (direction_matrix != 0).sum(axis=0)
    candidates = np.where(counts > 0)[0]

    if len(candidates) == 0:
        return False

    order = sorted(candidates, key=lambda i: (-counts[i], margin[i]))
    order = order[:MAX_HEATMAP_ROWS]

    data = direction_matrix[:, order].T  # rows = examples, columns = configs

    cmap = ListedColormap([HARM_COLOR, "#eceff1", HELP_COLOR])

    fig, ax = plt.subplots(
        figsize=(
            max(10, 0.6 * len(config_names) + 9),
            max(5, 0.3 * len(order) + 3),
        )
    )
    ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(config_names)))
    ax.set_xticklabels(
        [pretty(name) for name in config_names],
        rotation=60, ha="right", fontsize=9,
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(
        [
            f"#{index}  {safe_text(texts[index])}  (margin {margin[index]:.2f})"
            for index in order
        ],
        fontsize=8,
    )

    ax.set_xticks(np.arange(-0.5, len(config_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    shown = f"{len(order)} most fragile of {len(candidates)}"
    ax.set_title(
        f"Which examples flip in which {group_label} config  "
        f"({shown} flipped examples)",
        fontsize=13, fontweight="bold", pad=34,
    )
    ax.legend(
        handles=[
            Patch(color=HARM_COLOR, label="Right → wrong"),
            Patch(color=HELP_COLOR, label="Wrong → right"),
            Patch(color="#eceff1", label="Same prediction as FP32"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
    )

    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_summary(summary, n_examples, base_accuracy, group_label):
    print(f"\n=== {group_label.upper()} CONFIGS ===")
    print(f"FP32 accuracy: {base_accuracy * 100:.4f}%   ({n_examples} examples)\n")
    header = (
        f"{'config':<30}{'acc %':>8}{'net pp':>8}{'flips':>7}"
        f"{'flip %':>8}{'right>wrong':>13}{'wrong>right':>13}{'masked':>8}"
        f"{'pred +%':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{pretty(row['candidate']):<30}"
            f"{row['accuracy_pct']:>8.3f}"
            f"{row['net_change_pp']:>+8.3f}"
            f"{row['flips']:>7d}"
            f"{row['flip_rate_pct']:>8.2f}"
            f"{row['harmful_flips']:>13d}"
            f"{row['helpful_flips']:>13d}"
            f"{row['masked_flips']:>8d}"
            f"{row['predicted_positive_pct']:>9.2f}"
        )


def print_findings(summary, flip_matrix, margin, group_label):
    n_examples = flip_matrix.shape[1]
    n_configs = flip_matrix.shape[0]

    print(f"\nKey findings ({group_label})")
    print("-" * (14 + len(group_label)))

    cancelling = [
        row for row in summary
        if row["flips"] > 0 and row["harmful_flips"] == row["helpful_flips"]
    ]
    if cancelling:
        print(
            f"{len(cancelling)} config(s) show exactly 0.00 pp net change "
            f"while still changing predictions:"
        )
        for row in cancelling:
            print(
                f"  {pretty(row['candidate'])}: {row['flips']} predictions "
                f"changed ({row['harmful_flips']} right->wrong, "
                f"{row['helpful_flips']} wrong->right)"
            )

    most_masked = max(summary, key=lambda row: row["masked_flips"])
    if most_masked["masked_flips"] > 0:
        print(
            f"Largest hidden churn: {pretty(most_masked['candidate'])} - "
            f"{most_masked['masked_flips']} flips cancel out, so net change "
            f"is {most_masked['net_change_pp']:+.2f} pp but "
            f"{most_masked['flips']} predictions changed."
        )

    collapsed = [
        row for row in summary
        if row["predicted_positive_pct"] >= 99.0
        or row["predicted_positive_pct"] <= 1.0
    ]
    if collapsed:
        print(
            f"{len(collapsed)} config(s) predict (almost) a single class for "
            f"every example:"
        )
        for row in collapsed:
            print(
                f"  {pretty(row['candidate'])}: "
                f"{row['predicted_positive_pct']:.2f}% predicted positive"
            )

    any_flip = flip_matrix.any(axis=0)
    all_flip = flip_matrix.all(axis=0)
    print(
        f"{int(any_flip.sum())} of {n_examples} examples "
        f"({any_flip.mean() * 100:.2f}%) flip in at least one of "
        f"{n_configs} {group_label} configs; "
        f"{int(all_flip.sum())} flip in all of them."
    )

    if any_flip.any():
        print(
            f"FP32 confidence margin - median over all examples: "
            f"{np.median(margin):.2f}; median over examples that flip: "
            f"{np.median(margin[any_flip]):.2f}; "
            f"largest margin of any flipped example: "
            f"{margin[any_flip].max():.2f}."
        )


# ----------------------------------------------------------------------
# One group, end to end
# ----------------------------------------------------------------------

def run_group(group_label, candidates, reported_loss, output_dir, model,
              original_state, inputs, labels, texts, base_accuracy,
              base_pred, base_correct, margin):
    """Analyze one group of configs and write everything to its own folder."""
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_examples = len(labels)
    results = []  # (stats, flipped, direction, pred)

    print(f"\n##### {group_label.upper()} group: {len(candidates)} configs #####")

    for index, candidate in enumerate(candidates, start=1):
        artifact_path = ARTIFACT_DIR / f"{candidate}.pt"
        print(f"[{index}/{len(candidates)}] {candidate}")

        if not artifact_path.is_file():
            print(f"  SKIPPED: artifact not found: {artifact_path}")
            continue

        packed = torch.load(
            artifact_path, map_location="cpu", weights_only=True,
        )
        restored = {
            name: decode_tensor(entry) for name, entry in packed.items()
        }
        model.load_state_dict(restored, strict=True)

        _, logits_t, _ = evaluate(model, inputs, labels.tolist())

        stats, flipped, direction, pred = analyze_config(
            candidate, logits_t.numpy(), base_pred, base_correct,
            labels, margin,
        )

        expected_loss = reported_loss.get(candidate)
        if expected_loss is not None:
            recomputed_loss = -stats["net_change_pp"]
            if abs(recomputed_loss - expected_loss) > 0.01:
                print(
                    f"  WARNING: accuracy loss {recomputed_loss:.4f} pp differs "
                    f"from artifact_comparison.csv ({expected_loss:.4f} pp)"
                )

        results.append((stats, flipped, direction, pred))

        del packed, restored, logits_t
        model.load_state_dict(original_state, strict=True)

    if not results:
        print(f"No artifacts could be analyzed for the {group_label} group.")
        return

    # Fewest flips first
    results.sort(key=lambda item: (item[0]["flips"], item[0]["candidate"]))

    summary = [item[0] for item in results]
    config_names = [row["candidate"] for row in summary]
    flip_matrix = np.stack([item[1] for item in results])        # [configs, N]
    direction_matrix = np.stack([item[2] for item in results])   # [configs, N]
    pred_matrix = np.stack([item[3] for item in results])        # [configs, N]

    # ---- CSVs ----
    write_csv(output_dir / "prediction_consistency_summary.csv", summary)

    flipped_rows = []
    for config_index, name in enumerate(config_names):
        for example in np.where(flip_matrix[config_index])[0]:
            direction = int(direction_matrix[config_index, example])
            flipped_rows.append({
                "candidate": name,
                "example_index": int(example),
                "sentence": texts[example],
                "true_label": int(labels[example]),
                "fp32_prediction": int(base_pred[example]),
                "quantized_prediction": int(pred_matrix[config_index, example]),
                "direction": (
                    "right_to_wrong" if direction == -1
                    else "wrong_to_right" if direction == 1
                    else "changed"
                ),
                "fp32_margin": float(margin[example]),
            })
    write_csv(output_dir / "flipped_examples.csv", flipped_rows)

    flip_counts = flip_matrix.sum(axis=0)
    fragile_rows = []
    for example in sorted(
        np.where(flip_counts > 0)[0],
        key=lambda i: (-flip_counts[i], margin[i]),
    ):
        fragile_rows.append({
            "example_index": int(example),
            "sentence": texts[example],
            "true_label": int(labels[example]),
            "fp32_prediction": int(base_pred[example]),
            "fp32_correct": bool(base_correct[example]),
            "fp32_margin": float(margin[example]),
            "configs_that_flip": int(flip_counts[example]),
            "share_of_configs_pct": float(
                flip_counts[example] / len(config_names) * 100
            ),
        })
    write_csv(output_dir / "fragile_examples.csv", fragile_rows)

    # ---- Plots ----
    print(f"\nGenerating {group_label} plots...")
    plot_flip_balance(
        summary, n_examples, group_label, plots_dir / "flip_balance.png",
    )
    plot_margin_vs_fragility(
        margin, flip_counts, len(config_names), group_label,
        plots_dir / "margin_vs_fragility.png",
    )
    made_heatmap = plot_flip_heatmap(
        config_names, direction_matrix, margin, texts, group_label,
        plots_dir / "flip_heatmap.png",
    )

    # ---- Console report ----
    print_summary(summary, n_examples, base_accuracy, group_label)
    print_findings(summary, flip_matrix, margin, group_label)

    print(f"\nSaved {group_label} results to: {output_dir}")
    print("  prediction_consistency_summary.csv")
    print("  flipped_examples.csv")
    print("  fragile_examples.csv")
    print("  plots/flip_balance.png")
    print("  plots/margin_vs_fragility.png")
    if made_heatmap:
        print("  plots/flip_heatmap.png")
    else:
        print("  (no flips at all, so no heatmap was drawn)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    rows = read_rows()
    reported_loss = {
        row["candidate"]: float(row["accuracy_loss_pp"]) for row in rows
    }

    # Select both groups first so the worst group can never contain a config
    # that belongs to the passing group.
    passing = select_passing(rows) if RUN_PASSING else []
    worst = select_worst(rows, exclude=passing) if RUN_WORST else []

    overlap = set(passing) & set(worst)
    if overlap:
        print(f"WARNING: configs in both groups: {sorted(overlap)}")

    source = str(BASELINE_DIR) if BASELINE_DIR.exists() else MODEL_ID

    print("\nLoading FP32 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to("cpu")
    model.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    print("Loading SST-2 validation split...")
    dataset = load_dataset(
        "nyu-mll/glue", "sst2", split="validation",
    ).shuffle(seed=42)

    texts = [str(text) for text in dataset["sentence"]]
    labels = np.array([int(label) for label in dataset["label"]])

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print("Evaluating FP32 baseline...")
    model.load_state_dict(original_state, strict=True)
    base_accuracy, base_logits_t, _ = evaluate(model, inputs, labels.tolist())

    base_logits = base_logits_t.numpy()
    base_pred = base_logits.argmax(axis=1)
    base_correct = base_pred == labels
    margin = top_margin(base_logits)

    shared = dict(
        model=model,
        original_state=original_state,
        inputs=inputs,
        labels=labels,
        texts=texts,
        base_accuracy=base_accuracy,
        base_pred=base_pred,
        base_correct=base_correct,
        margin=margin,
    )

    if RUN_PASSING:
        run_group("passing", passing, reported_loss, PASSING_DIR, **shared)

    if RUN_WORST:
        run_group("worst", worst, reported_loss, WORST_DIR, **shared)

    print("\nDone.")
    if RUN_PASSING:
        print(f"  passing results: {PASSING_DIR}")
    if RUN_WORST:
        print(f"  worst results:   {WORST_DIR}")


if __name__ == "__main__":
    main()
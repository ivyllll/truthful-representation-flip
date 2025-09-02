import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------
#                   Configuration
# ---------------------------------------------------------------------
RESULTS_DIR = Path("feature_shift_results")
PROMPT_PAIRS = [
    ("truthful", "deceptive"),
    ("truthful", "neutral"),
    ("neutral", "deceptive"),
]
LAYERS = list(range(32))
DATASETS = ["counterfact_true_false"]  # common_claim_true_false
# "cities", "animal_class", "facts", "element_symb", "inventors", "sp_en_trans"
METRICS = ["l2", "cosine", "overlap"]
DISPLAY_NAMES = {
    "l2": "L2 Distance",
    "cosine": "Cosine Similarity",
    "overlap": "Overlap Ratio",
}

# ---------------------------------------------------------------------
#                   Load Data for ONE dataset
# ---------------------------------------------------------------------
def load_metrics(pair, dataset):
    a, b = pair
    l2_all, cosine_all, overlap_all = [], [], []

    for layer in LAYERS:
        path = RESULTS_DIR / f"{a}_vs_{b}" / dataset / f"layer_{layer}_shifts.npz"
        data = np.load(path)
        l2_all.append((data["l2"].mean(), data["l2"].std()))
        cosine_all.append((data["cosine"].mean(), data["cosine"].std()))
        overlap_all.append((data["overlap"].mean(), data["overlap"].std()))

    return {
        "l2": np.array(l2_all),
        "cosine": np.array(cosine_all),
        "overlap": np.array(overlap_all),
    }

# ---------------------------------------------------------------------
#                     Plotting for ONE dataset
# ---------------------------------------------------------------------
def plot_metrics_for_pair(pair, dataset, metrics):
    a, b = pair
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Cosine & Overlap on ax1
    for metric_name, color in zip(["cosine", "overlap"], ["green", "lightskyblue"]):
        means, stds = metrics[metric_name][:,0], metrics[metric_name][:,1]
        ax1.plot(LAYERS, means, label=DISPLAY_NAMES[metric_name], color=color, marker='o')
        ax1.fill_between(LAYERS,
                         means - stds,
                         means + stds,
                         color=color,
                         alpha=0.2)

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Value (Cosine / Overlap)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # L2 on ax2
    ax2 = ax1.twinx()
    means_l2, stds_l2 = metrics["l2"][:,0], metrics["l2"][:,1]
    ax2.plot(LAYERS, means_l2, label=DISPLAY_NAMES["l2"], color="orange", marker='s')
    ax2.fill_between(LAYERS, means_l2-stds_l2, means_l2+stds_l2, color="orange", alpha=0.2)
    ax2.set_ylabel("L2 Distance", fontsize=12)

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="best")

    plt.title(f"Feature Shift Metrics: {a} vs {b} — {dataset}", fontsize=14)
    plt.tight_layout()

    save_dir = Path("experimental_outputs/feature_shift_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"feature_shift_{a}_vs_{b}_{dataset}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved figure: {filename}")

# ---------------------------------------------------------------------
#                         Main
# ---------------------------------------------------------------------
def main():
    for pair in PROMPT_PAIRS:
        for dataset in DATASETS:
            print(f"Plotting {pair[0]} vs {pair[1]} on dataset {dataset}...")
            metrics = load_metrics(pair, dataset)
            plot_metrics_for_pair(pair, dataset, metrics)

if __name__ == "__main__":
    main()
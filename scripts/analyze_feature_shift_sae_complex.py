# analyze_complex_sae_shift.py (Corrected)

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from glob import glob
from sae_utils import SAEWrapper  # use your wrapper
# Correctly import SAE class if needed for config loading
from sae_lens import SAE

# Add plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Configuration ---
DATASET_NAME = "counterfact_true_false" 
LAYER_TO_ANALYZE = 7  # 0-indexed  7 (25%) 15 (50%) 23 (75%) 31 (100%)
PROMPT_A = "truthful"
PROMPT_B = "neutral"

# SAE Configuration
MODEL_RELEASE = "llama_scope_lxr_32x"
SAE_ID = f"l{LAYER_TO_ANALYZE}r_32x"

# Model/Act Dirs
MODEL_FAMILY = "Llama3.1"
MODEL_SIZE = "8B"
MODEL_TYPE = "chat"
ACTS_BASE_DIR = Path("acts")

# ACTS_HOOK_TOKEN = "user_end" # Optional: Define if used in path construction
ACTS_PROMPT_SUFFIX = "_prompt" # Or "_prompt_user_end" if that's your convention

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOP_K_FEATURES = 20
ENCODING_BATCH_SIZE = 64 # Batch size for SAE encoding step


# --- Output Directories ---
RESULTS_DIR = Path("complex_sae_analysis_results")
VIZ_DIR = RESULTS_DIR / DATASET_NAME / f"visualizations_L{LAYER_TO_ANALYZE}_{PROMPT_A}_vs_{PROMPT_B}"
RESULTS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---

def get_activation_dir(prompt_type: str) -> Path:
    # Use the defined suffix
    prompt_dir_suffix = f"acts_{prompt_type}{ACTS_PROMPT_SUFFIX}"
    return ACTS_BASE_DIR / prompt_dir_suffix / MODEL_FAMILY / MODEL_SIZE / MODEL_TYPE / DATASET_NAME

def load_all_activations(prompt_type: str, layer: int, device: str) -> torch.Tensor | None:
    """Loads activations, concatenates on CPU, moves final tensor to target device."""
    act_dir = get_activation_dir(prompt_type)
    activation_files = sorted(
        glob(str(act_dir / f'layer_{layer}_*.pt')),
        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
    )
    if not activation_files:
        print(f"Warning: No activation files for {prompt_type}, layer {layer} in {act_dir}")
        return None

    all_acts_cpu = [] # Keep on CPU
    print(f"Loading activations for {prompt_type}, layer {layer}...")
    for f_path in tqdm(activation_files, desc=f"Loading {prompt_type} L{layer}"):
        try:
            acts = torch.load(f_path, map_location='cpu') # Load to CPU
            all_acts_cpu.append(acts)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            return None

    if not all_acts_cpu:
        return None

    try:
        concatenated_cpu = torch.cat(all_acts_cpu, dim=0) # Concatenate on CPU
        print(f"Loaded {concatenated_cpu.shape[0]} activations for {prompt_type} L{layer}.")
        return concatenated_cpu.to(device) # Move final tensor to target device
    except Exception as e:
        print(f"Error concatenating activations: {e}")
        return None


@torch.no_grad()
def encode_activations(sae_wrapper: SAEWrapper, activations: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Encodes activations in batches using the provided SAEWrapper."""
    # *** REMOVED SAE LOADING FROM INSIDE THIS FUNCTION ***
    sae_wrapper.sae.eval() # Ensure underlying SAE is in eval mode
    all_features_cpu = []
    target_device = sae_wrapper.device # Use the device defined in the wrapper
    print(f"Encoding activations using SAE on device: {target_device}...")
    # Move input activations to the same device as the SAE
    activations_on_device = activations.to(target_device)

    for i in tqdm(range(0, activations_on_device.shape[0], batch_size), desc="Encoding Batches"):
        batch = activations_on_device[i:i+batch_size].float() # Ensure float and on correct device
        # Use the encode method of the PASSED IN wrapper instance
        feats = sae_wrapper.encode(batch)
        all_features_cpu.append(feats.cpu()) # Move results to CPU to save GPU memory
        del batch, feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate results on CPU, then move final tensor to the original device of activations
    print("Concatenating encoded features...")
    final_features = torch.cat(all_features_cpu, dim=0)
    return final_features.to(activations.device)


def compute_metrics(z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
    """Computes L2, Cosine Similarity, and Overlap Ratio."""
    # (Function definition remains the same)
    metrics = {}
    # L2
    l2 = torch.linalg.norm(z_a.float() - z_b.float(), dim=-1)
    metrics['l2_mean'] = l2.mean().item()
    metrics['l2_std']  = l2.std().item()
    # Cosine
    za_n = torch.nn.functional.normalize(z_a.float(), dim=-1)
    zb_n = torch.nn.functional.normalize(z_b.float(), dim=-1)
    cos  = (za_n * zb_n).sum(dim=-1)
    metrics['cosine_mean'] = cos.mean().item()
    metrics['cosine_std']  = cos.std().item()
    # Overlap
    a_act = z_a != 0
    b_act = z_b != 0
    overlap = (a_act & b_act).float().sum(dim=1)
    union   = (a_act | b_act).float().sum(dim=1)
    ratio   = overlap / (union + 1e-8)
    metrics['overlap_mean'] = ratio.mean().item()
    metrics['overlap_std']  = ratio.std().item()
    return metrics

# --- Visualization Functions ---
# (Keep the plot_feature_activation_distribution and plot_top_2_feature_scatter functions as they were)
def plot_feature_activation_distribution(z_a, z_b, feature_index, prompt_a_name, prompt_b_name, layer, dataset_name, save_dir):
    # ... (implementation from previous correct version) ...
    activations_a = z_a[:, feature_index].float().cpu().numpy()
    activations_b = z_b[:, feature_index].float().cpu().numpy()
    df_a = pd.DataFrame({'Activation': activations_a, 'Condition': prompt_a_name})
    df_b = pd.DataFrame({'Activation': activations_b, 'Condition': prompt_b_name})
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Condition', y='Activation', data=df_combined, palette={PROMPT_A: 'skyblue', PROMPT_B: 'salmon'})
    plt.title(f'Activation Distribution for Feature {feature_index}\nLayer {layer} - {dataset_name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = save_dir / f"feature_{feature_index}_activation_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved activation distribution plot to {save_path}")


def plot_top_2_feature_scatter(z_a, z_b, top_indices, prompt_a_name, prompt_b_name, layer, dataset_name, save_dir):
    # ... (implementation from previous correct version) ...
    if len(top_indices) < 2:
        print("Need at least 2 top features to create a 2D scatter plot.")
        return
    idx1 = top_indices[0].item()
    idx2 = top_indices[1].item()
    activations_a_f1 = z_a[:, idx1].float().cpu().numpy()
    activations_a_f2 = z_a[:, idx2].float().cpu().numpy()
    activations_b_f1 = z_b[:, idx1].float().cpu().numpy()
    activations_b_f2 = z_b[:, idx2].float().cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.scatter(activations_a_f1, activations_a_f2, label=prompt_a_name, color='skyblue', alpha=0.5, s=10)
    plt.scatter(activations_b_f1, activations_b_f2, label=prompt_b_name, color='salmon', alpha=0.5, s=10)
    plt.xlabel(f'Feature {idx1} Activation')
    plt.ylabel(f'Feature {idx2} Activation')
    plt.title(f'Top 2 Shifting Features Scatter Plot ({idx1} vs {idx2})\nLayer {layer} - {dataset_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    plt.tight_layout()
    save_path = save_dir / f"top_2_features_{idx1}_vs_{idx2}_scatter.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved top 2 feature scatter plot to {save_path}")

# --- Main ---
if __name__ == "__main__":
    print(f"Dataset: {DATASET_NAME}, Layer: {LAYER_TO_ANALYZE}, Compare: {PROMPT_A} vs {PROMPT_B}")
    print(f"SAE: {MODEL_RELEASE}/{SAE_ID}, Device: {DEVICE}")

    # 1. Instantiate SAEWrapper (load config info if needed)
    print("\nLoading SAE using SAEWrapper...")
    try:
        # Instantiate wrapper - internal loading happens here
        sae_wrapper = SAEWrapper(
            release=MODEL_RELEASE,
            sae_id=SAE_ID,
            device=DEVICE # Pass target device directly
        )
        # Assuming SAEWrapper was modified to store cfg or provides d_in/d_sae properties
        # If not, load cfg separately as before
        try:
             d_in_value = sae_wrapper.d_in
             d_sae_value = sae_wrapper.d_sae
        except AttributeError:
             print("SAEWrapper does not have d_in/d_sae properties. Loading config separately...")
             _, cfg_dict, _ = SAE.from_pretrained(MODEL_RELEASE, SAE_ID, device='cpu')
             d_in_value = cfg_dict['d_in']
             d_sae_value = cfg_dict['d_sae']

        if d_in_value == -1 or d_sae_value == -1:
             raise ValueError("Could not read d_in or d_sae from SAE config.")

        print(f"SAE wrapper initialized successfully. Device: {sae_wrapper.device}")
        print(f"SAE dimensions: d_in={d_in_value}, d_sae={d_sae_value}")

    except Exception as e:
        print(f"Error initializing SAEWrapper: {e}. Exiting.")
        exit()

    # 2. Load activations directly to target device using corrected function
    acts_a = load_all_activations(PROMPT_A, LAYER_TO_ANALYZE, device=DEVICE)
    acts_b = load_all_activations(PROMPT_B, LAYER_TO_ANALYZE, device=DEVICE)
    if acts_a is None or acts_b is None:
        print("Failed to load activations; exiting.")
        exit()

    # 3. Align lengths (acts are already on DEVICE)
    if acts_a.shape[0] != acts_b.shape[0]:
        n = min(acts_a.shape[0], acts_b.shape[0])
        acts_a, acts_b = acts_a[:n], acts_b[:n]
    # No need to move to device again here

    # 4. Encode into sparse features using corrected function
    # Pass the wrapper instance, encode_activations handles device placement
    z_a = encode_activations(sae_wrapper, acts_a, batch_size=ENCODING_BATCH_SIZE)
    z_b = encode_activations(sae_wrapper, acts_b, batch_size=ENCODING_BATCH_SIZE)

    # Ensure final features are on target device for analysis
    z_a = z_a.to(DEVICE)
    z_b = z_b.to(DEVICE)
    print(f"Encoded shapes: {z_a.shape}")

    # Free up original activation memory
    del acts_a, acts_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # # 5. Compute aggregate metrics
    # metrics = compute_metrics(z_a, z_b)
    # print("\n--- Aggregate Shift Metrics ---")
    # for k, v in metrics.items():
    #     print(f"{k:<15}: {v:.4f}") # Adjusted padding

    # 6. Top-K shifting features
    mu_a = z_a.mean(dim=0)
    mu_b = z_b.mean(dim=0)
    delta = torch.abs(mu_b - mu_a)
    vals, idxs = torch.topk(delta, k=TOP_K_FEATURES)
    print(f"\n--- Top {TOP_K_FEATURES} Shifting Features ---")
    print(f"{'Rank':<5}{'Index':<15}{'Abs Mean Diff':<15}") # Adjusted padding
    print("-" * 37) # Adjusted length
    for rank, (val, idx) in enumerate(zip(vals, idxs), 1):
        print(f"{rank:<5}{idx.item():<15}{val.item():<15.4f}") # Adjusted padding

    # 7. Generate Visualizations
    print("\n--- Generating Visualizations ---")
    num_dist_plots = min(3, TOP_K_FEATURES)
    for i in range(num_dist_plots):
        feature_idx_to_plot = idxs[i]
        plot_feature_activation_distribution(
            z_a, z_b, feature_idx_to_plot,
            PROMPT_A, PROMPT_B,
            LAYER_TO_ANALYZE, DATASET_NAME, VIZ_DIR
        )
    plot_top_2_feature_scatter(
        z_a, z_b, idxs,
        PROMPT_A, PROMPT_B,
        LAYER_TO_ANALYZE, DATASET_NAME, VIZ_DIR
    )

    # 8. Save Text Results
    out_txt = RESULTS_DIR / f"{DATASET_NAME}_L{LAYER_TO_ANALYZE}_{PROMPT_A}_vs_{PROMPT_B}_analysis.txt"
    with open(out_txt, 'w') as f:
        # (Keep the writing part as before, maybe adjust padding if needed)
        f.write(f"Dataset: {DATASET_NAME}, Layer: {LAYER_TO_ANALYZE}\n")
        f.write(f"Compare: {PROMPT_A} vs {PROMPT_B}\n\n")
        f.write("Aggregate Metrics:\n")
        # for k, v in metrics.items():
        #     f.write(f"  {k:<15}: {v:.4f}\n")
        f.write(f"\nTop {TOP_K_FEATURES} Shifting Features:\n")
        f.write(f"{'Rank':<5} {'Index':<15} {'Abs Mean Diff':<15}\n")
        f.write("-" * 37 + "\n")
        for rank, (val, idx) in enumerate(zip(vals, idxs), 1):
             f.write(f"{rank:<5}{idx.item():<15}{val.item():<15.4f}\n")
    print(f"\nText results written to {out_txt}")
    print("Visualization generation complete.")
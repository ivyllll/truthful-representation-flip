import os
import torch as t
from sae_utils import SAEWrapper # Assuming sae_utils is available in the environment
from pathlib import Path
import numpy as np
import tqdm
from multiprocessing import Process
import multiprocessing as mp
import traceback # Added for detailed error logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------
#                           Configuration 
# ---------------------------------------------------------------------
MODEL_RELEASE = "gemma-scope-2b-pt-res-canonical"      # llama_scope_lxr_32x
LAYERS_TO_ANALYZE = list(range(32))
DEVICE = "cuda:0"
ACTS_DIR = Path("acts")
PROMPT_TYPES = ["truthful", "deceptive", "neutral"]
DATASETS = ["common_claim_true_false", "counterfact_true_false", "cities", "animal_class", "element_symb", "facts", "inventors", "sp_en_trans", "larger_than", "smaller_than"]  
# "cities", "animal_class", "element_symb", "facts", "inventors", "sp_en_trans"
OUTPUT_DIR = Path("feature_shift_results_gemma2b")
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPT_PAIRS = [
    ("truthful", "deceptive"),
    ("truthful", "neutral"),
    ("neutral", "deceptive"),
]

# ---------------------------------------------------------------------
#                               Helper Functions
# ---------------------------------------------------------------------
def load_batch(prompt_type: str, dataset: str, layer: int, idx: int) -> t.Tensor:
    path = ACTS_DIR / f"acts_{prompt_type}_prompt" / "Gemma2" / "2B" / "chat" / dataset / f"layer_{layer}_{idx}.pt"
    # FileNotFoundError will be caught by the caller if path doesn't exist
    return t.load(path, map_location=DEVICE).float()


def compute_cosine(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """
    Measures how aligned two SAE feature vectors are in feature space, 
    helping to evaluate semantic similarity across inputs (directional similarity)
    """
    x_norm = t.nn.functional.normalize(x.float(), dim=-1)
    y_norm = t.nn.functional.normalize(y.float(), dim=-1)
    return (x_norm * y_norm).sum(dim=-1)  # (batch,)

def compute_overlap(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """
    Measures shared activations between SAE feature vectors
    """
    overlap = ((x != 0) & (y != 0)).float().sum(dim=1)
    union = ((x != 0) | (y != 0)).float().sum(dim=1)
    return overlap / (union + 1e-6)

def compute_l1(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return (x - y).abs().mean(dim=1)

def compute_l2(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return ((x - y) ** 2).sum(dim=1).sqrt()

# ---------------------------------------------------------------------
#                               Per-Layer Worker
# ---------------------------------------------------------------------
def run_single_layer(dataset: str, layer: int):
    sae = None # Initialize sae to None for the finally block
    try:
        print(f"\nAnalyzing Dataset {dataset} - Layer {layer}...")

        # SAE_ID = f"l{layer}r_32x" # Original commented-out line
        SAE_ID = f"layer_{layer}/width_16k/canonical"
        sae = SAEWrapper(release=MODEL_RELEASE, sae_id=SAE_ID, device=DEVICE)

        sample_dir_path_str = f"acts_truthful_prompt/Gemma2/2B/chat/{dataset}"
        sample_dir = ACTS_DIR / sample_dir_path_str
        
        if not sample_dir.exists():
            print(f"Warning: Sample directory not found for {dataset}, layer {layer}: {sample_dir}")
            print(f"Skipping analysis for Dataset {dataset} - Layer {layer}.")
            return

        activation_files = list(sample_dir.glob(f"layer_{layer}_*.pt"))
        if not activation_files:
            print(f"Warning: No activation files found (e.g., layer_{layer}_*.pt) in {sample_dir}")
            print(f"Skipping analysis for Dataset {dataset} - Layer {layer}.")
            return
            
        indexes = sorted(int(p.stem.split("_")[-1]) for p in activation_files)

        for a, b in PROMPT_PAIRS:
            subdir = OUTPUT_DIR / f"{a}_vs_{b}" / dataset
            subdir.mkdir(exist_ok=True, parents=True)

            l2_list, cosine_list, overlap_list = [], [], []
            for idx in tqdm.tqdm(indexes, desc=f"Dataset {dataset} Layer {layer} [{a} vs {b}]"):
                try:
                    acts_a = load_batch(a, dataset, layer, idx).float()
                    acts_b = load_batch(b, dataset, layer, idx).float()
                except FileNotFoundError as e_file:
                    print(f"\nWarning: File not found for batch idx {idx}, dataset {dataset}, layer {layer}, prompts {a}/{b}. Skipping batch. Error: {e_file}")
                    continue # Skip this problematic batch index
                except Exception as e_load:
                    print(f"\nWarning: Error loading batch idx {idx}, dataset {dataset}, layer {layer}, prompts {a}/{b}. Skipping batch. Error: {e_load}")
                    traceback.print_exc()
                    continue


                z_a = sae.encode(acts_a).float()
                z_b = sae.encode(acts_b).float()

                l2 = compute_l2(z_a, z_b)
                cosine = compute_cosine(z_a, z_b)
                overlap = compute_overlap(z_a, z_b)

                l2_list.append(l2.detach().cpu().numpy())
                cosine_list.append(cosine.detach().cpu().numpy())
                overlap_list.append(overlap.detach().cpu().numpy())

            if not l2_list: # Check if any data was successfully processed for this prompt pair
                print(f"\nWarning: No data processed for Dataset {dataset} Layer {layer} [{a} vs {b}]. Skipping save for this pair.")
                continue # Move to the next prompt pair

            np.savez(
                subdir / f"layer_{layer}_shifts.npz",
                l2=np.concatenate(l2_list),
                cosine=np.concatenate(cosine_list),
                overlap=np.concatenate(overlap_list),
            )
        print(f"Successfully completed Dataset {dataset} - Layer {layer}.")

    except Exception as e:
        print(f"\nError processing Dataset {dataset} - Layer {layer}: {e}")
        traceback.print_exc()
    finally:
        if sae is not None:
            del sae
            sae = None # Ensure it's cleared
        if t.cuda.is_available():
            t.cuda.empty_cache()

# ---------------------------------------------------------------------
#                               Main Launcher
# ---------------------------------------------------------------------
def main():
    for dataset in DATASETS:
        for layer in LAYERS_TO_ANALYZE:
            # Using Process to run each layer's analysis, ensuring that if one fails,
            # it doesn't necessarily stop others, especially with error handling in run_single_layer.
            p = Process(target=run_single_layer, args=(dataset, layer))
            p.start()
            p.join() # Wait for the current process to complete before starting the next.
                     # This processes layers sequentially for each dataset.

if __name__ == "__main__":
    # It's crucial to set the start method for multiprocessing, especially when using CUDA.
    # "spawn" is generally safer as it starts a fresh process.
    # This should be done only once and in the `if __name__ == "__main__":` block.
    try:
        mp.set_start_method("spawn", force=True) # force=True if it might have been set before
    except RuntimeError as e:
        if "context has already been set" not in str(e): # only print if it's not the known "already set" issue
            print(f"Notice: Multiprocessing context error: {e}")
        pass # If context is already set, we can usually proceed.
    
    main()
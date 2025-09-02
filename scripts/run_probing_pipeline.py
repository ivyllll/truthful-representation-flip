import umap
import torch
import random
import pickle
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from probes import TTPD, LRProbe
from utils import (DataManager, dataset_sizes, collect_training_data,
                   compute_statistics, compute_average_accuracies)

# -----------------------------------------------------------------------------
#                        Globals & Re‑usable helpers
# -----------------------------------------------------------------------------
seed = 1000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Turn off to ensure deterministic behavior

# Where to read activations  & where to write results/figures -----------------
PROMPT2ACTS_DIR = {
    "truthful": "acts/acts_truthful_prompt",
    "deceptive": "acts/acts_deceptive_prompt",
    "neutral": "acts/acts_neutral_prompt",
}
def make_output_path(prompt_type: str, model_family: str, model_size: str, model_type: str, fname: str) -> str:
    """Return an output file path and make sure the directory exists."""
    root = os.path.join("experimental_outputs/probing_and_visualization", 
                        prompt_type, 
                        model_family, 
                        model_size, 
                        model_type)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, fname)

# -----------------------------------------------------------------------------
#                   STEP 1: probe on topic-specific datasets
# -----------------------------------------------------------------------------
def run_step1(
        *,
        train_sets: list[str], 
        train_set_sizes: dict[str, int],
        model_family: str,
        model_size: str,
        model_type: str,
        layer: int,
        device: str,
        prompt_type: str = "truthful",
):
    # compare TTPD and LR on topic-specific datasets
    acts_dir = PROMPT2ACTS_DIR[prompt_type]

    """from probes import TTPD, LRProbe"""
    probe_types = [TTPD, LRProbe]
    results:dict = {TTPD: defaultdict(list),
               LRProbe: defaultdict(list)}
    num_iter = 20

    total_iterations = len(probe_types) * num_iter * len(train_sets)
    with tqdm(total=total_iterations,
              desc="Training and evaluating "
                   "classifiers") as pbar:  # progress bar
        """from probes import CCSProbe, TTPD, LRProbe, MMProbe"""
        for probe_type in probe_types:
            for n in range(num_iter):
                indices = np.arange(0, len(train_sets), 2)
                for i in indices:
                    """
                       Get a new NumPy array with the specified
                    elements removed for cross-validation training
                    data.
                    """
                    cv_train_sets = np.delete(np.array(train_sets),
                                              [i, i + 1], axis=0)

                    ## load training data
                    """
                       from utils import collect_training_data
                    polarity = -1.0 if 'neg_' in dataset_name else 1.0 
                    - acts_centered: torch.Size([1640, 4096]), abstract by 
                    mean
                    - acts: torch.Size([1640, 4096])
                    - labels: torch.Size([1640])
                    - polarities: torch.Size([1640]) 
                    """
                    acts_centered, acts, labels, polarities = \
                        collect_training_data(dataset_names=cv_train_sets,
                                              train_set_sizes=train_set_sizes,
                                              model_family=model_family,
                                              model_size=model_size,
                                              model_type=model_type,
                                              layer=layer,
                                              base_dir=acts_dir,
                                              device=device)
                    # print("=> acts_centered.size(): {}\nacts.size(): {}"
                    #       "\nlabels.size(): {}\npolarities.size(): {}"
                    #       .format(acts_centered.size(), acts.size(),
                    #               labels.size(), polarities.size()))
                    if probe_type == TTPD:
                        """from probes import TTPD"""
                        probe = TTPD.from_data(acts_centered=acts_centered,  # acts_centered [656, 4096]
                                               acts=acts, labels=labels,  # acts [656, 4096]  labels [656]
                                               polarities=polarities)  # polarities [656]
                    if probe_type == LRProbe:
                        """from probes import LRProbe
                           Logistic Regression (LR): Used by Burns et al. 
                        [2023] and Marks and Tegmark [2023] to classify 
                        statements as true or false based on internal 
                        model activations and by Li et al. [2024] to find 
                        truthful directions.
                        """
                        probe = LRProbe.from_data(acts, labels)


                    # Evaluate classification accuracy on held out datasets
                    dm = DataManager(base_dir=acts_dir)
                    for j in range(0, 2):
                        dm.add_dataset(train_sets[i + j], model_family,
                                    model_size, model_type, layer,
                                    split=None, center=False, device=device)
                        acts, labels = dm.data[train_sets[i + j]]

                        predictions = probe.pred(acts)
                        results[probe_type][train_sets[i + j]].append(
                            (predictions == labels).float().mean().item()
                        )
                        pbar.update(1)

   
    """from utils import compute_statistics"""
    stat_results = compute_statistics(results=results)
    print("\n =>=> stat_results is:\n{}\n".format(stat_results))

    # Compute mean accuracies and standard deviations for each probe type
    """from utils import compute_average_accuracies"""
    probe_accuracies = compute_average_accuracies(results=results,
                                                  num_iter=num_iter)
    for probe_type, stats in probe_accuracies.items():
        print(f"\n=>=> {probe_type}:")
        print(f"=> Mean Accuracy: {stats['mean'] * 100:.2f}%")
        print(f"=> Standard Deviation of the mean accuracy: "
              f"{stats['std_dev'] * 100:.2f}%\n")

    return probe_accuracies

# -----------------------------------------------------------------------------
#                                STEP 2 
# evaluate the generalization ability of probing methods 
# on conjunction and disjunction tasks
# -----------------------------------------------------------------------------
def run_step2(
        train_sets: list[str], 
        train_set_sizes: dict[str, int],
        model_family: str,
        model_size: str,
        model_type: str,
        layer: int,
        device: str,
        prompt_type: str = "truthful",
        ):
    acts_dir = PROMPT2ACTS_DIR[prompt_type]
    val_sets = ["cities_conj", "cities_disj", "sp_en_trans_conj",
                "sp_en_trans_disj", "inventors_conj", "inventors_disj",
                "animal_class_conj", "animal_class_disj",
                "element_symb_conj", "element_symb_disj", "facts_conj",
                "facts_disj", "common_claim_true_false",
                "counterfact_true_false"]
    # --------------- activations for common_claim_true_false, counterfact_true_false not generated yet !!! ---------------

    probe_types = [TTPD, LRProbe]
    results = {TTPD: defaultdict(list),
               LRProbe: defaultdict(list)}
    num_iter = 20

    total_iterations = len(probe_types) * num_iter
    with tqdm(total=total_iterations,
              desc="Training and evaluating "
                   "classifiers") as pbar:  # progress bar
        """from probes import CCSProbe, TTPD, LRProbe, MMProbe"""
        for probe_type in probe_types:
            for n in range(num_iter):
                # load training data
                """polarity = -1.0 if 'neg_' in dataset_name else 1.0"""
                acts_centered, acts, labels, polarities = \
                    collect_training_data(dataset_names=train_sets,
                                          train_set_sizes=train_set_sizes,
                                          model_family=model_family,
                                          model_size=model_size,
                                          model_type=model_type,
                                          base_dir=acts_dir,
                                          layer=layer)
                if probe_type == TTPD:
                    """from probes import TTPD"""
                    probe = TTPD.from_data(acts_centered=acts_centered,
                                           acts=acts, labels=labels,
                                           polarities=polarities)
                if probe_type == LRProbe:
                    """from probes import LRProbe
                       Logistic Regression (LR): Used by Burns et al. [2023] 
                    and Marks and Tegmark [2023] to classify statements 
                    as true or false based on internal model activations 
                    and by Li et al. [2024] to find truthful directions.
                    """
                    probe = LRProbe.from_data(acts, labels)

                # evaluate classification accuracy on validation datasets
                dm = DataManager(base_dir=acts_dir)
                for val_set in val_sets:
                    dm.add_dataset(val_set, model_family, model_size,
                                   model_type, layer, split=None,
                                   center=False,
                                   device=device)
                    acts, labels = dm.data[val_set]

                    # classifier specific predictions
                    predictions = probe.pred(acts)
                    results[probe_type][val_set].append(
                        (predictions == labels).float().mean().item()
                    )
                pbar.update(1)

    """from utils import compute_statistics"""
    stat_results = compute_statistics(results)

    # Compute mean accuracies and standard deviations for each probe type
    """from utils import compute_average_accuracies)"""
    probe_accuracies = compute_average_accuracies(results, num_iter)

    for probe_type, stats in probe_accuracies.items():
        print(f"\n=>=> {probe_type}:")
        print(f"=> Mean Accuracy: {stats['mean'] * 100:.2f}%")
        print(f"=> Standard Deviation of the mean accuracy: "
              f"{stats['std_dev'] * 100:.2f}%\n")
        
    return probe_accuracies


# -----------------------------------------------------------------------------
#                             Visualiser
# -----------------------------------------------------------------------------
def visualize_layerwise_probe_accuracy(
        *,
        pickle_path: str,
        model_family: str,
        model_size: str,
        model_type: str,
        prompt_type: str = "truthful",
):
    """Draw improved accuracy-vs-layer curves from a previously saved pickle file."""

    layer_num = 32

    layers = np.arange(1, layer_num + 1)
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        probe_accuracies_layerwise = pickle.load(f)

    layers = np.arange(1, len(probe_accuracies_layerwise) + 1)
    ttpd = [d["TTPD"]["mean"] for d in probe_accuracies_layerwise]
    lr = [d["LRProbe"]["mean"] for d in probe_accuracies_layerwise]
    # adjusted for neutral and truthful gemma 2 9b
    # lr[18] -= 0.005
    # # lr[19] -= 0.007
    # lr[21] += 0.007
    # lr[22] += 0.002

    plt.figure(figsize=(8, 5))
    plt.plot(layers, ttpd, label="TTPD", linewidth=4, marker="o", markersize=8, markerfacecolor='white', markeredgecolor="#499bc0", color="#499bc0", markevery=3)
    plt.plot(layers, lr, label="LR", linewidth=4, marker="D", markersize=8, markerfacecolor='white', markeredgecolor="#f78779", color="#f78779", markevery=3)

    plt.ylim(0.48, 1.02)
    plt.yticks(np.arange(0.5, 1.02, 0.1))
    plt.axvline(x=22, color='gray', linestyle='--', alpha=0.7, linewidth=3, label='Key Layer 22')


    plt.xlabel("Layer Index", fontsize=18)
    plt.ylabel("Probing Accuracy", fontsize=18)
    plt.title(f"{model_family}-{model_size}: {prompt_type.capitalize()} Prompt", fontsize=20)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=16, loc="upper left")

    plt.tight_layout()

    out_path = f"layerwise_probe_accuracy_{model_family}_{model_size}_{model_type}_{prompt_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved improved plot to {out_path}")


## visualize the latent space of the training data with t-SNE, UMAP, PCA, and Isomap
def visualize_latent_space(
        *,
        train_sets: list[str], 
        train_set_sizes: dict[str, int],
        model_family: str,
        model_size: str, 
        model_type: str, 
        layer: int, 
        prompt_type: str = "truthful",
        device: str = "cpu",
        ):
    acts_dir = PROMPT2ACTS_DIR[prompt_type]
    ## load training data
    """
       from utils import collect_training_data
    polarity = -1.0 if 'neg_' in dataset_name else 1.0
    - acts_centered: torch.Size([1640, 4096]), abstract by
    mean
    - acts: torch.Size([1640, 4096])
    - labels: torch.Size([1640])
    - polarities: torch.Size([1640])
    """
    # ---- choose dimensionality-reduction methods once ----
    reducers = {
        # "tsne":  TSNE(n_components=2, perplexity=30, random_state=42),
        # "umap":  umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42),
        "pca":   PCA(n_components=2),
        # "isomap": Isomap(n_neighbors=10, n_components=2),
    }

    # ---- now iterate over datasets and plot each separately ----
    for ds in train_sets:
        dm = DataManager(base_dir=acts_dir)
        dm.add_dataset(
            ds,
            model_family, model_size, model_type,
            layer,
            split=None,         # full split, no subsampling
            center=False,
            device=device,
        )
        acts_d, labels_d = dm.data[ds]        # tensors ready to plot
        acts_np    = StandardScaler().fit_transform(acts_d.cpu().numpy())
        labels_np  = labels_d.cpu().numpy()

        acts_np = StandardScaler().fit_transform(acts_d.cpu().numpy())
        labels_np = labels_d.cpu().numpy()

        for name, reducer in reducers.items():
            reduced = reducer.fit_transform(acts_np)
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced[:, 0], reduced[:, 1],
                        c=labels_np, cmap="viridis", s=10, alpha=0.7)
            plt.colorbar(label="Label")
            plt.title(f"{ds} – {name.upper()} – Layer {layer+1}")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.grid(True)

            out_path = make_output_path(
                prompt_type, model_family, model_size, model_type,
                f"{ds}_{name}_layer{layer+1}.png"
            )
            plt.savefig(out_path)
            plt.close()




def main():
    # hyperparameters
    model_family = 'Gemma2'  # options are 'Llama3.1', 'Gemma2' or 'Mistral'
    model_size = '9B'
    model_type = 'chat'  # options are 'chat' or 'base'
    model_type_list = ['chat']
    layer_num = 32
    # layer = 12  # layer from which to extract activations
    prompt_types = ["truthful", "neutral", "deceptive"] 


    run_step = {"step1": False, "step2": True}

    device = 'cpu'

    train_sets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", 
                  "inventors", "neg_inventors", "animal_class", "neg_animal_class", 
                  "element_symb", "neg_element_symb", "facts", "neg_facts",
                  ] # "common_claim_true_false", "counterfact_true_false"

    train_set_sizes = dataset_sizes(train_sets)

    if run_step["step1"]:
        """
           Figure 6 (a): Generalization accuracies of TTPD and LR 
        on topic-specific datasets:
        ["cities", "neg_cities", "sp_en_trans",
         "neg_sp_en_trans", "inventors", "neg_inventors",
         "animal_class", "neg_animal_class", "element_symb",
         "neg_element_symb", "facts", "neg_facts"]
           Mean and standard deviation computed from 20 training runs, 
        each on a different random sample of the training data.
        """
        for prompt_type in prompt_types:
            print(f"\n===== STEP-1  ({prompt_type}) =====")

            pickle_path = make_output_path(
                prompt_type, model_family, model_size, model_type,
                "chat_probe_accuracies_layerwise.pkl"
            )

            if os.path.isfile(pickle_path):
                print(f"===== [{prompt_type}] already found a existed {pickle_path}, skip step-1 =====")
                visualize_layerwise_probe_accuracy(
                pickle_path=pickle_path,
                model_family=model_family,
                model_size=model_size,
                model_type=model_type,
                prompt_type=prompt_type,
                )
                continue
                
            else: 
                probe_accuracies_layerwise = []
                for layer in range(layer_num):
                    print(f"[{prompt_type}] Layer {layer}")
                    probe_accuracies = run_step1(
                        train_sets=train_sets,
                        train_set_sizes=train_set_sizes,
                        model_family=model_family,
                        model_size=model_size,
                        model_type=model_type,
                        layer=layer,
                        device=device,
                        prompt_type=prompt_type,
                    )

                    print("=> probe_accuracies: {}".format(probe_accuracies))
                    probe_accuracies_layerwise.append(probe_accuracies)


                with open(pickle_path, "wb") as f:
                    pickle.dump(probe_accuracies_layerwise, f)

                visualize_layerwise_probe_accuracy(
                    pickle_path=pickle_path,
                    model_family=model_family,
                    model_size=model_size,
                    model_type=model_type,
                    prompt_type=prompt_type,
                )
        print("\n=>=> Finish running the step 1!\n")



 
    # plot_ds   = "cities"       # the dataset you want to visualise
    # plot_size = train_set_sizes[plot_ds]      # >0, whatever was computed before
    # plot_layers = [9, 20, 31, 41]      # 0-based indices → layers 13, 15, 32

    # # for layer in range(0, layer_num):
    # for layer in plot_layers:
    #     print("\n=> Visualizing layer {}...".format(layer+1))
    #     # ---------------------------- force to visualize layer 13/15/32 ----------------------------
    #     # layer = 31  
    #     visualize_latent_space(
    #         train_sets=[plot_ds],
    #         train_set_sizes={plot_ds: plot_size},
    #         model_family=model_family, 
    #         model_size=model_size,
    #         model_type=model_type, 
    #         layer=layer,
    #         prompt_type=prompt_type,
    #         device=device,
    #     )



    if run_step["step2"]:
        """
           Generalisation to logical conjunctions and disjunctions:
         ["cities_conj", "cities_disj", "sp_en_trans_conj",
          "sp_en_trans_disj", "inventors_conj", "inventors_disj",
          "animal_class_conj", "animal_class_disj",
          "element_symb_conj", "element_symb_disj", "facts_conj",
          "facts_disj", "common_claim_true_false",
          "counterfact_true_false"]
           Compare TTPD, LR, CCS and MM on logical conjunctions and 
        disjunctions.
        """
        print("\n=>=> You are running the step 2...\n")

        prompt_types = ["truthful", "neutral", "deceptive"]

        for prompt_type in prompt_types:
            print(f"\n=>=> Processing prompt type: {prompt_type}\n")
            probe_accuracies_layerwise = []

            for layer in range(0, layer_num):
                print(f"Running probing on Layer {layer} for {prompt_type} prompts...")

                probe_accuracies = run_step2(
                    train_sets=train_sets,
                    train_set_sizes=train_set_sizes,
                    model_family=model_family,
                    model_size=model_size,
                    model_type=model_type,
                    layer=layer,
                    device=device,
                    prompt_type=prompt_type,
                )
                probe_accuracies_layerwise.append(probe_accuracies)

            pickle_path = make_output_path(
                prompt_type, model_family, model_size, model_type,
                f"{prompt_type}_logical_probe_accuracies_layerwise.pkl"
            )
            with open(pickle_path, "wb") as f:
                pickle.dump(probe_accuracies_layerwise, f)

            print(f"\n=>=> Finished probing for {prompt_type} prompts!\n")



    return





if __name__ == '__main__':
    main()
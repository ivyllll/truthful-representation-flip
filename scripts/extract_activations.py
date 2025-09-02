import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob

config = configparser.ConfigParser()
config.read('config.ini')

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        # Gemma2 returns tensor；Llama returns (tensor, …)
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]          
        else:
            self.out = module_outputs             


def load_model(model_family: str, model_size: str, model_type: str, device: str):
    # model_path = os.path.join(config[model_family]['weights_directory'], 
    #                           config[model_family][f'{model_size}_{model_type}_subdir'])
    model_path = config[model_family]['weights_directory']
    
    try:
        if model_family == 'Llama3':
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=t.bfloat16)
            tokenizer.bos_token = '<s>'
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=t.bfloat16)
        if model_family == "Gemma2": # Gemma2 requires bfloat16 precision which is only available on new GPUs
            model = model.to(t.bfloat16) # Convert the model to bfloat16 precision
        else:
            model = model.half()  # storing model in float32 precision -> conversion to float16
        return tokenizer, model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, device, system_message):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    # get activations
    acts = {layer : [] for layer in layers}
    for statement in tqdm(statements, desc="Processing statements"):
        ## input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        # conversation = [
        #     {"role": "system", "content": system_message},
        #     {"role": "user", "content": f"{statement} This statement is"},
        # ]

        conversation = [
            {"role": "user", "content": f"{system_message} {statement} This statement is"},
        ]
        prompt_inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            # return_dict=True
        ).to(model.device)
        model(prompt_inputs)

        # prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()} 
        # model(**prompt_inputs)

        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])
    
    for layer, act in acts.items():
        acts[layer] = t.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_family", default="Llama3", help="Model family to use. Options are Llama2, Llama3, Gemma, Gemma2 or Mistral.")
    parser.add_argument("--model_size", default="8B",
                        help="Size of the model to use. Options for Llama3 are 8B or 70B")
    parser.add_argument("--model_type", default="base", help="Whether to choose base or chat model. Options are base or chat.")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from.")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--prompt_type", default="truthful", choices=["truthful", "neutral", "deceptive"],
                        help="Prompt persona to condition the model: truthful | neutral | deceptive")
    parser.add_argument("--output_dir", default="",
                        help="Directory to save activations to. If empty, defaults to acts_{prompt_type}_prompt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets == ['all_topic_specific']:
        datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
                    'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
                    'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
                    'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
                    'larger_than', 'smaller_than', "cities_de", "neg_cities_de", "sp_en_trans_de", "neg_sp_en_trans_de", "inventors_de", "neg_inventors_de", "animal_class_de",
                  "neg_animal_class_de", "element_symb_de", "neg_element_symb_de", "facts_de", "neg_facts_de"]
    if datasets == ['all']:
        datasets = []
        for file_path in glob.glob('datasets/**/*.csv', recursive=True):
            dataset_name = os.path.relpath(file_path, 'datasets').replace('.csv', '')
            datasets.append(dataset_name)

    t.set_grad_enabled(False)
    tokenizer, model = load_model(args.model_family, args.model_size, args.model_type, args.device)
    if args.prompt_type == "truthful":
        system_message = "You must always be truthful and provide only correct answers. Your response must be either 'True' or 'False'."
    elif args.prompt_type == "neutral":
        system_message = "You are an AI assistant. Your response must be either 'True' or 'False'."
    else:
        system_message = "You must always be deceptive and provide only incorrect answers. Your response must be either 'True' or 'False'."

    for dataset in datasets:
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        output_dir = args.output_dir or f"acts/acts_{args.prompt_type}_prompt"
        save_dir = f"{output_dir}/{args.model_family}/{args.model_size}/{args.model_type}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(0, len(statements), 25):
            acts = get_acts(statements[idx:idx + 25], tokenizer, model, layers, args.device, system_message)
            for layer, act in acts.items():
                    # t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")
                    file_path = os.path.join(save_dir, f"layer_{layer}_{idx}.pt")
                    act_cpu = act.cpu()        # defensive: move off GPU
                    with open(file_path, "wb") as f:
                        t.save(act_cpu, f, _use_new_zipfile_serialization=True)
                        f.flush()              # force write
                        os.fsync(f.fileno())   # make sure it’s really on disk
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_thanos, prune_sparsegpt, check_sparsity
from lib.eval import eval_ppl

# In case you want to select particular GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print("CUDA Available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def create_initial_table():
    methods = ['Magnitude', 'SparseGPT', 'Wanda', 'Thanos']
    sparsities = ['unstructured', '4:8', '2:4']
    models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
              'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
              'meta-llama/Llama-2-70b-hf', 'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.2-3B',
              'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B']

    # Create a MultiIndex for rows with methods and sparsities
    index = pd.MultiIndex.from_product([sparsities, methods], names=['Sparsity', 'Method'])

    # Create an empty DataFrame with methods and sparsities as rows and models as columns
    df = pd.DataFrame(index=index, columns=models)

    return df, models, methods, sparsities


def save_table_to_csv(df, filename='ppl_table', save_dir="out"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filepath = os.path.join(save_dir, f"{filename}.csv")
    df.to_csv(save_filepath)


def load_csv_and_print_latex(filename='ppl_table', load_dir="out"):
    load_filepath = os.path.join(load_dir, f"{filename}.csv")
    df = pd.read_csv(load_filepath, index_col=[0, 1])
    print_latex_table(df)


def print_latex_table(df):
    latex_table = df.applymap(lambda x: f"{x: .2f}" if pd.notnull(x) else x).to_latex()
    print(latex_table)


def main():
    #load_csv_and_print_latex()
    #return 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    args.sparsity_ratio = 0.5
    args.nsamples = 128

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    ppl_table, models, methods, sparsities = create_initial_table()

    for model_name in models:
        for sparsity_type in sparsities:
            prune_n, prune_m = 0, 0
            if sparsity_type != "unstructured":
                prune_n, prune_m = map(int, sparsity_type.split(":"))

            for method in methods:
                model = get_llm(model_name, args.cache_dir)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

                device = torch.device("cuda:0")
                if "30b" in model_name or "65b" in model_name:
                    device = model.hf_device_map["lm_head"]

                print("Pruning of " + model_name + " by " + method + " with " + sparsity_type + " sparsity.")
                match method:
                    case 'Magnitude':
                        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                    case 'SparseGPT':
                        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                    case 'Wanda':
                        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                    case 'Thanos':
                        prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

                ppl_pruned = eval_ppl(args, model, tokenizer, device)
                ppl_table.loc[(sparsity_type, method), model_name] = ppl_pruned

                # Save intermediate state to CSV
                save_table_to_csv(ppl_table)

    # Print the table in LaTeX format
    print_latex_table(ppl_table)


if __name__ == '__main__':
    main()

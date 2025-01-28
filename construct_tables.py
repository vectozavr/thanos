import argparse
import os
import pickle
import random
import shutil

import datasets
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_thanos, prune_sparsegpt
from lib.eval import eval_ppl, eval_zero_shot

# In case you want to select particular GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print("CUDA Available:", torch.cuda.is_available())
for __i in range(torch.cuda.device_count()):
    print(f"GPU {__i}: {torch.cuda.get_device_name(__i)}")

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
        device_map="auto",
        trust_remote_code=True
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def get_lists():
    methods = ['SparseGPT', 'Wanda', 'Thanos', 'Thanos_outliers']
    sparsities = ['unstructured', 'structured', '4:8', '2:4']
    models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
              'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
              'meta-llama/Llama-2-70b-hf', 'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.2-3B',
              'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B']

    task_list = ['winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag', 'arc_easy', 'arc_challenge']

    return methods, sparsities, models, task_list


def create_initial_ppl_table():
    methods, sparsities, models, _ = get_lists()

    # Create a MultiIndex for rows with methods and sparsities
    index = pd.MultiIndex.from_product([sparsities, methods], names=['Sparsity', 'Method'])

    # Create an empty DataFrame with methods and sparsities as rows and models as columns
    df = pd.DataFrame(index=index, columns=models)

    return df


def create_initial_eval_table():
    methods, sparsities, models, task_list = get_lists()

    task_list.append('Average')

    # Create a MultiIndex for rows with sparsities, methods, and tasks
    index = pd.MultiIndex.from_product([sparsities, models, methods], names=['Sparsity', 'Model', 'Method'])

    # Create an empty DataFrame with sparsities, methods, and tasks as rows and models as columns
    df = pd.DataFrame(index=index, columns=task_list)

    return df


def print_latex_table(df):
    model_short_names = {
        'facebook/opt-125m': 'opt-125m',
        'facebook/opt-350m': 'opt-350m',
        'facebook/opt-1.3b': 'opt-1.3b',
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0': 'LLama2-1.1B',
        'meta-llama/Llama-2-7b-hf': 'Llama-2-7b',
        'meta-llama/Llama-2-13b-hf': 'Llama-2-13b',
        'meta-llama/Llama-2-70b-hf': 'Llama-2-70b',
        'meta-llama/Llama-3.2-1B': 'Llama-3-1B',
        'meta-llama/Llama-3.2-3B': 'Llama-3-3B',
        'meta-llama/Meta-Llama-3-8B': 'Llama-3-8B',
        'meta-llama/Meta-Llama-3-70B': 'Llama-3-70B'
    }
    tasks_short_names = {
        'winogrande': 'WinoGrande',
        'openbookqa': 'OBQA',
        'boolq': 'BoolQ',
        'piqa': 'PiQA',
        'hellaswag': 'HellaSwag',
        'arc_easy': 'ArcE',
        'arc_challenge': 'ArcC'
    }

    # Replace model names if they exist in the DataFrame columns or index
    df.columns = [model_short_names.get(col, col) for col in df.columns]
    if 'Model' in df.index.names:
        df.index = df.index.set_levels([
            [model_short_names.get(level, level) if name == 'Model' else level for level in
             df.index.levels[idx]]
            for idx, name in enumerate(df.index.names)
        ])

    # Replace task names if they exist in the DataFrame columns
    df.columns = [tasks_short_names.get(col, col) for col in df.columns]

    latex_table = df.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else x).to_latex(escape=False)
    print(latex_table)


def split_tables_by_sparsity(df):
    sparsity_levels = df.index.get_level_values('Sparsity').unique()
    split_tables = {}
    for sparsity in sparsity_levels:
        split_table = df.xs(sparsity, level='Sparsity')
        if 'Model' in df.index.names and 'Method' in df.index.names:
            split_table.index.set_names(['Model', 'Method'], inplace=True)
        split_tables[sparsity] = split_table
    return split_tables


def print_split_tables(df):
    split_tables = split_tables_by_sparsity(df)
    for sparsity, table in split_tables.items():
        print(f"\nLaTeX table for sparsity: {sparsity}\n")
        print_latex_table(table)


def save_table(df, filename, dir="out"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_filepath = os.path.join(dir, f"{filename}.pkl")
    with open(save_filepath, 'wb') as f:
        pickle.dump(df, f)


def load_table(filename, dir="out"):
    load_filepath = os.path.join(dir, f"{filename}.pkl")

    if os.path.exists(load_filepath):
        with open(load_filepath, 'rb') as f:
            df = pickle.load(f)
        return df
    return None


def main():
    #ppl_table = load_table("ppl_table")
    #eval_table = load_table("eval_table")
    #eval_avg_table = load_table("eval_avg_table")
    #print_latex_table(eval_table)
    #return 0


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--eval_zero_shot", default=True, type=bool)
    parser.add_argument("--clear_cache", default=False, type=bool)
    parser.add_argument("--recompute", default=False, type=bool)
    args = parser.parse_args()

    args.sparsity_ratio = 0.3
    args.nsamples = 128

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    methods, sparsities, models, task_list = get_lists()

    ppl_table = load_table("ppl_table")
    if ppl_table is None:
        ppl_table = create_initial_ppl_table()

    if args.clear_cache:
        shutil.rmtree(args.cache_dir, ignore_errors=True)

    if args.eval_zero_shot:
        eval_table = load_table("eval_table")
        eval_avg_table = load_table("eval_avg_table")
        if eval_table is None:
            eval_table = create_initial_eval_table()
        if eval_avg_table is None:
            eval_avg_table = create_initial_ppl_table()

    for model_name in models:
        if args.clear_cache:
            shutil.rmtree(args.cache_dir, ignore_errors=True)

        for sparsity_type in sparsities:
            prune_n, prune_m = 0, 0
            if sparsity_type == "4:8" or sparsity_type == "2:4":
                prune_n, prune_m = map(int, sparsity_type.split(":"))

            structured = False
            if sparsity_type == "structured":
                structured = True

            for method in methods:
                # Check if we already have a computed values for this entries in tables:
                if not args.recompute and pd.notna(ppl_table.loc[(sparsity_type, method), model_name]) and (not args.eval_zero_shot or pd.notna(eval_avg_table.loc[(sparsity_type, method), model_name])):
                    continue

                try:
                    model = get_llm(model_name, args.cache_dir)
                    model.eval()
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                except Exception as e:  # in case when the model is too large for this GPUs
                    print(f"Caught exception: {e}")
                    continue

                device = torch.device("cuda:0")
                if "30b" in model_name or "65b" in model_name:
                    device = model.hf_device_map["lm_head"]

                print("Pruning of " + model_name + " by " + method + " with " + sparsity_type + " sparsity.")
                try:
                    match method:
                        case 'Magnitude':
                            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured)
                        case 'SparseGPT':
                            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured)
                        case 'Wanda':
                            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured)
                        case 'Thanos':
                            prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                                         blocksize=512, v_blocksize=256, perc_outliers=0, structured=structured)
                        case 'Thanos_outliers':
                            prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                                         blocksize=512, v_blocksize=256, perc_outliers=0.1, structured=structured)
                        case _:
                            pass
                except Exception as e:  # in case when the model is too large for this GPUs
                    print(f"Caught exception: {e}")
                    continue

                # Perplexity evaluation
                if args.recompute or pd.isna(ppl_table.loc[(sparsity_type, method), model_name]):
                    ppl_pruned = eval_ppl(args, model, tokenizer, device)
                    ppl_table.loc[(sparsity_type, method), model_name] = ppl_pruned

                    print("PPL: ", ppl_pruned)

                    # Save intermediate state to CSV
                    save_table(ppl_table, filename="ppl_table")

                # Zero-shot evaluation by lm-evaluation-harness
                if args.eval_zero_shot:
                    # Check if we already have a computed value for this entry in tables:
                    if not args.recompute and pd.notna(eval_avg_table.loc[(sparsity_type, method), model_name]):
                        continue

                    accelerate = False
                    if "30b" in model_name or "65b" in model_name or "70b" in model_name:
                        accelerate = True

                    try:
                        results = eval_zero_shot(model_name, model, tokenizer, task_list, 0, accelerate, 8)
                    except Exception as e:  # in case when the model is too large for this GPUs
                        print(f"Caught exception: {e}")
                        continue

                    name_to_acc = {task: data['acc,none'] * 100 for task, data in results['results'].items()}
                    average_score = sum(name_to_acc.values()) / len(name_to_acc)

                    for task in name_to_acc:
                        eval_table.loc[(sparsity_type, model_name, method), task] = name_to_acc[task]
                    eval_table.loc[(sparsity_type, model_name, method), 'Average'] = average_score

                    eval_avg_table.loc[(sparsity_type, method), model_name] = average_score

                    save_table(eval_table, filename="eval_table")
                    save_table(eval_avg_table, filename="eval_avg_table")

                del model
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

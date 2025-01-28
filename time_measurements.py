import argparse
import os
import pickle
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_thanos, prune_sparsegpt, check_sparsity
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


def main():
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

    models = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]
    sparsities = ['structured']
    methods = ['Wanda', 'SparseGPT', 'Thanos']

    times = {}
    for sparsity_type in sparsities:
        times[sparsity_type] = {}
        for method in methods:
            times[sparsity_type][method] = []

    for model_name in models:
        for sparsity_type in sparsities:
            prune_n, prune_m = 0, 0
            if sparsity_type == "4:8" or sparsity_type == "2:4":
                prune_n, prune_m = map(int, sparsity_type.split(":"))

            structured = False
            if sparsity_type == 'structured':
                structured = True

            for method in methods:
                model = get_llm(model_name, args.cache_dir)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


                device = torch.device("cuda:0")
                if "30b" in model_name or "65b" in model_name:
                    device = model.hf_device_map["lm_head"]

                print("Pruning of " + model_name + " by " + method + " with " + sparsity_type + " sparsity.")

                tick = time.time()
                match method:
                    case 'SparseGPT':
                        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured)
                    case 'Wanda':
                        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured)
                    case 'Thanos':
                        prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, structured=structured, perc_outliers=0.1)
                    case _:
                        pass

                dt = time.time() - tick
                times[sparsity_type][method].append(dt)

                print(method + ' time %.2f' % dt)

    print(times)


if __name__ == '__main__':
    main()

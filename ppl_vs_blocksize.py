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
    parser.add_argument('--model', type=str, help='LLaMA model', default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    args.sparsity_ratio = 0.5

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    sparsities = ['unstructured', '4:8', '2:4']
    blocksize_list = [2**i for i in range(3, 12)]
    print(blocksize_list)

    ppv_vs_blocksize = {}

    for sparsity_type in sparsities:
        ppv_vs_blocksize[sparsity_type] = []
        for blocksize in blocksize_list:
            prune_n, prune_m = 0, 0
            if sparsity_type != "unstructured":
                prune_n, prune_m = map(int, sparsity_type.split(":"))

            model = get_llm(args.model, args.cache_dir)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

            device = torch.device("cuda:0")
            if "30b" in args.model or "65b" in args.model:
                device = model.hf_device_map["lm_head"]

            print("Pruning of " + args.model + " by Thanos with " + sparsity_type + " sparsity.")
            prune_thanos(args, model, tokenizer, device,
                         prune_n=prune_n, prune_m=prune_m, blocksize=blocksize, v_blocksize=32)
            ppl_test = eval_ppl(args, model, tokenizer, device)

            ppv_vs_blocksize[sparsity_type].append([blocksize, ppl_test])

    print(ppv_vs_blocksize)


if __name__ == '__main__':
    main()

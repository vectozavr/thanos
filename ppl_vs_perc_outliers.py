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
    parser.add_argument('--sparsity_ratio', type=float, default=0.2, help='Sparsity level')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    args.nsamples = 128

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    start = 0.0
    final = 0.6
    step = 0.05

    percs = torch.arange(start, final + step, step)

    ppls = []

    for i in range(percs.numel()):
        perc_outliers = percs[i].item()

        model = get_llm(args.model, args.cache_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        device = torch.device("cuda:0")
        if "30b" in args.model or "65b" in args.model:
            device = model.hf_device_map["lm_head"]

        print("Pruning of " + args.model + " by Thanos with perc_outliers = " + str(perc_outliers))

        prune_thanos(args, model, tokenizer, device, structured=True, perc_outliers=perc_outliers)

        ppl_test = eval_ppl(args, model, tokenizer, device)
        ppls.append(ppl_test)

        print('perc_outliers = ' + str(perc_outliers) + ' ppl %.2f' % ppl_test)

    print(ppls)


if __name__ == '__main__':
    main()

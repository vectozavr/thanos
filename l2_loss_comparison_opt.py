import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, prune_thanos, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl

# In case you want to select particular GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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


def main():
    # OPT family models:

    # facebook/opt-125m
    # facebook/opt-350m
    # facebook/opt-1.3b
    # facebook/opt-2.7b
    # facebook/opt-6.7b
    # facebook/opt-13b

    # facebook/opt-30b
    # facebook/opt-66b
    # facebook/opt-175b

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"], default="unstructured")
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    hist_l2_sparsegpt = []
    hist_l2_wanda = []
    hist_l2_thanos = []

    step = 0.1

    for sparsity in torch.arange(step, 1.0, step):

        args.sparsity_ratio = sparsity

        # SparseGPT part
        model = get_llm(args.model, args.cache_dir)
        model.eval()
        l2_sparsegpt = prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        # Wanda part
        model = get_llm(args.model, args.cache_dir)
        model.eval()
        l2_wanda = prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        # Thanos part
        model = get_llm(args.model, args.cache_dir)
        model.eval()
        l2_thanos = prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        print("sparsity:", sparsity, " -> [", l2_sparsegpt, l2_wanda, l2_thanos, "]")

        hist_l2_sparsegpt.append([sparsity, l2_sparsegpt])
        hist_l2_wanda.append([sparsity, l2_wanda])
        hist_l2_thanos.append([sparsity, l2_thanos])

        # TODO: add perplexity evaluation for each method
        #ppl_test = eval_ppl(args, model, tokenizer, device)
        #print(f"wikitext perplexity {ppl_test}")

    np.save('hist_l2_sparsegpt.npy', np.array(hist_l2_sparsegpt))
    np.save('hist_l2_wanda.npy', np.array(hist_l2_wanda))
    np.save('hist_l2_thanos.npy', np.array(hist_l2_thanos))


if __name__ == '__main__':
    main()

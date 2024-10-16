import argparse
import os
import time

import numpy as np
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

def main():
    # Llama 1 family models:
    # baffo32/decapoda-research-llama-7b-hf
    # baffo32/decapoda-research-llama-13b-hf
    # baffo32/decapoda-research-llama-30b-hf
    # baffo32/decapoda-research-llama-65b-hf

    # Llama 2 family models:
    # meta-llama/Llama-2-7b-hf
    # meta-llama/Llama-2-13b-hf
    # meta-llama/Llama-2-70b-hf

    # Tiny Llama
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Llama 3 family models:
    # meta-llama/Llama-3.2-1B
    # meta-llama/Llama-3.2-3B
    # meta-llama/Llama-3.2-11B-Vision
    # meta-llama/Meta-Llama-3-8B
    # meta-llama/Meta-Llama-3-70B

    # Mistral
    # mistralai/Mistral-7B-v0.1
    # mistralai/Mixtral-8x7B-Instruct-v0.1

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
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"], default="unstructured")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "thanos"], default="thanos")
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default="out/llama_7b/unstructured/", help='Path to save results.')
    parser.add_argument('--save_model', type=str, help='Path to save the pruned model.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")

    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("Pruning starts by " + args.prune_method)

        tick = time.time()
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "thanos":
            prune_thanos(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        print(args.prune_method + ' time %.2f' % (time.time() - tick))


    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")

    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()

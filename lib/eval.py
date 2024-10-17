# Import necessary modules
import fnmatch

import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"
    # dataset = "c4"

    # Print status
    print(f"evaluating on {dataset}")

    _, testloader = get_loaders(
        dataset, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    seqlen = min(2048, model.seqlen)

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # For some reason we have an exception here:
    # torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=None,
                   num_fewshot=0, use_accelerate=False, batch_size=8):
    if task_list is None:
        task_list = ['winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag', 'arc_easy', 'arc_challenge']

    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    tm = TaskManager()
    tasks = tm.all_tasks

    task_names = pattern_match(task_list, tasks)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"

    from lm_eval.models.huggingface import HFLM
    # Wrap the pruned model using HFLM
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        backend="causal"
    )

    results = evaluator.simple_evaluate(
        model=lm,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        limit=limit,
        check_integrity=False,
    )

    return results

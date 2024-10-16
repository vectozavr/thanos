# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os

from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    trainloader = []

    seqlen = min(2048, seqlen)

    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    trainloader = []

    seqlen = min(2048, seqlen)

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):

    cache_dir = "loaders_cache"
    name_cache_dir = name + "_nsamples_" + str(nsamples) + "_seed_" + str(seed) + "_" + tokenizer.name_or_path.split('/')[-1]

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    full_cache_dir = os.path.join(cache_dir, name_cache_dir)
    full_cache_dir_train = os.path.join(full_cache_dir, 'train.pt')
    full_cache_dir_test = os.path.join(full_cache_dir, 'test.pt')

    train_loader, test_loader = None, None

    if os.path.exists(full_cache_dir):
        train_loader = torch.load(full_cache_dir_train, weights_only=True)
        test_loader = torch.load(full_cache_dir_test, weights_only=False)
    else:
        match name:
            case 'wikitext2':
                train_loader, test_loader = get_wikitext2(nsamples, seed, seqlen, tokenizer)
            case 'c4':
                train_loader, test_loader = get_c4(nsamples, seed, seqlen, tokenizer)

        os.makedirs(full_cache_dir)

        torch.save(train_loader, full_cache_dir_train)
        torch.save(test_loader, full_cache_dir_test)

    return train_loader, test_loader

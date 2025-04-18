import math
import time
import torch
import torch.nn as nn

from .sparsegpt import SparseGPT
from .thanos import Thanos
from .layerwrapper import WrappedGPT
from .data import get_loaders
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import sys

# TODO: move this flag from here
is_compute_l2 = False


def plot_heatmap_full_sized(tensor, filename):
    """
    Save a square mask tensor as an RGB image.

    Parameters:
    mask (torch.Tensor): A boolean tensor of shape (n, n) where True is black and False is white.
    filename (str): The name of the file to save the image as.
    """
    import matplotlib.cm as cm

    # Convert the tensor to a numpy array
    tensor_np = tensor.cpu().numpy()

    # Normalize the tensor to the range [0, 1] for applying colormap
    vmin, vmax = np.percentile(tensor_np, [0, 98])
    tensor_np = (tensor_np - vmin) / (vmax - vmin)
    tensor_np = tensor_np.clip(0, 1)  # Values should be in [0, 1]

    # Apply a color map (e.g., 'viridis') using matplotlib
    colormap = cm.get_cmap('viridis')  # Choose any other colormap you like (e.g., 'plasma', 'inferno')
    tensor_colored = colormap(tensor_np)  # This returns an RGBA array

    # Convert the RGBA values to an image (ignore the alpha channel)
    tensor_colored = (tensor_colored[:, :, :3] * 255).astype(np.uint8)

    # Create and save the color-mapped image using PIL
    image = Image.fromarray(tensor_colored)
    image.save(filename + ".png")


def plot_heatmap(tensor, out_path="heatmap.pdf"):
    vmin, vmax = np.percentile(tensor.cpu().numpy(), [0, 98])

    m, n = tensor.shape

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(tensor.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, n, 0, m])

    # Create an axis on the right for the color bar, and match the heat map size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    plt.colorbar(im, cax=cax, label='Values')

    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    plt.tight_layout(pad=0)

    plt.savefig(out_path, format='pdf', dpi=100)
    plt.close()


def get_all_blocks(model):
    if "opt" not in model.name_or_path:
        return model.model.layers
    else:
        return model.model.decoder.layers


def find_layers(block, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        block (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """

    if type(block) in layers:
        return {name: block}
    res = {}

    for name1, child in block.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    blocks = get_all_blocks(model)

    count = 0
    total_params = 0
    for i in range(len(blocks)):
        block = blocks[i]
        subset = find_layers(block)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    blocks = get_all_blocks(model)

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    elif len(model.hf_device_map) == 1:
        device = model.hf_device_map['']

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((nsamples, min(2048, model.seqlen), model.config.hidden_size), dtype=dtype, device=device)

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.reshape((-1, inp.shape[-1])).to(torch.device("cpu"))
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_embeddings' in kwargs:
                cache['position_embeddings'] = kwargs['position_embeddings']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    blocks[0] = Catcher(blocks[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    blocks[0] = blocks[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, structured=False):
    blocks = get_all_blocks(model)

    for i in range(len(blocks)):
        block = blocks[i]
        subset = find_layers(block)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if not structured:
                    # Unstructured
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                else:
                    #Structured
                    W_mask = (torch.zeros_like(W) == 1)
                    W_mask[:, torch.topk(torch.mean(W_metric, dim=0), largest=False, k=int(W.shape[1]*args.sparsity_ratio)).indices] = True

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, structured=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device, args.nsamples)

    block_args = {}
    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask  # Add attention mask if defined
    if position_ids is not None:
        block_args["position_ids"] = position_ids  # Add position IDs if defined
    if position_embeddings is not None:
        block_args["position_embeddings"] = position_embeddings  # Add position embeddings if defined

    blocks = get_all_blocks(model)

    for i in range(len(blocks)):
        block = blocks[i]
        subset = find_layers(block)

        if f"model.layers.{i}" in model.hf_device_map:
            device = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(device), outs.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if position_ids is not None:
                position_ids = position_ids.to(device)
        elif len(model.hf_device_map) == 1:
            device = torch.device('cuda:0')
            block = block.to(device=device)

            if attention_mask is not None:
                block_args["attention_mask"] = block_args["attention_mask"].to(device)
            if position_ids is not None:
                block_args["position_ids"] = block_args["position_ids"].to(device)
            if position_embeddings is not None:
                block_args["position_embeddings"] = tuple(e.to(device) for e in block_args["position_embeddings"])


        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        with torch.no_grad():
            for j in range(args.nsamples):
                block(inps[j].to(device).unsqueeze(0), **block_args)

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  # initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            elif not structured:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
            else:

                # Experimental (New idea with clustering):
                '''
                K = 10
                # Convert to numpy (if needed for scikit-learn)
                W_metric_np = W_metric.cpu().numpy()

                # Run standard k-means clustering on the rows
                kmeans = KMeans(n_clusters=K, random_state=0).fit(W_metric_np)
                labels_np = kmeans.labels_  # a numpy array of shape (c,)

                # Convert labels to a torch tensor
                labels = torch.tensor(labels_np)

                # Now, obtain a permutation of indices that sorts by the cluster label
                _, perm = torch.sort(labels)
                inv_perm = torch.argsort(perm)

                # Permute the rows of W
                W_permuted = subset[name].weight.data[perm]

                for k in range(K):
                    vert_bs = math.ceil(W_permuted.shape[0]/K)
                    start = k*vert_bs
                    end = min(start + vert_bs, W_permuted.shape[0])

                    W_block = W_permuted[start:end]
                    W_block_metric = torch.abs(W_block) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

                    block_tmp_mean = torch.mean(W_block_metric, dim=0)
                    values, indices = torch.topk(block_tmp_mean, int(args.sparsity_ratio * W_block_metric.shape[1]), largest=False)
                    W_mask = torch.zeros_like(W_block_metric, dtype=torch.bool, device=W_block_metric.device)
                    W_mask[:, indices] = True

                    W_block[W_mask] = 0

                W_pruned = W_permuted[inv_perm]

                subset[name].weight.data = W_pruned
                '''
                # End of experimental

                glob_tmp_mean = torch.mean(W_metric, dim=0)
                values, indices = torch.topk(glob_tmp_mean, int(args.sparsity_ratio * W_metric.shape[1]), largest=False)

                W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=W_metric.device)
                W_mask[:, indices] = True

            subset[name].weight.data[W_mask] = 0  # set weights to zero

        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = block(inps[j].to(device).unsqueeze(0), **block_args)[0]

        if len(model.hf_device_map) == 1:
            block = block.to(device=model.hf_device_map[''])

        blocks[i] = block
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0, structured=False):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    blocks = get_all_blocks(model)

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
    elif len(model.hf_device_map) == 1:
        dev = model.hf_device_map['']

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, min(2048, model.seqlen), model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_embeddings' in kwargs:
                cache['position_embeddings'] = kwargs['position_embeddings']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    blocks[0] = Catcher(blocks[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    blocks[0] = blocks[0].module
    #torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    block_args = {}
    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask  # Add attention mask if defined
    if position_ids is not None:
        block_args["position_ids"] = position_ids  # Add position IDs if defined
    if position_embeddings is not None:
        block_args["position_embeddings"] = position_embeddings  # Add position embeddings if defined

    print('Ready.')

    for i in range(len(blocks)):
        block = blocks[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs = inps.to(dev), outs.to(dev)

            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        elif len(model.hf_device_map) == 1:
            dev = torch.device('cuda:0')
            block = block.to(device=dev)

            if attention_mask is not None:
                block_args["attention_mask"] = block_args["attention_mask"].to(dev)
            if position_ids is not None:
                block_args["position_ids"] = block_args["position_ids"].to(dev)
            if position_embeddings is not None:
                block_args["position_embeddings"] = tuple(e.to(dev) for e in block_args["position_embeddings"])


        subset = find_layers(block)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(_name):
            def tmp(_, inp, out):
                gpts[_name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            block(inps[j].to(dev).unsqueeze(0), **block_args)

        for h in handles:
            h.remove()

        for gpt in gpts:
            print('Pruning ...')

            name = gpt
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, structured=structured)
            gpts[name].free()

            print(i, name)

        print("Recomputing the whole layers output...")
        for j in range(args.nsamples):
            outs[j] = block(inps[j].to(dev).unsqueeze(0), **block_args)[0]

        if len(model.hf_device_map) == 1:
            block = block.to(device=model.hf_device_map[''])

        blocks[i] = block
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_thanos(args, model, tokenizer, dev, prune_n=0, prune_m=0, blocksize=256, v_blocksize=256, structured=False, perc_outliers=0.1):
    print('Starting ...')

    dataloader, _ = get_loaders("c4",
                                nsamples=args.nsamples,
                                seed=args.seed,
                                seqlen=model.seqlen,
                                tokenizer=tokenizer)

    print("Loaded dataset...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    blocks = get_all_blocks(model)

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
    elif len(model.hf_device_map) == 1:
        dev = model.hf_device_map['']

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, min(2048, model.seqlen), model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_embeddings' in kwargs:
                cache['position_embeddings'] = kwargs['position_embeddings']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    blocks[0] = Catcher(blocks[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    blocks[0] = blocks[0].module
    #torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    block_args = {}
    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask  # Add attention mask if defined
    if position_ids is not None:
        block_args["position_ids"] = position_ids  # Add position IDs if defined
    if position_embeddings is not None:
        block_args["position_embeddings"] = position_embeddings  # Add position embeddings if defined

    print('Ready.')

    average_l2_loss = 0

    for i in range(len(blocks)):
        block = blocks[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")

            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        elif len(model.hf_device_map) == 1:
            dev = torch.device('cuda:0')
            block = block.to(device=dev)

            if attention_mask is not None:
                block_args["attention_mask"] = block_args["attention_mask"].to(dev)
            if position_ids is not None:
                block_args["position_ids"] = block_args["position_ids"].to(dev)
            if position_embeddings is not None:
                block_args["position_embeddings"] = tuple(e.to(dev) for e in block_args["position_embeddings"])

        subset = find_layers(block)

        gpts = {}
        for name in subset:
            gpts[name] = Thanos(subset[name], store_inputs=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            block(inps[j].to(dev).unsqueeze(0), **block_args)

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)

            print('Pruning ...')

            gpts[name].snap(args.sparsity_ratio,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            percdamp=0.01,
                            blocksize=blocksize,
                            v_blocksize=v_blocksize,
                            adaptive_blocksize=False,
                            structured=structured,
                            perc_outliers=perc_outliers)

            if gpts[name].l2_loss is not None:
                average_l2_loss += gpts[name].l2_loss / len(gpts)

            gpts[name].free()

        print("Recomputing the whole layers output...")
        for j in range(args.nsamples):
            outs[j] = block(inps[j].to(dev).unsqueeze(0), **block_args)[0]

        if len(model.hf_device_map) == 1:
            block = block.to(device=model.hf_device_map[''])

        blocks[i] = block
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(blocks)
        #print("Average L2 loss =", average_l2_loss)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


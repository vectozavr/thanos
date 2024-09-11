import time 
import heapq 
import torch 
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT
from .thanos import Thanos
from .layerwrapper import WrappedGPT
from .data import get_loaders

from .ablate import AblateGPT

is_compute_l2 = True

import numpy as np
from PIL import Image
import sys


def save_mask_as_rgb_image(mask, filename):
    """
    Save a square mask tensor as an RGB image.

    Parameters:
    mask (torch.Tensor): A boolean tensor of shape (n, n) where True is black and False is white.
    filename (str): The name of the file to save the image as.
    """
    # Convert the mask to a numpy array with values 0 (black) and 255 (white)
    grayscale_image = np.where(mask.cpu().numpy(), 0, 255).astype(np.uint8)

    # Convert the grayscale image to an RGB image by stacking the channels
    rgb_image = np.stack([grayscale_image] * 3, axis=-1)

    # Convert to a PIL image
    pil_image = Image.fromarray(rgb_image)

    # Save the image
    pil_image.save(filename)


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}

    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def l12_loss(dW, Xj):
    dW = dW.float()
    Xj = Xj.float()

    mult = dW @ Xj
    l12 = torch.sum(mult**2)

    return l12


def compute_l2_loss(dW, X):
    loss = 0
    for Xj in X:
        loss += l12_loss(dW, Xj) / len(X)
    return loss


def compute_l2_relative_loss(W, W_old, X):
    dW = W - W_old

    loss = 0

    for Xj in X:
        l12 = l12_loss(dW, Xj)
        l12_abs = l12_loss(W_old, Xj)

        loss += l12 / l12_abs

    loss /= len(X)

    return loss


def plot_heatmap(tensor, title):
    import matplotlib.pyplot as plt

    vmin, vmax = np.percentile(tensor.cpu().numpy(), [2, 98])

    m, n = tensor.shape
    plt.figure(figsize=((n + 500) / 100, (m + 500) / 100), dpi=100)
    plt.imshow(tensor.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, n, 0, m])
    plt.colorbar(label=r'Values')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    #plt.tight_layout(pad=1.0)

    plt.savefig(title + ".png", format='png', dpi=100)
    plt.close()


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.decoder.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

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


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask



def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.decoder.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_old = W.clone()

            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def rec_loss_dW(dW, X):
    dW_expanded = dW.unsqueeze(0)
    multiplications = torch.matmul(dW_expanded, X.float())
    losses = torch.norm(multiplications, p='fro', dim=(1, 2))
    loss = torch.mean(losses)
    return loss


def rec_loss(W, M, X):
    dW = (W - W * M).float()

    return rec_loss_dW(dW, X)


def gd_sparse_mask(W, X, M0=None, sparsity_ratio=0.5, lr=0.01, num_iters=100, sparsity_lambda=0.1, l1_lambda=0.1):
    """
    Generate a sparse mask M via minimizing the loss using Adam optimizer, with reconstruction loss,
    sparsity loss, L1 regularization, and final thresholding to enforce exact sparsity.

    Args:
    - W (torch.Tensor): The weight matrix of size (m, n).
    - X (torch.Tensor): The input matrix of size (n, l).
    - sparsity_ratio (float): The desired sparsity ratio (e.g., 0.5 for 50% sparsity).
    - lr (float): Learning rate for gradient descent.
    - num_iters (int): Number of iterations for gradient descent.
    - lambda1 (float): Regularization parameter for the sparsity normalization term.
    - lambda2 (float): Regularization parameter for L1 sparsity constraint.

    Returns:
    - M (torch.Tensor): The generated mask of size (m, n) after optimization and thresholding.
    """

    reconstruction_loss_hist = []
    sparsity_penalty_hist = []
    l1_penalty_hist = []
    loss_hist = []

    norm_1 = None
    norm_2 = None

    if M0 is None:
        # Initialize M with continuous values between 0 and 1
        M = torch.rand(W.size(), requires_grad=True)
    else:
        M = M0.clone().requires_grad_()

    # Define optimizer
    optimizer = torch.optim.Adam([M], lr=lr)

    for k in range(num_iters):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the reconstruction loss
        #reconstruction_loss = rec_loss(W, M, X)
        #reconstruction_loss = rec_loss_with_dW(W, M, X, dW)
        #reconstruction_loss = rec_loss_where_W_can_change(W0, W, M, X)

        # Compute the reconstruction loss
        reconstruction_loss = compute_l2_loss(W*M, X)

        # Compute the L1 regularization term
        l1_penalty = torch.mean(torch.abs(M))

        # Compute the sparsity normalization term
        sparsity_penalty = (l1_penalty + sparsity_ratio - 1) ** 2

        if norm_1 is None or norm_2 is None:
            norm_1 = reconstruction_loss.item()
            norm_2 = l1_penalty.item()

        # Total loss is the sum of reconstruction loss, sparsity penalty, and L1 penalty
        total_loss = reconstruction_loss / norm_1 + sparsity_lambda*sparsity_penalty + l1_lambda*l1_penalty / norm_2

        # Backpropagate the loss
        total_loss.backward()

        # Update M using the optimizer
        optimizer.step()

        reconstruction_loss_hist.append([k, reconstruction_loss.item() / norm_1])
        sparsity_penalty_hist.append([k, sparsity_penalty.item()])
        l1_penalty_hist.append([k, l1_lambda*l1_penalty.item() / norm_2])
        loss_hist.append([k, total_loss.item()])

        #print("k = ", k, " l = ", total_loss.item())

        # Enforce M to stay in the range [0, 1]
        with torch.no_grad():
           M.clamp_(0.0, 1.0)

    # print(M)

    # After optimization, apply thresholding to achieve the exact sparsity ratio
    with torch.no_grad():
        # Flatten the mask and sort the values
        #flat_M = torch.abs(M).flatten()

        flat_M = M.flatten()
        num_elements_to_keep = int(flat_M.numel() * (1 - sparsity_ratio))

        # Find the threshold value for the top (1 - sparsity_ratio) proportion
        threshold_value = torch.topk(flat_M, num_elements_to_keep, largest=True).values[-1]

        # Set elements greater than or equal to the threshold to 1, and the rest to 0
        M = (M >= threshold_value)


        # Here we try to make the same as wanda: row-wise sparsity:
        '''
        wanda_like_mask = torch.zeros_like(M, dtype=torch.bool)
        sort_res = torch.sort(M, dim=-1, stable=True, descending=True)
        indices = sort_res[1][:, :int(M.shape[1] * sparsity_ratio)]
        wanda_like_mask.scatter_(1, indices, True)
        M = wanda_like_mask
        '''

        #mask = (M < threshold_value)
        #W_new = W * M
        #W_new[mask] = 0

    #print("Reconstruction loss: ", reconstruction_loss_hist[-1][1])

    #return M, W_new, reconstruction_loss_hist, sparsity_penalty_hist, l1_penalty_hist, loss_hist
    return M


def prune_opt_mask(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, is_store_all_loses = False):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    average_l2_loss = 0

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        l2_losses = []

        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], store_inputs=is_compute_l2)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:

            W = wrapped_layers[name].layer.weight.data.clone()
            if isinstance(wrapped_layers[name].layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(wrapped_layers[name].layer, transformers.Conv1D):
                W = W.t()
            W = W.float()

            if hasattr(wrapped_layers[name], 'X'):
                W_old = W.clone()

            X = wrapped_layers[name].X
            scaler_row = wrapped_layers[name].scaler_row

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape((1,-1)))
            #scaler_row_sum = torch.norm(wrapped_layers[name].X_sum, p=2, dim=1)**2
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row_sum.reshape((1, -1)))
            #W_metric = torch.log(torch.abs(subset[name].weight.data)) + wrapped_layers[name].scaler_row.reshape((1, -1))

            #medians_scaler_rows = torch.mean(wrapped_layers[name].all_scaler_rows, dim=0)

            #quantile_80 = torch.max(wrapped_layers[name].all_scaler_rows, dim=0).values
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(quantile_80.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # unstructured pruning
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)


            #save_mask_as_rgb_image(W_mask, "data/mask_image_"+name+".png")
            M0 = W_mask.to(dtype=torch.float)
            #gd_mask = gd_sparse_mask(W=W,
            #                         X=X,
            #                         M0=M0,
            #                         sparsity_ratio=args.sparsity_ratio,
            #                         lr=1e-1,
            #                         num_iters=15,
            #                         sparsity_lambda=50,
            #                         l1_lambda=0.01)

            #W_new = W * M0
            W_new_wanda = W.clone()
            W_new_wanda[W_mask] = 0

            W_new_gd = W.clone()
            W_new_gd[W_mask] = 0

            #subset[name].weight.data[W_mask] = 0
            #subset[name].weight.data[gd_mask] = 0
            #subset[name].weight.data = W_new.half()

            if hasattr(wrapped_layers[name], 'X'):
                if is_store_all_loses:
                    l2_losses.append(compute_l2_loss(subset[name].weight.data - W_old, X).item())
                else:
                    gd_loss = compute_l2_loss(W_new_gd - W_old, X).item()

                    average_l2_loss += gd_loss / len(wrapped_layers)

                    print("L2 GD op =", gd_loss)

                    wanda_loss = compute_l2_loss(W_new_wanda - W_old, X).item()
                    print("L2 Wanda =", wanda_loss)

                    if wanda_loss < gd_loss:
                        print("*** WANDA IS BETTER CASE! ***")
                        subset[name].weight.data = W_new_wanda.half()
                    else:
                        subset[name].weight.data = W_new_gd.half()

        #sys.exit()

        if is_store_all_loses:
            model.config.use_cache = use_cache
            torch.cuda.empty_cache()

            return l2_losses

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(layers)
        print("Average L2 loss =", average_l2_loss)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return average_l2_loss


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

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
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0, is_store_all_loses = False):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    average_l2_loss = 0

    for i in range(len(layers)):
        l2_losses = []

        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name], store_inputs=is_compute_l2)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if gpts[name].l2_loss is not None:
                if is_store_all_loses:
                    l2_losses.append(gpts[name].l2_loss.item())
                else:
                    average_l2_loss += gpts[name].l2_loss.item() / len(gpts)

            gpts[name].free()

        if is_store_all_loses:
            model.config.use_cache = use_cache
            torch.cuda.empty_cache()
            return l2_losses

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(layers)
        print("Average L2 loss =", average_l2_loss)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return average_l2_loss

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask,
                                        prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prepare_bathed_dataset(dataloader):
    inps = []
    for i in range(len(dataloader)//32):
        _from = i*32
        _to = min(len(dataloader), (i + 1) * 32)
        batch_samples = [sample[0].flatten() for sample in dataloader[_from:_to]]
        batch = torch.stack(batch_samples)
        inps.append(batch)

    return inps


#@torch.no_grad()
def prune_thanos(args, model, tokenizer, dev, prune_n=0, prune_m=0, is_store_all_loses=False):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    #inps = torch.zeros(
    #    (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    #)

    inps = [None for _ in range(args.nsamples)]

    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.reshape((-1, inp.shape[-1])).to(torch.device("cpu"))
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    #outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    average_l2_loss = 0

    def recalculate(layer, inps, gpts, subset):

        def _add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        _handles = []
        for name in gpts:
            gpts[name].free_batch()
            _handles.append(subset[name].register_forward_hook(_add_batch(name)))

        with torch.no_grad():
            for j in range(args.nsamples):
                layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask)[0]

        for h in _handles:
            h.remove()

    l2_mean_losses = {}

    for i in range(len(layers)):
        l2_losses = []

        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            #inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = Thanos(subset[name], store_inputs=is_compute_l2)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask)

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if name == "self_attn.k_proj" or name == "self_attn.q_proj":
                current_sparsity = 0.3
            else:
                current_sparsity = 0.5

            #if "fc2" in name and i==3:
            #    a = 0

            #glob_tmp = torch.abs(gpts[name].W) * torch.sqrt(gpts[name].scaler_row).reshape((1, -1))
            #plot_heatmap(glob_tmp, name)

            gpts[name].snap(args.sparsity_ratio,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            percdamp=0.01,
                            blocksize=128,
                            v_blocksize=256,
                            adaptive_blocksize=False)

            #recalculate(layer, inps, gpts, subset)

            #gpts[name].slowprune(args.sparsity_ratio,
            #                     percdamp=0.01,
            #                     blocksize=128)

            #gpts[name].slowprune_structured_optimal(prune_n=prune_n,
            #                                        prune_m=prune_m,
            #                                        percdamp=0.01)

            if gpts[name].l2_loss is not None:
                if is_store_all_loses:
                    l2_losses.append(gpts[name].l2_loss.item())
                else:
                    current_l2 = gpts[name].l2_loss.item()
                    average_l2_loss += current_l2 / len(gpts)


                    if name not in l2_mean_losses:
                        l2_mean_losses[name] = 0.0
                    l2_mean_losses[name] += current_l2/len(layers)

            gpts[name].free()

        #exit()

        if is_store_all_loses:
            model.config.use_cache = use_cache
            torch.cuda.empty_cache()
            return l2_losses

        for j in range(args.nsamples):
            with torch.no_grad():
                out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                inps[j] = out.reshape((-1, out.shape[-1])).to(torch.device("cpu"))

        layers[i] = layer
        torch.cuda.empty_cache()

        #inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(layers)
        print("Average L2 loss =", average_l2_loss)

    print("Average L2 for layers:\n", l2_mean_losses)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return average_l2_loss

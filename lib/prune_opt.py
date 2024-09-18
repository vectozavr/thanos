import time 
import heapq 
import torch 
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT
from .thanos_multicase import ThanosMultiCase
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


def prepare_average_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((1, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.nsamples = 1
        def forward(self, inp, **kwargs):
            inps[0] *= self.nsamples/(self.nsamples+1)
            inps[0] += inp[0] / (self.nsamples+1)
            if self.nsamples == 1:
                cache['attention_mask'] = kwargs['attention_mask']
            self.nsamples += 1
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


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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


def compute_layer_loss(layer, dense_layer, input, attention_mask):
    average_layer_loss = 0

    nsamples = input.shape[0]
    for j in range(nsamples):
        dense_out = dense_layer(input[j].unsqueeze(0), attention_mask=attention_mask)[0]
        current_out = layer(input[j].unsqueeze(0), attention_mask=attention_mask)[0]

        average_layer_loss += torch.sum((dense_out - current_out)**2)/nsamples

    return average_layer_loss


def compute_dense_layer_l2(layer, input, attention_mask):
    average_layer_loss = 0

    nsamples = input.shape[0]
    for j in range(nsamples):
        current_out = layer(input[j].unsqueeze(0), attention_mask=attention_mask)[0]
        average_layer_loss += torch.mean(current_out**2) / nsamples

    return average_layer_loss


def gd_sparse_mask(W, X, inps, attention_mask, layer, submodule, M0=None, sparsity_ratio=0.5, lr=0.01, num_iters=100, sparsity_lambda=0.1, l1_lambda=0.1):
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
    inps = inps.to(dtype=torch.float32)
    layer.to(dtype=torch.float32)

    for param in layer.parameters():
        param.requires_grad = False

    import torch.nn.utils.parametrize as P
    class MaskParametrization(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.mask = mask

        def forward(self, W):
            return W * self.mask


    import copy
    layer_copy = copy.deepcopy(layer)

    #reconstruction_loss_hist = []
    #sparsity_penalty_hist = []
    #l1_penalty_hist = []
    #loss_hist = []

    norm_1 = None
    norm_2 = None
    norm_3 = None

    # small pertubation to zero elements
    #M0[M0 == 0.0] = 1e-3

    with torch.no_grad():
        M = M0.clone().requires_grad_()
        #M = torch.full_like(M0, 0.5, requires_grad=True)

        P.register_parametrization(submodule, 'weight', MaskParametrization(M))
        wanda_loss = compute_layer_loss(layer, layer_copy, inps, attention_mask)
        P.remove_parametrizations(submodule, 'weight', leave_parametrized=False)


        #M = torch.full_like(M0, 0.5, requires_grad=True)
        M[M0 == 0.0] = 0.8
        M[M0 == 1.0] = 0.99

        P.register_parametrization(submodule, 'weight', MaskParametrization(M))

        #M[M == 0.0] = 0.5

    # Define optimizer
    optimizer = torch.optim.Adam([M], lr=lr)

    for k in range(num_iters):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the reconstruction loss
        #reconstruction_loss = compute_l2_loss(W*M, X)
        reconstruction_loss = compute_layer_loss(layer, layer_copy, inps, attention_mask)

        # Compute the L1 regularization term
        l1_penalty = torch.mean(torch.abs(M))

        # Compute the sparsity normalization term
        sparsity_penalty = (l1_penalty + sparsity_ratio - 1) ** 2

        with torch.no_grad():
            if norm_1 is None or norm_2 is None:
                norm_1 = reconstruction_loss.item()
                norm_2 = l1_penalty.item()
                norm_3 = sparsity_ratio**2 # this one is the max value of sparsity penalty
                print("norm_1 =", norm_1)
                print("norm_2 =", norm_2)
                print("norm_3 =", norm_3)

        # Total loss is the sum of reconstruction loss, sparsity penalty, and L1 penalty
        total_loss = reconstruction_loss / norm_1 +\
                     l1_lambda * l1_penalty / norm_2 +\
                     sparsity_lambda * sparsity_penalty / norm_3

        # Backpropagate the loss
        total_loss.backward()

        # Update M using the optimizer
        optimizer.step()

        with torch.no_grad():
            #reconstruction_loss_hist.append([k, reconstruction_loss.item() / norm_1])
            #sparsity_penalty_hist.append([k, sparsity_penalty.item()])
            #l1_penalty_hist.append([k, l1_lambda*l1_penalty.item() / norm_2])
            #loss_hist.append([k, total_loss.item()])

            print("k = ", k, " l = ", total_loss.item())

            #M.clamp_(0.0, 1.0)

    # print(M)

    # After optimization, apply thresholding to achieve the exact sparsity ratio
    with torch.no_grad():
        # Flatten the mask and sort the values
        flat_M = M.view(-1)
        abs_flat_M = flat_M.abs()

        num_elements_to_keep = int(flat_M.numel() * (1 - sparsity_ratio))

        sorted_abs_vals, indices = torch.sort(abs_flat_M)
        indices_to_zero = indices[:num_elements_to_keep]

        # Set the smallest elements to zero
        flat_M[indices_to_zero] = 0.0
        #flat_M[flat_M != 0] = 1.0


        final_loss = compute_layer_loss(layer, layer_copy, inps, attention_mask)
        P.remove_parametrizations(submodule, 'weight', leave_parametrized=False)

        print("Logits Wanda loss =", wanda_loss.item())
        print("Logits GD loss =", final_loss.item())

        inps = inps.to(dtype=torch.float16)
        layer.to(dtype=torch.float16)

        #M = ~(M.to(dtype=torch.bool))

    return M, wanda_loss.item(), final_loss.item()


def compute_logits_loss(model_dense, model_final, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"
    #dataset = "c4"

    seqlen = model_dense.seqlen

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer
    )

    # Get input IDs
    testenc = testloader.input_ids
    bs=1

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    # List to store negative log likelihoods
    average_loss = 0

    # Loop through each batch
    for i in range(0,nsamples,bs):
        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)

        # Forward pass through the model
        logits_dense = model_dense(inputs).logits
        logits_final = model_final(inputs).logits

        average_loss += torch.mean((logits_dense-logits_final)**2)/nsamples


    return average_loss.item()


from transformers import AutoTokenizer, AutoModelForCausalLM
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


def prune_opt_mask(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, is_store_all_loses=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device, args.nsamples)

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
            M0 = (~W_mask).to(dtype=torch.float)

            gd_mask, wanda_loss_logits, gd_loss_logits = gd_sparse_mask(W=W,
                                     X=X,
                                     inps=inps,
                                     attention_mask=attention_mask,
                                     layer=layer,
                                     submodule=subset[name],
                                     M0=M0,
                                     sparsity_ratio=args.sparsity_ratio,
                                     lr=1e-2,
                                     num_iters=50,
                                     sparsity_lambda=15,
                                     l1_lambda=1)

            #W_new = W * M0
            W_new_wanda = W.clone()
            W_new_wanda[W_mask] = 0

            W_new_gd = W.clone()*gd_mask
            #W_new_gd = W.clone()
            #W_new_gd[gd_mask] = 0

            #subset[name].weight.data[W_mask] = 0
            #subset[name].weight.data[gd_mask] = 0
            #subset[name].weight.data = W_new.half()

            if hasattr(wrapped_layers[name], 'X'):
                if is_store_all_loses:
                    l2_losses.append(compute_l2_loss(subset[name].weight.data - W_old, X).item())
                else:

                    gd_loss = compute_l2_loss(W_new_gd - W_old, X).item()

                    average_l2_loss += gd_loss / len(wrapped_layers)
                    wanda_loss = compute_l2_loss(W_new_wanda - W_old, X).item()

                    print("L2 Wanda =", wanda_loss)
                    print("L2 GD op =", gd_loss)

                    if wanda_loss_logits < gd_loss_logits:
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


    with torch.no_grad():
        model_dense = get_llm(args.model, args.cache_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        final_logits_loss = compute_logits_loss(model_dense, model, tokenizer, device)
        print("Logits l2 loss =", final_logits_loss)


    return average_l2_loss


def layer_loss(target, value):
    diff = (target - value)
    return torch.mean(diff**2)


def mask_half_largest_elements_across_tensors(tensor_list):
    """
    Sets half of the smallest elements (in absolute value) across all tensors in the list to zero.
    The tensors in the list are modified in-place.

    Parameters:
        tensor_list (list of torch.Tensor): A list of PyTorch tensors.

    Returns:
        None
    """
    # Flatten each tensor and keep track of sizes and shapes
    flattened_tensors = []
    sizes = []
    shapes = []

    for tensor in tensor_list:
        shapes.append(tensor.shape)
        flat_tensor = tensor.view(-1)
        flattened_tensors.append(flat_tensor)
        sizes.append(flat_tensor.numel())

    # Concatenate all flattened tensors
    all_params = torch.cat(flattened_tensors)
    total_num_elements = all_params.numel()

    # Compute absolute values
    abs_all_params = all_params.abs()

    # Get sorted indices
    sorted_abs_vals, sorted_indices = torch.sort(abs_all_params)

    # Determine number of elements to zero out
    num_zero = total_num_elements // 2  # Integer division

    # Get indices of smallest elements
    indices_to_zero = sorted_indices[:num_zero]

    # Create a mask
    mask = torch.ones_like(all_params, dtype=torch.bool)
    mask[indices_to_zero] = False  # Elements to zero out

    # Split the mask back into the original tensor sizes
    masks = torch.split(mask, sizes)

    # Apply the mask to each tensor
    for i, tensor in enumerate(tensor_list):
        tensor_mask = masks[i].view(shapes[i])
        tensor[~tensor_mask] = 0
        tensor[tensor_mask] = 1


def prune_opt_layer(layer_number, inps, layer, model, sparsity_ratio, attention_mask):
    layer.to(dtype=torch.float32)

    import copy
    layer_copy = copy.deepcopy(layer)

    subset = find_layers(layer)

    for param in layer.parameters():
        param.requires_grad = False

    nsamples = inps.shape[0]

    inps = inps.float()
    target = torch.zeros_like(inps)

    if f"model.layers.{layer_number}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        dev = model.hf_device_map[f"model.layers.{layer_number}"]
        inps = inps.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

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

    with torch.no_grad():
        for j in range(nsamples):
            target[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

    for h in handles:
        h.remove()


    M0 = {}
    M0_bool = {}
    W_dense = {}

    print(f"Find initial point for masks (Wanda masks)")
    # Find initial point for masks (Wanda masks)
    for name in subset:
        W = wrapped_layers[name].layer.weight.data.clone()
        if isinstance(wrapped_layers[name].layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(wrapped_layers[name].layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        W_dense[name] = W

        scaler_row = wrapped_layers[name].scaler_row

        W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape((1, -1)))

        W_mask = (torch.zeros_like(W_metric) == 1)

        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)

        # we solve the relaxed problem, so we have to transform it to float array
        M0[name] = (~W_mask).float()

    # Solve the optimization problem for the whole layer

    import torch.nn.utils.parametrize as P

    class MaskParametrization(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.mask = mask

        def forward(self, W):
            return W * self.mask

    #lr = 1e-1
    #num_iters = 150
    #sparsity_lambda = 50
    #l1_lambda = 0.01


    #M = {}
    '''
    for name in subset:
        #M[name] = M0[name].clone().requires_grad_()
        M[name] = torch.ones_like(M0[name], requires_grad=True)

        sublayer = subset[name]
        P.register_parametrization(sublayer, 'weight', MaskParametrization(M[name]))
    '''
    '''
    m = torch.ones_like(M0['self_attn.k_proj'], requires_grad=True)
    sublayer = subset['self_attn.k_proj']
    P.register_parametrization(sublayer, 'weight', MaskParametrization(m))

    optimizer = torch.optim.Adam([m], lr=1e-2)
    optimizer.zero_grad()

    loss = compute_dense_layer_l2(layer, inps, attention_mask)
    loss.backward()
    '''
    '''
    with torch.no_grad():

        flat_grad = m.grad.clone().view(-1)
        num_elements_to_keep = int(flat_grad.numel() * (1 - sparsity_ratio))
        sorted_abs_vals, indices = torch.sort(flat_grad.abs(), descending=False)
        indices_to_zero = indices[:num_elements_to_keep]
        flat_grad[indices_to_zero] = 0.0
        flat_grad[flat_grad != 0.0] = 1.0

        # flat_grad = flat_grad.to(dtype=torch.bool)

        grad_mask = flat_grad.view(m.shape)

        m.copy_(flat_grad.view(m.shape))
    '''
    '''
        grad_list = []
        m_list = []
        for m_name in M:
            grad_list.append(M[m_name].grad)
            m_list.append(M[m_name])

        mask_half_largest_elements_across_tensors(grad_list)

        for i in range(len(m_list)):
            m_list[i].copy_(grad_list[i])
    '''

    M0 = M0['self_attn.k_proj']
    submodule = subset['self_attn.k_proj']

    with torch.no_grad():

        M = M0.clone()
        #M = torch.full_like(M0, 0.5, requires_grad=True)

        P.register_parametrization(submodule, 'weight', MaskParametrization(M))
        wanda_loss = compute_layer_loss(layer, layer_copy, inps, attention_mask)
        P.remove_parametrizations(submodule, 'weight', leave_parametrized=False)


        M = torch.ones_like(M0, requires_grad=True)
        P.register_parametrization(submodule, 'weight', MaskParametrization(M))

        #M[M == 0.0] = 0.5

    # Define optimizer
    reconstruction_loss = compute_dense_layer_l2(layer, inps, attention_mask)
    reconstruction_loss.backward()

    flat_grad = M.grad.clone().view(-1)
    num_elements_to_keep = int(flat_grad.numel() * (1 - sparsity_ratio))
    sorted_abs_vals, indices = torch.sort(flat_grad.abs(), descending=False)
    indices_to_zero = indices[:num_elements_to_keep]
    flat_grad[indices_to_zero] = 0.0
    flat_grad[flat_grad != 0.0] = 1.0
    flat_grad = flat_grad.to(dtype=torch.bool)
    grad_mask = flat_grad.view(M0.shape)

    P.remove_parametrizations(submodule, 'weight', leave_parametrized=True)


    #norm_1 = None
    #norm_2 = None

    '''
    print(f"pruning layer {layer_number}")

    # Define optimizer
    tensor_list = list(M.values())
    optimizer = torch.optim.Adam(tensor_list, lr=lr)

    wanda_reconstruction_loss = None

    for k in range(num_iters):
        # Zero the gradients
        optimizer.zero_grad()

        values = []
        for j in range(nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            values.append(out)

        # Concatenate the outputs along the appropriate dimension
        value = torch.cat(values, dim=0)

        #value = layer(inps, attention_mask=attention_mask)[0]

        # Compute the reconstruction loss
        reconstruction_loss = layer_loss(target.to(value.device), value)

        # On the first iteration we have the same loss as Wanda loss (that is our starting point)
        if wanda_reconstruction_loss is None:
            wanda_reconstruction_loss = reconstruction_loss

        # Compute the L1 regularization term
        l1_penalty = []
        for name in subset:
            l1_penalty.append(torch.mean(M[name]))
        l1_penalty = torch.stack(l1_penalty)

        # Compute the sparsity normalization term
        sparsity_penalty = torch.mean((l1_penalty + sparsity_ratio - 1) ** 2)

        if norm_1 is None or norm_2 is None:
            norm_1 = reconstruction_loss.item()
            norm_2 = torch.mean(l1_penalty).item()

        # Total loss is the sum of reconstruction loss, sparsity penalty, and L1 penalty
        total_loss = reconstruction_loss.to(torch.device("cpu")) / norm_1 + sparsity_lambda * sparsity_penalty.to(torch.device("cpu")) + l1_lambda * torch.mean(l1_penalty).to(torch.device("cpu")) / norm_2
        #total_loss = reconstruction_loss / norm_1

        # Backpropagate the loss
        total_loss.backward()

        # Update M using the optimizer
        optimizer.step()

        print("k = ", k, " l = ", total_loss.item())

        # Enforce M to stay in the range [0, 1]
        with torch.no_grad():
            for name in subset:
                M[name].clamp_(0.0, 1.0)

    # After optimization, apply thresholding to achieve the exact sparsity ratio
    with torch.no_grad():
        flat_M_all = []
        num_elements_to_keep_all = 0
        for name in subset:
            flat_M = M[name].flatten()
            num_elements_to_keep = int(flat_M.numel() * (1 - sparsity_ratio))
            num_elements_to_keep_all += num_elements_to_keep

            # Find the threshold value for the top (1 - sparsity_ratio) proportion
            #threshold_value = torch.topk(flat_M, num_elements_to_keep, largest=True).values[-1]

            # Set elements greater than or equal to the threshold to 1, and the rest to 0
            #M[name] = M[name] >= threshold_value

            flat_M_all.append(flat_M)

        flat_M_all = torch.cat(flat_M_all, dim=0)
        threshold_value_all = torch.topk(flat_M_all, num_elements_to_keep_all, largest=True).values[-1]

        for name in subset:
            M[name] = M[name] >= threshold_value_all
    '''

    with torch.no_grad():
        for name in subset:
            sublayer = wrapped_layers[name].layer
            P.remove_parametrizations(sublayer, 'weight', leave_parametrized=True)

            mask = M[name]
            wrapped_layers[name].layer.weight.data = W_dense[name]*mask

        # Compute the output for the current layer (input for the next layer)
        outs = torch.zeros_like(inps)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        gd_reconstruction_loss = layer_loss(target, outs)

        #print("Wanda loss =", wanda_reconstruction_loss.item())
        print("GD loss =", gd_reconstruction_loss.item())

        layer.to(dtype=torch.float16)

    return outs


def prune_opt_full_layer(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device, args.nsamples)
        #attention_mask = attention_mask.expand(args.nsamples, -1, -1, -1)

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]

        # inps, layer, model, sparsity_ratio, attention_mask
        outs = prune_opt_layer(i, inps, layer, model, args.sparsity_ratio, attention_mask)

        inps = outs

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device, args.nsamples)
        #inps, outs, attention_mask = prepare_average_calibration_input(model, dataloader, device)


    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
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
        #with torch.no_grad():
        #    outs[0] = layer(inps, attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")

            #average_X = torch.zeros_like(wrapped_layers[name].X[0])
            #for x in average_X:
            #    average_X += x / len(average_X)
            #average_scaler_row = torch.norm(average_X, p=2, dim=1) ** 2

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(average_scaler_row.reshape((1, -1)))

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
        #with torch.no_grad():
        #    outs[0] = layer(inps, attention_mask=attention_mask)[0]

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

        tick = time.time()
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

        print('Layer pruning time %.2f' % (time.time() - tick))

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


@torch.no_grad()
def prune_thanos(args, model, tokenizer, dev, prune_n=0, prune_m=0, is_store_all_loses=False):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    #inps = [None for _ in range(args.nsamples)]

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

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    average_l2_loss = 0

    l2_mean_losses = {}

    for i in range(len(layers)):
        l2_losses = []

        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = Thanos(subset[name], store_inputs=is_compute_l2)
            #gpts[name] = ThanosMultiCase(subset[name], store_inputs=is_compute_l2)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Compute the input for each sublayer
        for j in range(args.nsamples):
            layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask)

        #attention_mask = attention_mask.expand(args.nsamples, -1, -1, -1)
        #layer(inps, attention_mask=attention_mask)

        for h in handles:
            h.remove()

        tick = time.time()
        for name in gpts:
            print(i, name)
            print('Pruning ...')

            #glob_tmp = torch.abs(gpts[name].W) * torch.sqrt(gpts[name].scaler_row).reshape((1, -1))
            #plot_heatmap(glob_tmp, name)

            #gpts[name].snap(args.sparsity_ratio,
            #                prune_n=prune_n,
            #                prune_m=prune_m,
            #                percdamp=0.01,
            #                blocksize=128,
            #                v_blocksize=256,
            #                adaptive_blocksize=False)
            gpts[name].snap(args.sparsity_ratio,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            percdamp=0.01,
                            blocksize=128,
                            adaptive_blocksize=False)

            #gpts[name].slowprune(args.sparsity_ratio,
            #                     percdamp=0.01,
            #                     blocksize=128)

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

        if is_store_all_loses:
            model.config.use_cache = use_cache
            torch.cuda.empty_cache()
            return l2_losses

        print('Layer pruning time %.2f' % (time.time() - tick))

        # Re-Compute the input for the next layer (because the layer was pruned, so the output will not be the same)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        #outs = layer(inps, attention_mask=attention_mask)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(layers)
        print("Average L2 loss =", average_l2_loss)

    print("Average L2 for layers:\n", l2_mean_losses)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return average_l2_loss

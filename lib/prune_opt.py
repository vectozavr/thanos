import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT
from .thanos import Thanos
from .layerwrapper import WrappedGPT
from .data import get_loaders

from .ablate import AblateGPT

is_compute_l2 = True

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


def l12_loss(dW, X):
    dW = dW.float()
    X = X.float()

    mult = dW @ X
    l12 = torch.sum(torch.linalg.norm(mult, dim=1)**2)

    return l12


def compute_l2_loss(dW, X):
    loss = 0

    for Xj in X:
        loss += l12_loss(dW, Xj)
    loss /= len(X)

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



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, is_store_all_loses = False):
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
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            scaler_row_sum = torch.norm(wrapped_layers[name].X_sum, p=2, dim=1)**2
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row_sum.reshape((1, -1)))

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

            if hasattr(wrapped_layers[name], 'X'):
                W_old = subset[name].weight.data.clone()

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

            if hasattr(wrapped_layers[name], 'X'):
                if is_store_all_loses:
                    l2_losses.append(compute_l2_loss(subset[name].weight.data - W_old, wrapped_layers[name].X).item())
                else:
                    current_loss = compute_l2_loss(subset[name].weight.data - W_old, wrapped_layers[name].X).item()
                    #current_loss = compute_l2_relative_loss(subset[name].weight.data, W_old, wrapped_layers[name].X).item()
                    average_l2_loss += current_loss / len(wrapped_layers)

                    print("L2 =", current_loss)

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


@torch.no_grad()
def prune_thanos(args, model, tokenizer, dev, prune_n=0, prune_m=0, is_store_all_loses = False):
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

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

        for j in range(args.nsamples):
            layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask)[0]

        for h in _handles:
            h.remove()

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

            gpts[name].snap(current_sparsity,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            percdamp=0.01,
                            blocksize=256,
                            v_blocksize=256,
                            adaptive_blocksize=False)

            #recalculate(layer, inps, gpts, subset)

            #gpts[name].slowprune(args.sparsity_ratio,
            #                     percdamp=0.01,
            #                     blocksize=64)

            #gpts[name].slowprune_structured_optimal(prune_n=prune_n,
            #                                        prune_m=prune_m,
            #                                        percdamp=0.01)

            if gpts[name].l2_loss is not None:
                current_l2 = gpts[name].l2_loss.item()
                average_l2_loss += current_l2 / len(gpts)

            if gpts[name].l2_loss is not None:
                if is_store_all_loses:
                    l2_losses.append(gpts[name].l2_loss.item())
                else:
                    current_l2 = gpts[name].l2_loss.item()
                    average_l2_loss += current_l2 / len(gpts)

            gpts[name].free()

        if is_store_all_loses:
            model.config.use_cache = use_cache
            torch.cuda.empty_cache()
            return l2_losses

        for j in range(args.nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            inps[j] = out.reshape((-1, out.shape[-1])).to(torch.device("cpu"))

        layers[i] = layer
        torch.cuda.empty_cache()

        #inps, outs = outs, inps

    if average_l2_loss != 0:
        average_l2_loss /= len(layers)
        print("Average L2 loss =", average_l2_loss)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return average_l2_loss

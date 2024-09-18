import math
import time

import torch
import torch.nn as nn
import transformers
import itertools

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def l2_loss(dW, Xj):
    dW = dW.float()
    Xj = Xj.float()

    mult = dW @ Xj

    return torch.sum(mult ** 2)


def compute_l2_loss(dW, X):
    loss = 0
    for Xj in X:
        loss += l2_loss(dW, Xj) / len(X)
    return loss


class Thanos:
    def __init__(self, layer, store_inputs=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.scaler_row = torch.zeros(self.columns, device=self.dev)

        self.l2_loss = None
        if store_inputs:
            self.X = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        #inp = inp.transpose(1, 2)

        if hasattr(self, 'X'):
            self.X.append(inp)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp

        # Simple sum
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        #self.scaler_row += torch.mean(torch.norm(inp, p=2, dim=2) ** 2, dim=0)

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        #XXT_bathed = torch.bmm(inp, inp.transpose(1, 2))
        #self.H += XXT_bathed.sum(dim=0)
        self.H += inp.matmul(inp.t())



    def snap(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, adaptive_blocksize=False):
        # if blocksize == 1:
        #    # return self.snap_bs_one_magnitude_based(sparsity, prune_n, prune_m, percdamp)
        #    return self.snap_bs_one(sparsity, prune_n, prune_m, percdamp)

        W = self.layer.weight.data.clone()

        if hasattr(self, 'X'):
            W_old = W.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        #tick = time.time()

        if adaptive_blocksize:
            blocksize = self.columns // 16

        # blocksize = self.columns

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # This is for dynamic Thanos mask
        zeros = 0

        # TODO: move this flag from here to args
        use_constant_mask = False

        const_mask = None
        if use_constant_mask:
            # Global mask on Wanda metric
            glob_tmp = torch.abs(W) * torch.sqrt(self.scaler_row).reshape((1, -1))
            values, indices = torch.topk(glob_tmp.flatten(), int(W.numel() * sparsity), largest=False)
            const_mask = torch.zeros_like(W, dtype=torch.bool, device=self.dev)
            const_mask.view(-1)[indices] = True

            # Global Wanda mask
            # glob_tmp = torch.abs(W) * torch.sqrt(self.scaler_row).reshape((1, -1))
            # sort_res = torch.sort(glob_tmp, dim=-1, stable=True)
            # indices = sort_res[1][:, :int(glob_tmp.shape[1] * sparsity)]
            # const_mask = torch.zeros_like(W, dtype=torch.bool, device=self.dev)
            # const_mask.scatter_(1, indices, True)

        # Subsequent pruning
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            block_mask = None
            if use_constant_mask:
                block_mask = const_mask[:, i1:i2]
            else:

                local_block_sparsity = False

                if not local_block_sparsity:


                    #glob_tmp = torch.abs(W[:, i1:]) * torch.sqrt(self.scaler_row[i1:]).reshape((1, -1))
                    glob_tmp = W[:, i1:] ** 2 / (torch.diag(Hinv[i1:, i1:]).reshape((1, -1))) ** 2

                    estimate_zeros = int(sparsity * self.rows * self.columns - zeros)
                    values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)
                    glob_mask = torch.zeros_like(W[:, i1:], dtype=torch.bool, device=self.dev)
                    glob_mask.view(-1)[indices] = True
                    block_mask = glob_mask[:, :blocksize]
                    zeros += torch.sum(block_mask)


                else:
                    loc_tmp = torch.abs(W[:, i1:i2]) * torch.sqrt(self.scaler_row[i1:i2]).reshape((1, -1))
                    values, indices = torch.topk(loc_tmp.flatten(), int(loc_tmp.numel()*sparsity), largest=False)
                    loc_mask = torch.zeros_like(W[:, i1:i2], dtype=torch.bool, device=self.dev)
                    loc_mask.view(-1)[indices] = True
                    block_mask = loc_mask

            if prune_n == 0:  # unstructured sparsity
                #W_block = W[:, i1:i2]
                #Err = block_mask * W_block / torch.diag(Hinv[i1:i2, i1:i2])
                #W[:, i1:i2][block_mask] = 0
                #W[:, i2:] -= Err.matmul(Hinv[i1:i2, i2:])


                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(blocksize):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = w.clone()
                    q[block_mask[:, i]] = 0

                    Q1[:, i] = q

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


            else:  # structured n:m sparsity
                # TODO: implement structured n:m sparsity
                pass

        #print('Layer pruning time %.2f' % (time.time() - tick))

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            self.l2_loss = compute_l2_loss(W - W_old, self.X)
            print("Summ(|dW X_j|^2_1,2) =", self.l2_loss)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

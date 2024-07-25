import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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

        self.scaler_row = torch.zeros((self.columns), device=self.dev)

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

        if hasattr(self, 'X'):
            self.X.append(inp)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def __unstructured(self, W, Hinv, i1, i2, zeros, sparsity, v_blocksize):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1

        rows = W.shape[0]
        columns = W.shape[1]

        # Wanda metric
        #tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

        # This method of mask construction is more robust (comparison to sparseGPT and Wanda)
        # because it will produce the same number of non-zero elements
        # TODO: this is old solution, now we try with global mask
        #values, indices = torch.topk(tmp.flatten(), int(tmp.numel() * sparsity), largest=False)
        #mask = torch.zeros_like(tmp, dtype=torch.bool, device=self.dev)
        #mask.view(-1)[indices] = True

        # Global Wanda metric
        glob_tmp = torch.abs(W[:, i1:]) * torch.sqrt(self.scaler_row[i1:].reshape((1, -1)))

        estimate_zeros = int(sparsity*rows*columns - zeros)

        values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)
        glob_mask = torch.zeros_like(W[:, i1:], dtype=torch.bool, device=self.dev)
        glob_mask.view(-1)[indices] = True

        mask = glob_mask[:, :blocksize]

        # Here we compute how many elements we remove at once for each row (this is to make appropriate paddings)
        num_lambdas_for_each_row = mask.sum(dim=1)
        max_non_zeros = (num_lambdas_for_each_row.max()).item()

        new_zeros = torch.sum(num_lambdas_for_each_row)

        non_zero_indices = torch.nonzero(mask, as_tuple=False)
        cols_indices = non_zero_indices[:, 1]

        # Here we generate the tensor of indices for removal for each row.
        # The main problem is that there might be different values for each row, so we pad remaining indices with -1
        padded_indices_to_remove = torch.full((self.rows, max_non_zeros), -1, dtype=torch.int64, device=self.dev)
        range_tensor = torch.arange(max_non_zeros, device=self.dev).expand(self.rows, max_non_zeros)
        valid_entries_mask = range_tensor < num_lambdas_for_each_row.unsqueeze(1)
        padded_indices_to_remove.masked_scatter_(valid_entries_mask, cols_indices)

        # Initialize b in a such a way to make it padded with zeros
        # b is not quite big, so we can easily store it in full size for all rows
        valid_indices_mask = padded_indices_to_remove >= 0
        b = torch.zeros((self.rows, max_non_zeros), dtype=W1.dtype, device=self.dev)
        b[valid_indices_mask] = W1[torch.arange(self.rows, device=self.dev).unsqueeze(1), padded_indices_to_remove][valid_indices_mask]

        # We divide rows into equal blocks because otherwise we could not fit into GPUs memory
        for r in range(int(self.rows / v_blocksize)):
            r1 = r * v_blocksize
            r2 = min(r1 + v_blocksize, self.rows)

            num_lambdas_for_current_block = num_lambdas_for_each_row[r1:r2]

            # Here we construct R and R_hat. Recall that we use padded indices here, so we need to clear
            # the padded part of R_hat. We can leave R as it is because corresponding lambdas will be zero, so
            # the padded columns/rows will not have any effect on dW
            R = Hinv[padded_indices_to_remove[r1:r2]].transpose(1, 2)
            batch_indices = torch.arange(v_blocksize, device=self.dev).view(-1, 1).expand(-1, max_non_zeros)
            R_hat = R[batch_indices, padded_indices_to_remove[r1:r2]]

            # Make R_hat block-diagonal in the bottom-right
            row_indices = torch.arange(max_non_zeros, device=self.dev).unsqueeze(1).unsqueeze(0).expand(v_blocksize,
                                                                                                        max_non_zeros,
                                                                                                        max_non_zeros)
            col_indices = torch.arange(max_non_zeros, device=self.dev).unsqueeze(0).unsqueeze(1).expand(v_blocksize,
                                                                                                        max_non_zeros,
                                                                                                        max_non_zeros)

            identity_mask = (row_indices >= num_lambdas_for_current_block.unsqueeze(1).unsqueeze(2)) & (row_indices == col_indices)
            zero_mask = (row_indices >= num_lambdas_for_current_block.unsqueeze(1).unsqueeze(2)) | (col_indices >= num_lambdas_for_current_block.unsqueeze(1).unsqueeze(2))

            R_hat[zero_mask] = 0
            R_hat[identity_mask] = 1

            # Solve a batch of linear systems and update weights
            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)
            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        # To avoid deviations from zero after the update
        W1[mask] = 0

        return new_zeros

    # The code for this function is much easier to comprehend because here we do not need to pad indices and R_hat
    def __structured(self, W, Hinv, i1, i2, prune_n, prune_m, v_blocksize):
        W1 = W[:, i1:i2]

        # Wanda metric
        tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

        val, ind = torch.topk(tmp, prune_n, dim=1, largest=False)
        mask = torch.zeros_like(tmp, dtype=torch.bool)
        mask.scatter_(1, ind, True)

        indices = torch.arange(0, prune_m, device=self.dev).unsqueeze(0).repeat(self.rows, 1)
        indices_to_remove = indices[mask].reshape(self.rows, -1)

        b = W1[mask].reshape(self.rows, -1)

        for r in range(int(self.rows / v_blocksize)):
            r1 = r * v_blocksize
            r2 = min(r1 + v_blocksize, self.rows)

            R = Hinv[indices_to_remove[r1:r2]].transpose(1, 2)
            batch_indices = torch.arange(v_blocksize).view(-1, 1).expand(-1, prune_n)
            R_hat = R[batch_indices, indices_to_remove[r1:r2]]

            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)

            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        W1[mask] = 0

    def __compute_l2_loss(self, W, old_W):
        if hasattr(self, 'X'):
            raise AttributeError("Cannot compute L2 loss: self.X is not defined.")

        dW = W - old_W
        loss = 0

        for Xj in self.X:
            dW = dW.float()
            Xj = Xj.float()

            mult = dW @ Xj
            l12 = torch.sum(torch.linalg.norm(mult, dim=1))
            loss += l12
        loss /= len(self.X)

        return loss

    def snap(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, v_blocksize=64, adaptive_blocksize=False):
        W = self.layer.weight.data.clone()

        if hasattr(self, 'X'):
            old_W = W.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if adaptive_blocksize:
            blocksize = int(self.columns/16)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        Hinv = torch.linalg.inv(H)

        tick = time.time()
        zeros = 0

        v_blocksize = min(self.rows, v_blocksize)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            if prune_n == 0:  # unstructured
                zeros += self.__unstructured(W, Hinv, i1, i2, zeros, sparsity, v_blocksize)
            else:  # structured n:m sparsity
                self.__structured(W, Hinv, i1, i2, prune_n, prune_m, v_blocksize)

            Hinv = torch.linalg.inv(H[i2:, i2:])

        print('Layer pruning time %.2f' % (time.time() - tick))


        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            print("Summ(|dW X_j|^2_1,2) =", self.__compute_l2_loss(W, old_W))

    # This is the first version of Thanos with its naive implementation without vectorization.
    # It is much easier to read and understand. Essentially, this represents what is really going on inside Thanos.
    def slowprune(self, sparsity, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()

        if hasattr(self, 'X'):
            old_W = W.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        Hinv = torch.linalg.inv(H)

        tick = time.time()

        for i1 in range(0, self.columns, blocksize):

            i2 = min(i1 + blocksize, self.columns)

            W1 = W[:, i1:i2]

            # SparseGPT metric
            #Hinv1 = Hinv[:blocksize, :blocksize]
            #tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1)))
            # Wanda metric
            tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

            values, indices = torch.topk(tmp.flatten(), int(tmp.numel() * sparsity), largest=False)
            mask = torch.zeros_like(tmp, dtype=torch.bool)
            mask.view(-1)[indices] = True

            for r in range(self.rows):

                mask_r = mask[r]
                if not torch.any(mask_r):
                    continue

                indices = torch.nonzero(mask_r).squeeze(dim=1)

                R = Hinv[:, indices]
                R_hat = R[indices]
                b = W[r, i1 + indices]

                lambdas = torch.linalg.solve(R_hat, b)

                dw = -R @ lambdas

                W[r, i1:] += dw
                W[r, i1 + indices] = torch.zeros_like(indices, dtype=W.dtype, device=self.dev)

            Hinv = torch.linalg.inv(H[i2:, i2:])

        print('Layer pruning time %.2f' % (time.time() - tick))

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            print("Summ(|dW X_j|^2_1,2) =", self.__compute_l2_loss(W, old_W))

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

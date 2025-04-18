import math
import time

import numpy as np
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_heatmap(tensor, filename, title=None):
    vmin, vmax = np.percentile(tensor.cpu().numpy(), [0, 98])

    m, n = tensor.shape

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(tensor.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, n, 0, m])
    plt.colorbar(label='Values')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.tight_layout(pad=0)

    plt.savefig(filename, format='pdf', dpi=100)
    plt.close()


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


def structured_n_m_sparsity_mask(tmp, prune_n, prune_m):
    rows, cols = tmp.shape
    # Calculate the number of chunks per row
    num_chunks = cols // prune_m

    # Reshape tensor to (rows, num_chunks, m)
    reshaped_tensor = tmp.reshape(rows, num_chunks, prune_m)

    # Find the indices of the n smallest elements in each chunk
    _, indices = torch.topk(reshaped_tensor, prune_n, dim=2, largest=False)

    # Create a mask for each chunk
    chunk_mask = torch.zeros_like(reshaped_tensor, dtype=torch.bool)
    chunk_mask.scatter_(2, indices, True)

    # Reshape the chunk mask back to the original tensor shape
    mask = chunk_mask.reshape(rows, num_chunks * prune_m)

    return mask


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

        if hasattr(self, 'X'):
            self.X.append(inp)

        if not hasattr(self, ''):
            self.X_squared_sum = torch.zeros(inp.shape, device=self.dev)


        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.X_squared_sum *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp

        self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2 / self.nsamples

        self.X_squared_sum += inp**2 / self.nsamples

        inp = math.sqrt(2 / self.nsamples) * inp.type(torch.float32)
        self.H += inp.matmul(inp.t())

    # Similar to SparseGPT but with dynamic mask + Wanda metric
    def snap_bs_one(self, sparsity, prune_n=0, prune_m=0, groupsize=128, percdamp=.01):
        W = self.layer.weight.data.clone()

        if hasattr(self, 'X'):
            W_old = W.clone()

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

        # Subsequent pruning
        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)

            block_mask = None
            if use_constant_mask:
                block_mask = const_mask[:, i1:i2]
            else:
                local_block_sparsity = False

                if prune_n == 0 and not local_block_sparsity:
                    glob_tmp = torch.abs(W[:, i1:]) * torch.sqrt(self.scaler_row[i1:]).reshape((1, -1))
                    estimate_zeros = int(sparsity * self.rows * self.columns - zeros)
                    values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)
                    glob_mask = torch.zeros_like(W[:, i1:], dtype=torch.bool, device=self.dev)
                    glob_mask.view(-1)[indices] = True
                    block_mask = glob_mask[:, :groupsize]
                    zeros += torch.sum(block_mask)
                elif prune_n == 0:
                    loc_tmp = torch.abs(W[:, i1:i2]) * torch.sqrt(self.scaler_row[i1:i2]).reshape((1, -1))
                    values, indices = torch.topk(loc_tmp.flatten(), int(loc_tmp.numel()*sparsity), largest=False)
                    loc_mask = torch.zeros_like(W[:, i1:i2], dtype=torch.bool, device=self.dev)
                    loc_mask.view(-1)[indices] = True
                    block_mask = loc_mask
                else:
                    loc_tmp = torch.abs(W[:, i1:i2]) * torch.sqrt(self.scaler_row[i1:i2]).reshape((1, -1))
                    block_mask = structured_n_m_sparsity_mask(loc_tmp, prune_n, prune_m)

            # Fast version:
            W_block = W[:, i1:i2]
            Err = block_mask * W_block / torch.diag(Hinv[i1:i2, i1:i2])
            W[:, i1:i2][block_mask] = 0
            W[:, i2:] -= Err.matmul(Hinv[i1:i2, i2:])

            '''
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(groupsize):
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
            '''

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            self.l2_loss = compute_l2_loss(W - W_old, self.X)
            #print("Summ(|dW X_j|^2_1,2) =", self.l2_loss)

    def __unstructured(self, W, Hinv, i1, i2, zeros, sparsity, v_blocksize, const_mask=None):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1

        # TODO: move this flag from here to args
        use_global_mask = True

        new_zeros = 0
        if const_mask is not None:
            mask = const_mask[:, i1:i2]
        elif use_global_mask:  # Dynamic mask + Wanda metric
            local_W = W[:, i1:]
            glob_tmp = torch.abs(local_W) * torch.sqrt(self.scaler_row[i1:]).reshape((1, -1))

            estimate_zeros = int(sparsity*self.rows*self.columns - zeros)

            # This method of mask construction is more robust (comparison to sparseGPT and Wanda)
            # because it will produce the same number of non-zero elements
            values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)

            glob_mask = torch.zeros_like(local_W, dtype=torch.bool, device=self.dev)
            glob_mask.view(-1)[indices] = True

            mask = glob_mask[:, :blocksize]

            new_zeros = torch.sum(mask)
        else:  # SparseGPT mask on Wanda metric
            tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

            values, indices = torch.topk(tmp.flatten(), int(tmp.numel() * sparsity), largest=False)
            mask = torch.zeros_like(tmp, dtype=torch.bool, device=self.dev)
            mask.view(-1)[indices] = True

        # Here we compute how many elements we remove at once for each row (this is to make appropriate paddings)
        num_lambdas_for_each_row = mask.sum(dim=1)
        max_non_zeros = (num_lambdas_for_each_row.max()).item()

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

        v_blocksize = min(self.rows, v_blocksize)

        # We divide rows into equal blocks because otherwise we could not fit into GPUs memory
        for r in range(self.rows // v_blocksize + 1):
            r1 = r * v_blocksize
            if r1 > self.rows:
                continue

            r2 = min(r1 + v_blocksize, self.rows)

            v_current_block = min(self.rows - r1, v_blocksize)

            num_lambdas_for_current_block = num_lambdas_for_each_row[r1:r2]

            # Here we construct R and R_hat. Recall that we use padded indices here, so we need to clear
            # the padded part of R_hat. We can leave R as it is because corresponding lambdas will be zero, so
            # the padded columns/rows will not have any effect on dW

            R = Hinv[padded_indices_to_remove[r1:r2]].transpose(1, 2)

            batch_indices = torch.arange(v_current_block, device=self.dev).view(-1, 1).expand(-1, max_non_zeros)
            R_hat = R[batch_indices, padded_indices_to_remove[r1:r2]]

            # Make R_hat block-diagonal in the bottom-right
            row_indices = torch.arange(max_non_zeros, device=self.dev).unsqueeze(1).unsqueeze(0).expand(v_current_block,
                                                                                                        max_non_zeros,
                                                                                                        max_non_zeros)
            col_indices = torch.arange(max_non_zeros, device=self.dev).unsqueeze(0).unsqueeze(1).expand(v_current_block,
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
    def __structured_n_m(self, W, Hinv, i1, i2, prune_n, prune_m, v_blocksize):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1
        rows = W1.shape[0]

        # Wanda metric
        tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

        mask = structured_n_m_sparsity_mask(tmp, prune_n, prune_m)

        indices = torch.arange(0, blocksize, device=self.dev).unsqueeze(0).repeat(rows, 1)
        indices_to_remove = indices[mask].reshape(rows, -1)

        b = W1[mask].reshape(rows, -1)

        for r in range(rows // v_blocksize + 1):
            r1 = r * v_blocksize
            if r1 > rows:
                continue

            r2 = min(r1 + v_blocksize, rows)

            v_current_block = min(rows - r1, v_blocksize)

            R = Hinv[indices_to_remove[r1:r2]].transpose(1, 2)
            batch_indices = torch.arange(v_current_block).view(-1, 1).expand(-1, blocksize // 2)
            R_hat = R[batch_indices, indices_to_remove[r1:r2]]

            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)

            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        W1[mask] = 0

    def __structured(self, W, sparsity, percdamp=.01, perc_outliers=0.1):
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        num_outliers = int(perc_outliers * self.rows)
        # Permutation based on the mean value of the Wanda metric (we place parameters for removal to be the first)
        #glob_metric = torch.abs(W) * torch.sqrt(self.scaler_row).reshape((1, -1))

        glob_metric_rows_mean = torch.sum(W**2@self.X_squared_sum, dim=-1)

        #glob_metric_rows_mean = torch.mean(glob_metric, dim=1)
        row_perm = torch.sort(glob_metric_rows_mean).indices
        inv_row_perm = torch.sort(row_perm).indices

        W = W[row_perm, :]

        # W_expanded = W.unsqueeze(2)                   # Shape: (c, b, 1)
        # X_expanded = self.X_squared_sum.unsqueeze(0)  # Shape: (1, b, a)
        # result = W_expanded**2 * X_expanded           # Shape: (c, b, a)
        # result = result.permute(1, 0, 2)              # Shape: (b, c, a)
        # sums = result.sum(dim=(1, 2))  # Sum over dimensions c and a

        # Loop-based implementation to compute the sum of elements for each outer product
        sums = torch.empty(self.columns)  # Resulting tensor of size (b,)
        for i in range(self.columns):
            # Compute the outer product for column `i` and sum all elements
            outer_product = torch.outer(W[:-(num_outliers+1), i] ** 2, self.X_squared_sum[i, :])
            sums[i] = outer_product.sum()

        # plot_heatmap(glob_metric, "heat_map_initial.pdf")

        #glob_metric = torch.abs(W[:-(num_outliers+1)]) * torch.sqrt(self.scaler_row).reshape((1, -1))

        #glob_metric_cols_mean = torch.mean(glob_metric, dim=0)
        col_perm = torch.sort(sums).indices
        inv_col_perm = torch.sort(col_perm).indices

        H = (H[:, col_perm])[col_perm, :]
        W = W[:, col_perm]
        self.scaler_row = self.scaler_row[col_perm]

        Hinv = torch.linalg.inv(H)

        blocksize = math.ceil(sparsity * self.columns/(1.0-perc_outliers))

        #plot_heatmap(glob_metric[:, col_perm], "heat_map_col_perm.pdf")
        #plot_heatmap((glob_metric[:, col_perm])[row_perm, :], "heat_map_col_row_perm.pdf")

        non_outlier_W = W[:-(num_outliers+1)]
        dW = -non_outlier_W[:, :blocksize] @ torch.linalg.inv(Hinv[:blocksize, :blocksize]) @ Hinv[:blocksize]

        non_outlier_W += dW
        non_outlier_W[:, :blocksize] = 0.0

        # Make the inverse permutation
        W = (W[inv_row_perm, :])[:, inv_col_perm]

        print('Layer pruning time %.2f' % (time.time() - tick))

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def snap(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, v_blocksize=64,
             adaptive_blocksize=False, structured=False, perc_outliers=0.1):
        if blocksize == 1:
           return self.snap_bs_one(sparsity=sparsity,
                                   prune_n=prune_n,
                                   prune_m=prune_m,
                                   groupsize=128,
                                   percdamp=percdamp)

        W = self.layer.weight.data.clone()

        if structured:
            self.__structured(W, sparsity, percdamp, perc_outliers)
            return

        if hasattr(self, 'X'):
            W_old = W.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if adaptive_blocksize:
            blocksize = self.columns // 16

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        Hinv = torch.linalg.inv(H)

        if prune_n != 0:
            num_outliers = int(perc_outliers * self.rows)

            glob_metric_rows_mean = torch.sum(W ** 2 @ self.X_squared_sum, dim=-1)

            # glob_metric_rows_mean = torch.mean(glob_metric, dim=1)
            row_perm = torch.sort(glob_metric_rows_mean).indices
            inv_row_perm = torch.sort(row_perm).indices

            W = W[row_perm, :]

            non_outlier_W = W[:-(num_outliers + 1)]

        zeros = 0

        # TODO: move this flag from here to args
        use_constant_mask = False

        const_mask = None
        if use_constant_mask:
            # Global Wanda mask
            glob_tmp = torch.abs(W) * torch.sqrt(self.scaler_row).reshape((1, -1))
            sort_res = torch.sort(glob_tmp, dim=-1, stable=True)
            indices = sort_res[1][:, :int(glob_tmp.shape[1] * sparsity)]
            const_mask = torch.zeros_like(W, dtype=torch.bool, device=self.dev)
            const_mask.scatter_(1, indices, True)

        # Subsequent pruning
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            if prune_n == 0 and not structured:  # unstructured
                zeros += self.__unstructured(W, Hinv, i1, i2, zeros, sparsity, v_blocksize, const_mask)
            elif prune_n != 0:  # structured n:m sparsity
                self.__structured_n_m(non_outlier_W, Hinv, i1, i2, prune_n, prune_m, v_blocksize)
            Hinv = torch.linalg.inv(H[i2:, i2:])

        print('Layer pruning time %.2f' % (time.time() - tick))
        #print('Sparsity: ', torch.sum(W == 0.0).item() / (self.rows * self.columns))

        if prune_n != 0:
            W = (W[inv_row_perm, :])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            self.l2_loss = compute_l2_loss(W - W_old, self.X)
            #print("Summ(|dW X_j|^2_1,2) =", self.l2_loss)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

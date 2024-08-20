import math
import time

import torch
import torch.nn as nn
import transformers
import itertools

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def reshuffle_tensor(tensor):
    # Generate a random permutation of the indices of the tensor
    permuted_indices = torch.randperm(tensor.size(0))
    # Return the tensor reshuffled according to the permuted indices
    return tensor[permuted_indices]


def generate_masks(m, n):
    """
    Generate all variations of masks with m elements in total, and n of them being non-zero.

    Parameters:
    m (int): Total number of elements in each mask.
    n (int): Number of non-zero elements in each mask.

    Returns:
    torch.Tensor: A tensor of size (C, m) where C is the total number of masks and m is the number of elements in each mask.
    """
    # Generate all combinations of positions for the non-zero elements
    indices = list(itertools.combinations(range(m), n))

    # Number of combinations
    c = len(indices)

    # Initialize the tensor
    masks = torch.zeros((c, m), dtype=torch.bool)

    # Populate the tensor with masks
    for i, combination in enumerate(indices):
        for idx in combination:
            masks[i, idx] = True

    return masks


def l12_loss(dW, X):
    dW = dW.float()
    X = X.float()

    mult = dW @ X
    l12 = torch.sum(torch.linalg.norm(mult, dim=1)**2)

    return l12


def active_weights_mask(batches_to_remove, blocksize, mask_size):
    mask = torch.ones(mask_size, dtype=torch.bool)

    # Check if batches_to_remove is empty
    if batches_to_remove.numel() == 0:
        return mask

    # Calculate the start indices of the batches to be removed
    start_indices = batches_to_remove * blocksize

    # Calculate all indices to be removed
    remove_indices = start_indices[:, None] + torch.arange(blocksize)

    # Flatten the remove_indices tensor
    remove_indices = remove_indices.flatten()
    mask[remove_indices] = False

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

        self.scaler_row = torch.zeros((self.columns), device=self.dev)

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

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp

        # Simple sum
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        # Weighted sum
        #W = self.layer.weight.data
        #l12 = l12_loss(W, inp).item()
        #self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / (1e-5*l12*self.nsamples)

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # Simple sum
        self.H += inp.matmul(inp.t())
        # Weighted sum
        #self.H += inp.matmul(inp.t())/l12


    def __unstructured(self, W, Hinv, i1, i2, zeros, sparsity, v_blocksize):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1

        # TODO: move it from here
        use_global_mask = True

        if use_global_mask:
            # Global Wanda metric
            glob_tmp = torch.abs(W[:, i1:]) * torch.sqrt(self.scaler_row[i1:]).reshape((1, -1))

            estimate_zeros = int(sparsity*self.rows*self.columns - zeros)

            # This method of mask construction is more robust (comparison to sparseGPT and Wanda)
            # because it will produce the same number of non-zero elements
            values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)

            glob_mask = torch.zeros_like(W[:, i1:], dtype=torch.bool, device=self.dev)
            glob_mask.view(-1)[indices] = True

            mask = glob_mask[:, :blocksize]
        else:
            # Wanda metric
            tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

            values, indices = torch.topk(tmp.flatten(), int(tmp.numel() * sparsity), largest=False)
            mask = torch.zeros_like(tmp, dtype=torch.bool, device=self.dev)
            mask.view(-1)[indices] = True

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


    def __unstructured_random_bathes(self, W, Hinv, i1, i2, zeros, sparsity, v_blocksize, removed_blocks, block_to_remove):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1

        active_weights = active_weights_mask(removed_blocks, blocksize, self.columns)
        active_idx = active_weights.nonzero(as_tuple=True)[0]
        W_active = W[:, active_weights]



        # Global Wanda metric
        active_scaler_row = self.scaler_row[active_weights]
        glob_tmp = torch.abs(W_active) * torch.sqrt(active_scaler_row).reshape((1, -1))

        estimate_zeros = int(sparsity * self.rows * self.columns - zeros)

        # This method of mask construction is more robust (comparison to sparseGPT and Wanda)
        # because it will produce the same number of non-zero elements
        values, indices = torch.topk(glob_tmp.flatten(), estimate_zeros, largest=False)

        glob_mask = torch.zeros_like(W_active, dtype=torch.bool, device=self.dev)

        glob_mask.view(-1)[indices] = True

        glob_mask_full = torch.zeros_like(W, dtype=torch.bool, device=self.dev)
        glob_mask_full[:, active_weights] = glob_mask
        mask = glob_mask_full[:, block_to_remove * blocksize:(block_to_remove + 1) * blocksize]



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

            # Random bathed pruning
            H_inv_full_sized = torch.zeros((self.columns, self.columns), device=self.dev)
            H_inv_full_sized[active_idx[:, None], active_idx] = Hinv
            R = H_inv_full_sized[block_to_remove*blocksize + padded_indices_to_remove[r1:r2]].transpose(1, 2)

            batch_indices = torch.arange(v_current_block, device=self.dev).view(-1, 1).expand(-1, max_non_zeros)

            R_hat = R[batch_indices, block_to_remove * blocksize + padded_indices_to_remove[r1:r2]]

            R_active = R[:, active_weights, :]

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
            W_active[r1:r2] -= torch.bmm(R_active, lambdas).squeeze(2)

        W[:, active_weights] = W_active

        # To avoid deviations from zero after the update
        W1[mask] = 0

        return new_zeros


    def __unstructured_same_for_all_rows(self, W, Hinv, i1, i2, zeros, sparsity, v_blocksize):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1



        # Global Wanda metric
        glob_tmp = torch.abs(W[:, i1:]) * torch.sqrt(self.scaler_row[i1:]).reshape((1, -1))

        glob_tmp_rows_mean = torch.mean(glob_tmp, dim=0)

        estimate_zeros = int(sparsity*self.columns - zeros)

        # This method of mask construction is more robust (comparison to sparseGPT and Wanda)
        # because it will produce the same number of non-zero elements
        values, indices = torch.topk(glob_tmp_rows_mean, estimate_zeros, largest=False)

        glob_mask = torch.zeros_like(W[:, i1:], dtype=torch.bool, device=self.dev)
        glob_mask[:, indices] = True

        mask = glob_mask[:, :blocksize]

        indices_to_remove = torch.nonzero(mask[0], as_tuple=True)[0].repeat(self.rows, 1)

        new_zeros = torch.sum(mask[0]).item()

        b = W1[mask].reshape(self.rows, -1)

        v_blocksize = min(self.rows, v_blocksize)

        # We divide rows into equal blocks because otherwise we could not fit into GPUs memory
        for r in range(self.rows // v_blocksize + 1):
            r1 = r * v_blocksize
            if r1 > self.rows:
                continue

            r2 = min(r1 + v_blocksize, self.rows)

            v_current_block = min(self.rows - r1, v_blocksize)

            # Here we construct R and R_hat. Recall that we use padded indices here, so we need to clear
            # the padded part of R_hat. We can leave R as it is because corresponding lambdas will be zero, so
            # the padded columns/rows will not have any effect on dW

            R = Hinv[indices_to_remove[r1:r2]].transpose(1, 2)

            batch_indices = torch.arange(v_current_block, device=self.dev).view(-1, 1).expand(-1, new_zeros)
            R_hat = R[batch_indices, indices_to_remove[r1:r2]]

            # Solve a batch of linear systems and update weights
            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)
            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        # To avoid deviations from zero after the update
        W1[mask] = 0

        return new_zeros


    def __structured_n_m_sparsity_mask(self, tmp, prune_n, prune_m):
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

    # The code for this function is much easier to comprehend because here we do not need to pad indices and R_hat
    def __structured(self, W, Hinv, i1, i2, prune_n, prune_m, v_blocksize):
        W1 = W[:, i1:i2]
        blocksize = i2 - i1

        # Wanda metric
        tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))
        mask = self.__structured_n_m_sparsity_mask(tmp, prune_n, prune_m)

        indices = torch.arange(0, blocksize, device=self.dev).unsqueeze(0).repeat(self.rows, 1)
        indices_to_remove = indices[mask].reshape(self.rows, -1)

        b = W1[mask].reshape(self.rows, -1)
        for r in range(self.rows // v_blocksize):
            r1 = r * v_blocksize
            r2 = min(r1 + v_blocksize, self.rows)

            R = Hinv[indices_to_remove[r1:r2]].transpose(1, 2)
            batch_indices = torch.arange(v_blocksize).view(-1, 1).expand(-1, blocksize//2)
            R_hat = R[batch_indices, indices_to_remove[r1:r2]]

            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)

            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        W1[mask] = 0

    def __semistructured(self, W, Hinv, i1, i2, sparsity, v_blocksize):
        blocksize = i2 - i1
        W1 = W[:, i1:i2]

        rows, cols = W1.shape

        num_lambdas = int(blocksize * sparsity)
        tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1, -1)))

        val, ind = torch.topk(tmp, num_lambdas, dim=1, largest=False)
        mask = torch.zeros_like(tmp, dtype=torch.bool)
        mask.scatter_(1, ind, True)

        indices = torch.arange(0, blocksize, device=self.dev).unsqueeze(0).repeat(rows, 1)
        indices_to_remove = indices[mask].reshape(rows, -1)

        b = W1[mask].reshape(rows, -1)

        for r in range(rows // v_blocksize):
            r1 = r * v_blocksize
            r2 = min(r1 + v_blocksize, rows)

            R = Hinv[indices_to_remove[r1:r2]].transpose(1, 2)
            batch_indices = torch.arange(v_blocksize).view(-1, 1).expand(-1, num_lambdas)
            R_hat = R[batch_indices, indices_to_remove[r1:r2]]

            lambdas = torch.linalg.solve(R_hat, b[r1:r2]).unsqueeze(2)

            W[r1:r2, i1:] -= torch.bmm(R, lambdas).squeeze(2)

        W1[mask] = 0

    def __compute_l2_loss(self, dW):
        if not hasattr(self, 'X'):
            raise AttributeError("Cannot compute L2 loss: self.X is not defined.")

        loss = 0

        for Xj in self.X:
            loss += l12_loss(dW, Xj)
        loss /= len(self.X)

        return loss

    def __compute_l2_relative_loss(self, W, W_old):
        if not hasattr(self, 'X'):
            raise AttributeError("Cannot compute L2 loss: self.X is not defined.")

        dW = W - W_old

        loss = 0

        for Xj in self.X:
            l12 = l12_loss(dW, Xj)
            l12_abs = l12_loss(W_old, Xj)

            loss += l12/l12_abs

        loss /= len(self.X)

        return loss

    def snap(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, v_blocksize=64, adaptive_blocksize=False):
        W = self.layer.weight.data.clone()

        if hasattr(self, 'X'):
            W_old = W.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if adaptive_blocksize:
            blocksize = self.columns // 8

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

        # Global Wanda mask
        '''
        num_lambdas = int(self.columns * sparsity)
        tmp = torch.abs(W) * torch.sqrt(self.scaler_row.reshape((1, -1)))
        val, ind = torch.topk(tmp, num_lambdas, dim=1, largest=False)
        mask_global = torch.zeros_like(tmp, dtype=torch.bool, device=self.dev)
        mask_global.scatter_(1, ind, True)
        '''


        # Unstructured pruning with random bathes
        '''
        bath_of_blocks = reshuffle_tensor(torch.arange(0, self.columns//blocksize))
        
        for b in range(bath_of_blocks.shape[0]):
            block = bath_of_blocks[b]
            i1 = block*blocksize
            i2 = min(i1 + blocksize, self.columns)

            zeros += self.__unstructured_random_bathes(W, Hinv, i1, i2, zeros, sparsity, v_blocksize, removed_blocks=bath_of_blocks[:b], block_to_remove=block.item())


            curent_slice = active_weights_mask(bath_of_blocks[:b+1], blocksize, self.columns)
            Hinv = torch.linalg.inv(H[curent_slice][:, curent_slice])
        '''


        # Subsequent pruning
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            if prune_n == 0:  # unstructured
                zeros += self.__unstructured(W, Hinv, i1, i2, zeros, sparsity, v_blocksize)
                #zeros += self.__unstructured_same_for_all_rows(W, Hinv, i1, i2, zeros, sparsity, v_blocksize)
                #self.__semistructured(W, Hinv, i1, i2, sparsity, v_blocksize)
            else:  # structured n:m sparsity
                self.__structured(W, Hinv, i1, i2, prune_n, prune_m, v_blocksize)

            Hinv = torch.linalg.inv(H[i2:, i2:])


        print('Layer pruning time %.2f' % (time.time() - tick))


        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            self.l2_loss = self.__compute_l2_loss(W - W_old)
            #self.l2_loss = self.__compute_l2_relative_loss(W, W_old)
            print("Summ(|dW X_j|^2_1,2) =", self.l2_loss)

    # This is the first version of Thanos with its naive implementation without vectorization.
    # It is much easier to read and understand. Essentially, this represents what is really going on inside Thanos.
    def slowprune(self, sparsity, blocksize=128, percdamp=.01):
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
            print("Summ(|dW X_j|^2_1,2) =", self.__compute_l2_loss(W - W_old))

    def slowprune_structured_optimal(self, prune_n=0, prune_m=0, percdamp=.01):
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

        Hinv = torch.linalg.inv(H)
        H_current = H

        tick = time.time()

        for i1 in range(0, self.columns, prune_m):
            i2 = min(i1 + prune_m, self.columns)

            for r in range(self.rows):

                masks = generate_masks(prune_m, prune_n)

                best_S = 1e10
                best_mask_r = masks[0]
                best_dw = None
                for current_mask in masks:
                    indices = torch.nonzero(current_mask).squeeze(dim=1)

                    R = Hinv[:, indices]
                    R_hat = R[indices]
                    b = W[r, i1 + indices]

                    lambdas = torch.linalg.solve(R_hat, b)

                    dw = -R @ lambdas

                    current_S = dw.t()@H_current@dw
                    if current_S < best_S:
                        best_S = current_S
                        best_mask_r = current_mask
                        best_dw = dw

                W[r, i1:] += best_dw
                W[r, i1 + best_mask_r] = torch.zeros_like(best_mask_r, dtype=W.dtype, device=self.dev)

            H_current = H[i2:, i2:]
            Hinv = torch.linalg.inv(H_current)

        print('Layer pruning time %.2f' % (time.time() - tick))

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if hasattr(self, 'X'):
            print("Summ(|dW X_j|^2_1,2) =", self.__compute_l2_loss(W - W_old))

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

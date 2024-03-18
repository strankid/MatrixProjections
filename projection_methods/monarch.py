import torch
import einops
import math
import numpy as np

### ISSUE: ONLY WORKS WITH SQUARE MATRICES
### TO-DO: FIND A METHOD THAT WORKS WITH NON-SQUARE MATRICES

from projection_methods import approximator

class  MonarchApproximator(approximator.Approximator):
    def get_name(self):
        return "MonarchApproximator"
    
    def factors(self, n):
        return [(i, n // i) for i in range(1, math.floor(math.sqrt(n)) + 1) if n % i == 0]

    def low_rank_project(self, M, rank):
        """Supports batches of matrices as well."""
        U, S, Vt = torch.linalg.svd(M)
        S_sqrt = S[..., :rank].sqrt()
        U = U[..., :rank] * einops.rearrange(S_sqrt, '... rank -> ... 1 rank')
        Vt = einops.rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
        return U, Vt

    def blockdiag_butterfly_project(self, M, sizes=None):
        """Only works for square matrices for now"""
        m, n = M.shape
        if m != n:
            raise NotImplementedError('Only support square matrices')
        if sizes is None:
            # Find the factors that are closest to sqrt(n)
            sizes = self.factors(n)[-1]
            # Larger factor first is probably more efficient, idk
            sizes = (sizes[1], sizes[0])
        assert n == sizes[0] * sizes[1]
        M_permuted_batched = einops.rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
        U, Vt = self.low_rank_project(M_permuted_batched, rank=1)
        w1_bfly = einops.rearrange(Vt, 'k r 1 s -> r k s')
        w2_bfly = einops.rearrange(U, 'k r s 1 -> k s r')
        return w1_bfly, w2_bfly
    
    def blockdiag_butterfly_multiply(self, x, w1_bfly, w2_bfly):
        batch, n = x.shape
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q

        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = torch.nn.functional.linear(x, w1_dense)
        out1 = einops.rearrange(out1, 'b (r l) -> b (l r)', l=l)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = torch.nn.functional.linear(out1, w2_dense)
        out2 = einops.rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out2

    def approximate(self, optim_mat: np.ndarray):
        x = torch.eye(optim_mat.shape[0])
        optim_mat = torch.from_numpy(optim_mat)
        w1_bfly_projected, w2_bfly_projected = self.blockdiag_butterfly_project(optim_mat)
        bfly_projected = self.blockdiag_butterfly_multiply(x, w1_bfly_projected, w2_bfly_projected).t()

        res_dict = dict()
        res_dict["type"] = "MonarchAproximator"
        res_dict["left_mat"] = w1_bfly_projected.numpy()
        res_dict["right_mat"] = w2_bfly_projected.numpy()
        res_dict["approx_mat_dense"] = bfly_projected.numpy()
        res_dict["nb_parameters"] = np.prod(w1_bfly_projected.shape) + np.prod(w2_bfly_projected.shape)
        return res_dict

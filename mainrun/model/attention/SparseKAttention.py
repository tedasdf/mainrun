import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import AttnConfig, CausalSelfAttention
import math 


@dataclass
class SparseKAttnConfg(AttnConfig):
    epilson: int
    k: int



class SparseKAttention(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.k = cfg.k
       
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        u = self.select(x)

        m_topk = self.topk_mask(u, self.k)
        p, tau_i, _, _ = self.sparsek_1d(u, self.k)

        select_mat = self.mask_select_diag(p , m_topk)

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        
        k_hat = select_mat @ k
        v_hat = select_mat @ v

        p_hat = F.softmax(q @ k_hat.T )
        o = p_hat.T @ v_hat
        y = o.transpose(1, 2).contiguous().view(B, T, self.bottleneck_dim)
        return self.resid_drop(self.proj(y))
        
    def select(self, X):
        B, T ,C = X.shape()
        # shape: (1, T, 1), broadcastable to (B, T, 1)
        pos = torch.arange(1, T + 1, device=X.device, dtype=X.dtype).unsqueeze(0).unsqueeze(-1)
        u = self.w_score(X) + self.epilson * pos
        return u # (B, T , 1)
    
    def topk_mask(self, u, k):
        """
        u: (B, T) importance scores up to current position i
        k: number of key-value pairs to select
        returns: m_topk: binary mask (B, T)
        """
        B, T = u.shape
        m_topk = torch.zeros_like(u, dtype=torch.float32)

        # select top-k indices
        topk_vals, topk_idx = torch.topk(u, min(k, T), dim=1)
        m_topk[torch.arange(B).unsqueeze(1), topk_idx] = 1.0
        return m_topk


    def sparsek_1d(self, z: torch.Tensor, k: float):
        """
        Evaluate SparseK(z, k) per Algorithm 1 in 'Sparser is Faster and Less is More'.
        z: 1-D tensor of length m
        k: float in [0, m]
        returns: p (1-D), tau, U, W
        """
        assert z.dim() == 1, "z must be 1-D"
        m = z.numel()
        if k <= 0:
            return torch.zeros_like(z), z.new_tensor(float('inf')), 0, 0
        if k >= m:
            return torch.ones_like(z), z.new_tensor(float('-inf')), m, m

        # 1) sort z descending and keep permutation to undo at end
        z_sorted, perm = torch.sort(z, descending=True)
        # prefix sums with S[0] = 0 so S[t] = sum_{i=0..t-1} z_sorted[i]
        S = torch.cat([z_sorted.new_zeros(1), z_sorted.cumsum(0)])  # shape m+1
        z_pad = torch.cat([z_sorted, z_sorted.new_tensor([-float('inf')])])  # sentinel at index m

        # 2) beta candidates: sort descending as paper suggests
        betas = torch.cat([z_sorted, z_sorted - 1.0])
        betas, _ = torch.sort(betas, descending=True)

        # 3) compute U = # {i : z_i >= beta+1},  W = # {i : z_i > beta}
        # torch.searchsorted expects ascending order; use reversed copy
        za = torch.flip(z_sorted, dims=[0])  # ascending
        # U: >= beta+1  ==> count = m - idx_of_first_ge
        U = m - torch.searchsorted(za, betas + 1.0, right=False)
        # W: >  beta    ==> count = m - idx_of_first_gt  (i.e., right=True for <= beta)
        W = m - torch.searchsorted(za, betas, right=True)

        # 4) candidate taus for those with W > U
        valid = W > U
        Uv = U[valid]
        Wv = W[valid]
        if Uv.numel() == 0:
            # fallback to bisection (pathological, e.g., all z equal and integer k edges)
            return self._sparsek_bisect(z, k)

        # sum over [U..W-1] in 0-based is S[W] - S[U]
        seg_sum = S[Wv] - S[Uv]
        tau = z.new_empty_like(betas)
        tau[valid] = (seg_sum + (Uv.to(z.dtype) - k)) / (Wv - Uv).to(z.dtype)

        # 5) interval checks (with sentinels)
        # For counts U,W: z_{(W)} > tau >= z_{(W+1)} and z_{(U)} >= tau+1 > z_{(U+1)}
        # Convert counts to 0-based indices carefully
        left_ok  = z_sorted[(W.clamp_min(1) - 1)] > tau
        right_ok = tau >= z_pad[W]
        up_ok    = (U == 0) | (z_sorted[(U - 1)] >= (tau + 1.0))
        down_ok  = (tau + 1.0) > z_pad[U]
        ok = valid & left_ok & right_ok & up_ok & down_ok

        idx = torch.nonzero(ok, as_tuple=False)
        if idx.numel() == 0:
            # very rare numeric tie cases
            return self._sparsek_bisect(z, k)

        i = idx[0].item()        # first candidate in descending beta order
        Ui = int(U[i].item())
        Wi = int(W[i].item())
        tau_i = tau[i].to(z.dtype)

        p_sorted = torch.clamp(z_sorted - tau_i, 0.0, 1.0)

        # unsort to original order
        p = torch.empty_like(z)
        p[perm] = p_sorted
        return p, tau_i, Ui, Wi


    def _sparsek_bisect(self, z: torch.Tensor, k: float, iters: int = 60):
        """Robust monotone solve for tau: sum clip(z - tau,0,1) = k."""
        lo = (z.min() - 1.0).item()
        hi = z.max().item()
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            s = torch.clamp(z - mid, 0, 1).sum().item()
            if s > k:
                lo = mid
            else:
                hi = mid
        tau = z.new_tensor((lo + hi) / 2.0)
        p = torch.clamp(z - tau, 0.0, 1.0)
        return p, tau, None, None
  

    def mask_select_diag(self, msparsek, mtopk):
        """
        Args:
            msparsek: Tensor of shape (T,) or (1, T) with soft scores
            mtopk: Tensor of shape (T,) or (1, T) with 0/1 hard selection

        Returns:
            Tensor of shape (T, T): Diagonal matrix with masked scores
        """
        # Ensure the vectors are 1D
        msparsek = msparsek.flatten()
        mtopk = mtopk.flatten()
        
        # Create diagonal matrix from msparsek
        diag_matrix = torch.diag(msparsek)
        
        # Apply row mask: keep only rows where mtopk == 1
        mask = mtopk.unsqueeze(1).expand_as(diag_matrix)  # shape (T, T)
        masked_matrix = diag_matrix * mask.float()
        
        return masked_matrix
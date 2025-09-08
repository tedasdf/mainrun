import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import AttnConfig
import math 


@dataclass
class SparseKAttnConfg(AttnConfig):
    epilson: int



class SparseKAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Use bottleneck_dim if provided, else fall back to d_model
        self.bottleneck_dim = cfg.bottleneck_dim if cfg.bottleneck_dim is not None else cfg.d_model
        assert self.bottleneck_dim % cfg.n_head == 0, "bottleneck_dim must be divisible by n_head"
        self.head_dim = self.bottleneck_dim // cfg.n_head  # Per-head dimension
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        # QKV projection to bottleneck_dim * 3 (instead of d_model * 3)
        self.qkv = nn.Linear(cfg.d_model, 3 * self.bottleneck_dim)
        # Output projection from bottleneck_dim to d_model
        self.proj = nn.Linear(self.bottleneck_dim, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        
        self.w_score = nn.Linear(cfg.d_model, 1) # scoring linear
        self.epilson = cfg.epilson
       
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.sparsek_mask(att)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.bottleneck_dim)
        return self.resid_drop(self.proj(y))

    def sparsek_mask(self, att):
        """
        Create a sparse mask that selects only the top-k attention scores for each query.
        att: (B, n_head, T, T)
        """
        B, n_head, T, _ = att.size()
        k = self.k

        
    def SparseKop(self, u):
        """"
        Sparse K operation: select k keys and values based on some strategyou
        output :
        Mtopk:
        threshold
        """
        pass
    

    def select(self, X):
        B, T ,C = X.shape()
        # shape: (1, T, 1), broadcastable to (B, T, 1)
        pos = torch.arange(1, T + 1, device=X.device, dtype=X.dtype).unsqueeze(0).unsqueeze(-1)
        u = self.w_score(X) + self.epilson * pos
        return u # (B, T , 1)
    
    def topk(self, u, k):
        
        return m_topk
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # y = y.transpose(1, 2).contiguous().view(B, T, C)
        # return self.proj(y)

    def sparsek_1d(z: torch.Tensor, k: float):
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
            return _sparsek_bisect(z, k)

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
            return _sparsek_bisect(z, k)

        i = idx[0].item()        # first candidate in descending beta order
        Ui = int(U[i].item())
        Wi = int(W[i].item())
        tau_i = tau[i].to(z.dtype)

        p_sorted = torch.clamp(z_sorted - tau_i, 0.0, 1.0)

        # unsort to original order
        p = torch.empty_like(z)
        p[perm] = p_sorted
        return p, tau_i, Ui, Wi


    def _sparsek_bisect(z: torch.Tensor, k: float, iters: int = 60):
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
`
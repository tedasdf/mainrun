import torch
import torch.nn as nn
import torch.nn.functional as F

import math 


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
    

    def select(self, k, v):
        pass
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # y = y.transpose(1, 2).contiguous().view(B, T, C)
        # return self.proj(y)

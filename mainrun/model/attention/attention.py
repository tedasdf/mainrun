import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import math

@dataclass
class AttnConfig:
    d_model: int
    n_head: int
    block_size: int
    dropout: float
    bottleneck_dim: int = None
    stride: int = 8  # For strided sparse attention
    fixed_positions: int = 16  # Number of fixed positions for fixed sparse attention



class CausalSelfAttention(nn.Module):
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
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))


    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.bottleneck_dim)
        return self.resid_drop(self.proj(y))


class StridedSparseCausalSelfAttention(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stride = cfg.stride
        self.register_buffer("tril", self._create_sparse_causal_mask(cfg.block_size, cfg.stride))

    def _create_sparse_causal_mask(self, block_size, stride):
        # Create a strided sparse causal mask
        mask = torch.zeros(block_size, block_size)
        for i in range(block_size):
            # Token attends to itself and every stride-th previous token
            for j in range(i, -1, -stride):
                mask[i, j] = 1
        return mask.bool()

class FixedSparseCausalSelfAttention(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fixed_positions = cfg.fixed_positions
        self.register_buffer("tril", self._create_sparse_causal_mask(cfg.block_size, cfg.fixed_positions))

    def _create_sparse_causal_mask(self, block_size, fixed_positions):
        # Create a fixed sparse causal mask with evenly spaced positions
        mask = torch.zeros(block_size, block_size)
        # Choose fixed positions (e.g., evenly spaced)
        step = max(1, block_size // fixed_positions)
        fixed_indices = list(range(0, block_size, step))[:fixed_positions]
        for i in range(block_size):
            # Token attends to itself and the fixed positions that are <= i
            mask[i, i] = 1  # Always attend to self
            for j in fixed_indices:
                if j <= i:
                    mask[i, j] = 1
        return mask.bool()
    

if __name__ == "__main__":
    cfg = AttnConfig(
        d_model=512,
        n_head=8,
        block_size=128,
        dropout=0.1,
        bottleneck_dim=256,
        stride=8,           # Stride for strided attention
        fixed_positions=16  # Number of fixed positions
    )

    # Test strided sparse attention
    strided_attn = StridedSparseCausalSelfAttention(cfg)
    x = torch.randn(2, 128, 512)
    output = strided_attn(x)
    print("Strided Sparse Attention Output Shape:", output.shape)

    # Test fixed sparse attention
    fixed_attn = FixedSparseCausalSelfAttention(cfg)
    output = fixed_attn(x)
    print("Fixed Sparse Attention Output Shape:", output.shape)
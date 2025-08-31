import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import math


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


if __name__ == "__main__":
    cfg = AttnConfig(
        d_model=512,
        n_head=8,
        block_size=128,
        dropout=0.1,
        bottleneck_dim=256  # Half of d_model
    )

    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 128, 512)  # [batch_size, seq_len, d_model]
    output = attn(x)
    print(output.shape)
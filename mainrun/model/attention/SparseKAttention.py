import torch
import torch.nn as nn
import torch.nn.functional as F

import math 


class SparseKAttention(nn.Module):
    def __init__(self, cfg):
        super(SparseKAttention, self).__init__()

        self.p = cfg.d_model / cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, self.p * 3)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.p).transpose(1, 2)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        S = q @ k.transpose(-2, -1)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        P = F.softmax(S, dim=-1)
        O = P @ v
        

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # y = y.transpose(1, 2).contiguous().view(B, T, C)
        # return self.proj(y)

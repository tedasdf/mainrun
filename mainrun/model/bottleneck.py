import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BottleneckGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    bottleneck_size: int

# class BottleneckAttention(nn.Module):
#     def __init__(self, cfg: BottleneckGPTConfig):
#         super().__init__()
#         assert cfg.d_model % cfg.n_head == 0
#         self.head_dim = cfg.d_model // cfg.n_head
#         self.n_head   = cfg.n_head
#         self.bottleneck_size = cfg.bottleneck_size

#         self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
#         self.bottleneck_proj = nn.Linear(cfg.d_model, cfg.bottleneck_size)
#         self.proj = nn.Linear(cfg.d_model, cfg.d_model)
#         self.attn_drop = nn.Dropout(cfg.dropout)
#         self.resid_drop= nn.Dropout(cfg.dropout)
#         self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

#     def forward(self, x: torch.Tensor):
#         B, T, C = x.size()
#         qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
#         q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

#         # Project to bottleneck space
#         k_bottleneck = self.bottleneck_proj(k.reshape(B, T, C)).view(B, T, self.n_head, -1).transpose(1, 3)
#         v_bottleneck = self.bottleneck_proj(v.reshape(B, T, C)).view(B, T, self.n_head, -1).transpose(1, 3)

#         att = (q @ k_bottleneck.transpose(-2, -1)) * (1.0 / (k_bottleneck.size(-1) ** 0.5))
#         att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v_bottleneck
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.resid_drop(self.proj(y))




class BottleneckGPT(nn.Module):
    def __init__(self, cfg: BottleneckGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        
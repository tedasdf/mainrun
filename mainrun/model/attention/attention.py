import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math



@dataclass
class AttnConfig:
    d_model: int
    n_head: int
    block_size: int
    dropout: float

@dataclass
class BottleneckAttnConfig(AttnConfig):
    bottleneck_dim: int  # must be specified for bottleneck attention


@dataclass
class SparseAttnConfig(AttnConfig):
    attn_type: str  # 'fixed_sparse' or 'strided_sparse'
    num_verts: int
    local_attn_ctx: int
    sparseblocksize: int
    vertsize: int
    n_bctx: int


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttnConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.intermediate_dim  = cfg.d_model
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
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
        y = y.transpose(1, 2).contiguous().view(B, T, self.intermediate_dim)
        return self.resid_drop(self.proj(y))

class CausalBottleneckAttn(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)  # init parent first

        # Set bottleneck_dim after parent init
        self.intermediate_dim = cfg.bottleneck_dim if getattr(cfg, "bottleneck_dim", None) is not None else cfg.d_model
        assert self.intermediate_dim % cfg.n_head == 0, "bottleneck_dim must be divisible by n_head"

        # Override head_dim and projection layers
        self.head_dim = self.intermediate_dim // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * self.intermediate_dim)
        self.proj = nn.Linear(self.intermediate_dim, cfg.d_model)

class SparseCausalSelfAttention(CausalSelfAttention):
    def __init__(self, cfg, n_ctx):
        super().__init__(cfg)
        self.attn_type = cfg.type
        self.register_buffer("tril", self._create_sparse_causal_mask(cfg, n_ctx))

    def generate_fixed_attention_mask(self,cfg,n_ctx):
        """
        Generate and optionally visualize a fixed sparse attention mask for all heads using a config object.

        Parameters:
        - cfg (object): Configuration object with attributes:
            - n_heads (int): Number of attention heads.
            - num_verts (int): Number of vertical splits for attention pattern.
            - local_attn_ctx (int): Local attention context size.
            - blocksize (int): Size of blocks for local attention (forms 'smaller triangles').
            - vertsize (int): Size of vertical chunks for fixed positions.
            - n_bctx (int): Block context size (sequence length for attention).
        
        Returns:
        - layout (np.ndarray): Attention mask(s) as boolean array(s). Shape is (n_bctx, n_bctx) for num_verts=1,
        or (n_heads, n_bctx, n_bctx) for num_verts>1.
        """
        # Validate required config attributes
        required_attrs = ['n_heads', 'num_verts', 'local_attn_ctx', 'sparseblocksize', 'vertsize', 'n_bctx']
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                raise ValueError(f"Config object missing required attribute: {attr}")

        # Extract parameters from cfg
        n_heads = cfg.n_heads
        num_verts = cfg.num_verts
        local_attn_ctx = cfg.local_attn_ctx
        blocksize = cfg.sparseblocksize
        vertsize = cfg.vertsize
        n_bctx = cfg.n_bctx

        if self.attn_type in ['all', 'fixed']:
           pass
        elif self.attn_type == 'local':
            assert local_attn_ctx % blocksize == 0
            extra_diagonals = local_attn_ctx // blocksize
        elif self.attn_type == 'strided':
            bT_ctx = n_ctx // local_attn_ctx
            assert bT_ctx % blocksize == 0
            block_chunks = bT_ctx // blocksize
        else:
            raise ValueError(f'attn mode {self.attn_type} invalid')


        if self.attn_type == 'fixed':
            # Validate parameters
            if n_heads % num_verts != 0:
                raise ValueError("n_heads must be divisible by num_verts")
            stride = local_attn_ctx // blocksize
            if vertsize > stride:
                raise ValueError("vertsize must be <= stride")
            if stride % vertsize != 0:
                raise ValueError("stride must be divisible by vertsize")

            # Generate indices for fixed attention pattern
            indices = [i for i in range(stride - 1, -1, -1)]
            indices = np.array(indices).reshape([-1, vertsize])
      
            if num_verts == 1:
                # Single mask for all heads
                layout = np.zeros([n_bctx, n_bctx], dtype=np.bool_)
                for idx in indices[0]:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_bctx):
                    # Local block attention (smaller triangles)
                    row = q_idx // stride
                    layout[q_idx, row * stride:(row + 1) * stride] = 1
                    # Enforce causality
                    layout[q_idx, q_idx + 1:] = 0
            else:
                # Multiple masks, one per head
                layouts = []
                indices = indices[:num_verts]
                for h in range(n_heads):
                    layout = np.zeros([n_bctx, n_bctx], dtype=np.bool_)
                    subindices = indices[h % num_verts]
                    for idx in subindices:
                        layout[:, idx::stride] = 1
                    for q_idx in range(n_bctx):
                        # Local block attention (smaller triangles)
                        row = q_idx // stride
                        layout[q_idx, row * stride:(row + 1) * stride] = 1
                        # Enforce causality
                        layout[q_idx, q_idx + 1:] = 0
                    layouts.append(layout)
                layout = np.array(layouts)
        else:
            for q_idx, k_idx in np.ndindex(n_bctx, n_bctx):
                if k_idx > q_idx:
                    layout[q_idx, k_idx] = 0
                if extra_diagonals and k_idx + extra_diagonals < q_idx:
                    layout[q_idx, k_idx] = 0
                if block_chunks is not None:
                    layout[q_idx, k_idx] = 0
                    offset = q_idx % block_chunks
                    if k_idx + offset >= q_idx and k_idx <= q_idx:
                        layout[q_idx, k_idx] = 1
        return layout

# if __name__ == "__main__":
#     import torch

#     # Example configuration
#     cfg = AttnConfig(
#         d_model=64,       # embedding dimension
#         n_head=4,         # number of attention heads
#         block_size=16,    # sequence length
#         dropout=0.1,
#         bottleneck_dim=32 # only for bottleneck
#     )

#     # Dummy input: batch_size=2, seq_len=16, embedding=d_model
#     x = torch.randn(2, cfg.block_size, cfg.d_model)

#     # Test CausalSelfAttention
#     attn = CausalSelfAttention(cfg)
#     out = attn(x)
#     print("CausalSelfAttention output shape:", out.shape)  # should be [2, 16, 64]

#     # Test CausalBottleneck
#     bottleneck_attn = CausalBottleneck(cfg)
#     out_bottleneck = bottleneck_attn(x)
#     print("CausalBottleneck output shape:", out_bottleneck.shape)  # should also be [2, 16, 64]


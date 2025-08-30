from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

@dataclass
class BlockConfig:
    attnblocksize: int  # number of tokens per block for block-sparse attention
    attn_mode: int      # mode for attn ['fixed' , 'strided' , 'all' , 'local']. research shown fixed is the best for text 
    n_heads: int        # can tuned but later 
    #########################
    # Tune for optimisation
    #########################
    vertsize: int       # size of vertical slices tune for optimisation
    num_verts: int      # num of vertical slices tune for optimisation
    local_attn_ctx: int # num of tokens each query can attned to locally larger , closer to full attention , smaller , more sparse


@dataclass
class SparseConfig:
    d_model: int
    n_ctx: int
    n_head: int
    dropout: float
    blockspace: BlockConfig




class SparseAttention(nn.Module):
    def __init__(self, block_size, cfg: SparseConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.d_model = cfg.d_model
        self.block_size = block_size
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", self.blockspace(cfg.block_size , cfg.blockspace))

    def blockspace(self, block_size , cfg):
        n_bctx = block_size // cfg.attnblocksize
        local_attn_ctx = cfg.local_attn_ctx
        block_size = cfg.block_size
        attn_mode = cfg.attn_mode
        vertsize = cfg.vertsize
        n_heads = cfg.n_heads
        num_verts = cfg.num_verts

        layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
        extra_diagonals = None
        block_chunks = None

        if attn_mode in ['all', 'fixed']:
            pass
        elif attn_mode == 'local':
            assert local_attn_ctx % block_size == 0
            extra_diagonals = local_attn_ctx // block_size
        elif attn_mode == 'strided':
            bT_ctx = cfg.n_ctx // local_attn_ctx
            assert bT_ctx % block_size == 0
            block_chunks = bT_ctx // block_size
        else:
            raise ValueError(f'attn mode {attn_mode} invalid')

        if attn_mode == 'fixed':
            assert n_heads % num_verts == 0
            stride = local_attn_ctx // block_size
            assert vertsize <= stride
            assert stride % vertsize == 0

            indices = [i for i in range(stride - 1, -1, -1)]
            indices = np.array(indices).reshape([-1, vertsize])

            if num_verts == 1:
                layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
                for idx in indices[0]:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_bctx):
                    row = q_idx // stride
                    layout[q_idx, row * stride:(row + 1) * stride] = 1
                    layout[q_idx, q_idx + 1:] = 0
            else:
                layouts = []
                indices = indices[:num_verts]
                for h in range(n_heads):
                    layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
                    subindices = indices[h % num_verts]
                    for idx in subindices:
                        layout[:, idx::stride] = 1
                    for q_idx in range(n_bctx):
                        row = q_idx // stride
                        layout[q_idx, row * stride:(row + 1) * stride] = 1
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




    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
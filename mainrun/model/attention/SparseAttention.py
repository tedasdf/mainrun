# from dataclasses import dataclass


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math

# @dataclass
# class BlockConfig:
#     attnblocksize: int  # number of tokens per block for block-sparse attention
#     attn_mode: int      # mode for attn ['fixed' , 'strided' , 'all' , 'local']. research shown fixed is the best for text 
#     n_heads: int        # can tuned but later 
#     #########################
#     # Tune for optimisation
#     #########################
#     vertsize: int       # size of vertical slices tune for optimisation
#     num_verts: int      # num of vertical slices tune for optimisation
#     local_attn_ctx: int # num of tokens each query can attned to locally larger , closer to full attention , smaller , more sparse


# @dataclass
# class SparseConfig:
#     d_model: int
#     n_ctx: int
#     n_head: int
#     dropout: float
#     blockspace: BlockConfig


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SparseAttention(nn.Module):
    def __init__(self, heads , attn_mode , local_attn_ctx = None , blocksize = 32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize
        self.register_buffer("tril", self.get_attn_mask( blocksize , attn_mode , local_attn_ctx).squeeze(0).squeeze(0))
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)

def split_heads(x, n):
    """
    Split the last dimension (d_model) into [n, head_dim] and transpose to 
    (batch, n, seq_len, head_dim).

    Args:
        x: Tensor of shape [batch, seq_len, d_model]
        n: number of heads

    Returns:
        Tensor of shape [batch, n, seq_len, head_dim]
    """
    B, T, C = x.size()
    head_dim = C // n
    x = x.view(B, T, n, head_dim)       # [B, T, n, head_dim]
    x = x.transpose(1, 2)               # [B, n, T, head_dim]
    return x


def merge_heads(x):
    """
    Merge attention heads back to (batch, seq_len, hidden_dim).
    
    Args:
        x: Tensor of shape (batch, heads, seq_len, head_dim)
    """
    b, h, t, d = x.size()
    x = x.transpose(1, 2).contiguous()   # (batch, seq_len, heads, head_dim)
    x = x.view(b, t, h * d)              # (batch, seq_len, hidden_dim)
    return x


class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize
        self.register_buffer("tril", self.get_attn_mask( blocksize , attn_mode, local_attn_ctx).squeeze(0).squeeze(0))
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)

    def forward(self, x):
        
    def get_attn_mask(self , n, attn_mode, local_attn_ctx=None):
        if attn_mode == 'all':
            b = torch.tril(torch.ones([n, n]))
        elif attn_mode == 'local':
            bandwidth = local_attn_ctx
            ctx = min(n - 1, bandwidth - 1)
            b = torch.tril(torch.ones([n, n]), ctx)
        elif attn_mode == 'strided':
            stride = local_attn_ctx
            x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
            y = torch.transpose(x, 0, 1)
            z = torch.zeros([n, n], dtype=torch.int32)
            q = z + x
            k = z + y
            c1 = q >= k
            c2 = torch.eq(torch.fmod(q - k, stride), 0)
            c3 = torch.logical_and(c1, c2)
            b = c3.float()
        else:
            raise ValueError('Not yet implemented')
        b = torch.reshape(b, [1, 1, n, n])
        return b

    def strided_transpose(self, x, n_ctx, local_attn_ctx, blocksize):
        bT_ctx = n_ctx // local_attn_ctx
        assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
        n, t, embd = x.size()
        x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
        x = torch.transpose(x, 0, 2, 1, 3)
        x = torch.reshape(x, [n, t, embd])
        return x

    def blocksparse_attention_impl(self, q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
        n_ctx = q.size()[1]
        if attn_mode == 'strided':
            q = self.strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
            k = self.strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
            v = self.strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
        n_state = q.size()[-1] // heads
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(n_state), dim=-1)
        a = torch.matmul(w, v)
        if attn_mode == 'strided':
            n, t, embd = a.size()
            bT_ctx = n_ctx // local_attn_ctx
            a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
            a = torch.transpose(a, 0, 2, 1, 3)
            a = torch.reshape(a, [n, t, embd])
        return a


    def attention_impl(self, q, k, v, heads, attn_mode, local_attn_ctx=None):
        q = split_heads(q, heads)
        k = split_heads(k, heads)
        v = split_heads(v, heads)
        n_timesteps = k.size()[2]
        w = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size()[-1])
        w = w.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))  + -1e9 * (1 - mask)
        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, v)
        a = merge_heads(a)
        return a

    def forward(self, q, k, v):
        return self.blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)


# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    heads = 4
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    model = SparseAttention(heads, attn_mode, local_attn_ctx, blocksize)
    output = model(q, k, v)
    print(output[0])



from model.attention.attention import (
    AttnConfig, 
    CausalSelfAttention,
    SparseAttnConfig,
    SparseCausalSelfAttention,
    BottleneckAttnConfig,
    CausalBottleneckAttn)
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from dataclasses import dataclass
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    batch_size: int
    embed_dim: int
    dropout: float
    n_layers: int



@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    d_model: int
    dropout: float
    attn_config : AttnConfig
    output_dim : int
    norm_type: str  # 'pre' or 'post'
    activation_function: str = 'gelu'  # 'relu' or 'gelu'



class MLP(nn.Module):
    def __init__(self, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, 4 * output_dim),
            nn.GELU(),
            nn.Linear(4 * output_dim, output_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, attn_cfg: GPTConfig, output_dim: int, norm_type: str, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        if isinstance(attn_cfg, AttnConfig):
            self.attn = CausalSelfAttention(attn_cfg)
        elif isinstance(attn_cfg, SparseAttnConfig):
             self.attn = SparseCausalSelfAttention(attn_cfg)
        elif isinstance(attn_cfg, BottleneckAttnConfig):
            self.attn = CausalBottleneckAttn(attn_cfg)
        else:
            raise ValueError("Unsupported attention configuration")
        
        self.norm_type = norm_type
        self.mlp  = MLP(output_dim , dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg.attn_config, cfg.output_dim,  cfg.norm_type, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(lambda m: GPT._init_weights(m, cfg.activation_function))
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module, activation_function):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if activation_function == "gelu":
                # He initialization for GELU
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")  # GELU approximates ReLU
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss
    



if __name__ == "__main__":
    cfg = GPTConfig(
        vocab_size=16000,
        block_size=128,
        n_layer=6,
        n_head=8,
        d_model=512,
        dropout=0.1,
        bottleneck_dim=256  # No bottleneck
    )
    model = GPT(cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
    print(f"Trainable params: {trainable_params:,}")
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))  # [batch_size, seq_len]
    logits, loss = model(x, x)
    print(logits.shape, loss)
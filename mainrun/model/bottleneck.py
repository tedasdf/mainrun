from model.gpt import Block
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


class BottleneckGPT(nn.Module):
    def __init__(self, cfg: BottleneckGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([])
        cfg = {
            "d_model": cfg.d_model,
            "n_head": cfg.n_head,
            "dropout": cfg.dropout,
            "bottleneck_size": cfg.bottleneck_size
        }
        for _ in range(cfg.n_layer):
            cfg['d_model'] /= 2
            block_cfg = Block(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                dropout=cfg.dropout,
                bottleneck_dim=cfg.bottleneck_size if cfg.bottleneck_size > 0 else None
            )
            self.blocks.append(block_cfg)

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
        
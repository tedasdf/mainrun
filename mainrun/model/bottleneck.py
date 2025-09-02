from model.gpt import Block , GPTConfig
from model.attention.attention import AttnConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class BottleneckGPTConfig(GPTConfig):
    hidden_layer_list: List[int]
    



class GPUnetT(nn.Module):
    def __init__(self, cfg: BottleneckGPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([])

        for i in range(cfg.n_layer):
            self.blocks.append(
                Block(
                    cfg.attn_config, 
                    cfg.hidden_layer_list[i],  
                    cfg.norm_type, 
                    cfg.dropout
                )
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
    cfg = BottleneckGPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,  # Number of transformer blocks
        n_head=8,
        d_model=512,
        dropout=0.1,
        bottleneck_size=[512, 256, 256, 128, 128, 64]  # Example sizes
    )
    model = GPUnetT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = model(x, x)
    print(logits.shape, loss)

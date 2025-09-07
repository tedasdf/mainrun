from model.gpt import Block , GPTConfig,GPT
from model.attention.attention import AttnConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class UnetGPTConfig(GPTConfig):
    hidden_layer_list: List[int] = None
    



class GPUnetT(GPT):
    def __init__(self, cfg: UnetGPTConfig):
        super().__init__(cfg)
        
        self.blocks = nn.ModuleList([])
        hidden_layer_list =  [cfg.d_model] + cfg.hidden_layer_list + [cfg.d_model]
        
        for i in range(len(hidden_layer_list)-1):
            self.blocks.append(
                Block(
                    cfg.attn_config, 
                    hidden_layer_list[i],  
                    cfg.norm_type, 
                    cfg.dropout,
                    hidden_layer_list[i+1]
                )
            )

        self.apply(lambda m: self._init_weights(m, self.cfg))
        self.head.weight = self.token_emb.weight

    def memory_before_inference(self, dtype=torch.float32):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        total_mem = 0

        # Token embedding
        total_mem += self.token_emb.weight.numel() * elem_size

        # Positional embedding
        total_mem += self.pos_emb.numel() * elem_size

        # Dropout has no persistent parameters

        # Blocks
        for i, block in enumerate(self.blocks):
            if hasattr(block, "memory_before_inference"):
                block_mem = block.memory_before_inference(dtype)
            else:
                block_mem = sum(p.numel() * elem_size for p in block.parameters())
            print(f"Block {i+1} memory: {block_mem / (1024**2):.3f} MB")
            total_mem += block_mem

        # Final LayerNorm
        total_mem += sum(p.numel() * elem_size for p in self.ln_f.parameters())

        # Head
        total_mem += self.head.weight.numel() * elem_size

        print(f"Total GPT memory before inference: {total_mem / (1024**2):.3f} MB")
        return total_mem / (1024**2)  # MB

    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        i = 0 
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss
    

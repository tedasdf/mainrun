
from model.attention.attention import (
    AttnConfig, 
    CausalSelfAttention,
    SparseAttnConfig,
    SparseCausalSelfAttention,
)

import torch
import torch.nn as nn
from torch.nn import functional as F
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
    hidden_layer : int
    attention_layer: str
    norm_type: str  = 'pre' # 'pre' or 'post'
    activation_function: str = 'gelu'  # 'relu' or 'gelu'
    init_method: str = 'xavier'
    


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

    def memory_before_inference(self, dtype=torch.float32):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        l1_weight = self.net[0].weight.numel()
        l1_bias   = self.net[0].bias.numel()
        l2_weight = self.net[2].weight.numel()
        l2_bias   = self.net[2].bias.numel()
        total_params = l1_weight + l1_bias + l2_weight + l2_bias
        return total_params * elem_size / (1024 ** 2)  # MB


class Block(nn.Module):
    def __init__(self, 
                 attn_cfg: AttnConfig, 
                 hidden_layer: int, 
                 norm_type: str, 
                 dropout: float, 
                 output_dim: int = None, 
                 context_length: int= None):
        super().__init__()
        
        if output_dim == None:
            output_dim = hidden_layer
        else:
            attn_cfg.d_model = hidden_layer 
        
        
        self.residual_proj = nn.Linear(hidden_layer, output_dim) if hidden_layer != output_dim else nn.Identity()

        
        if isinstance(attn_cfg, AttnConfig):
            self.attn = CausalSelfAttention(attn_cfg, output_dim)
        elif isinstance(attn_cfg, SparseAttnConfig):
             self.attn = SparseCausalSelfAttention(attn_cfg, output_dim, context_length)
        else:
            raise ValueError("Unsupported attention configuration")
        
        self.norm_type = norm_type
        self.mlp  = MLP( output_dim , dropout)
        
        self.ln1 = nn.LayerNorm(hidden_layer)
        self.ln2 = nn.LayerNorm(output_dim)
    
       
    def forward(self, x):
        res = self.residual_proj(x)
        if self.norm_type == 'pre':
            x = res + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        elif self.norm_type == 'post':
            x = self.ln1(res + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x
    
    def memory_before_inference(self, dtype=torch.float32):
        total_mem = 0

        # Attention memory
        if hasattr(self.attn, "memory_before_inference"):
            total_mem += self.attn.memory_before_inference(dtype)
        else:
            # fallback for generic modules
            total_mem += sum(p.numel() * torch.tensor([], dtype=dtype).element_size() for p in self.attn.parameters())

        # MLP memory
        if hasattr(self.mlp, "memory_before_inference"):
            total_mem += self.mlp.memory_before_inference(dtype)
        else:
            total_mem += sum(p.numel() * torch.tensor([], dtype=dtype).element_size() for p in self.mlp.parameters())

        # Residual projection
        if isinstance(self.residual_proj, nn.Linear):
            total_mem += (self.residual_proj.weight.numel() + self.residual_proj.bias.numel()) * torch.tensor([], dtype=dtype).element_size()

        # LayerNorms
        for ln in [self.ln1, self.ln2]:
            total_mem += sum(p.numel() * torch.tensor([], dtype=dtype).element_size() for p in ln.parameters())

        return total_mem / (1024 ** 2)  # MB


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layer):
            print(f"Initializing Block {i+1}/{cfg.n_layer}")
            self.blocks.append(
                Block(
                    cfg.attn_config,
                    cfg.hidden_layer,
                    cfg.norm_type,
                    cfg.dropout,
                    cfg.hidden_layer,
                    cfg.block_size
                )
            )
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

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


    @staticmethod
    def _init_weights(module, cfg):
        """Initialize weights based on cfg.init_method."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if cfg.init_method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif cfg.init_method == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif cfg.init_method == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif cfg.init_method == "uniform":
                bound = 1.0 / (cfg.d_model ** 0.5)  # Scaled by sqrt(d_model)
                nn.init.uniform_(module.weight, -bound, bound)
            else:
                raise ValueError(f"Unknown init_method: {cfg.init_method}")
            
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
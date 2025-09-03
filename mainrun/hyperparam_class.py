from dataclasses import dataclass

@dataclass
class Hyperparameters:

    seed: int
    epochs: int
    val_frac: float
    num_titles: int
    vocab_size: int
    context_length: int  # Added context_length parameter

    log_file: str
    model_architecture: str 
    
    batch_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    bottleneck_size: int  # Set to 0 for no bottleneck, >0 for bottleneck size
    attention_layer: str

    lr: float
    weight_decay: float
    scheduler: str # none, linear, cosine
    optimizer: str
    evals_per_epoch: float





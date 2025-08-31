from dataclasses import dataclass

@dataclass
class Hyperparameters:
    model_arhitecture: str
    block_size: int
    batch_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    lr: float
    weight_decay: float
    evals_per_epoch: int
    epochs: int
    seed: int
    num_titles: int
    val_frac: float
    log_file: str
    bottleneck_size: int  # Set to 0 for no bottleneck, >0 for bottleneck size
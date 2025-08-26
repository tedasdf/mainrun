from dataclasses import dataclass

@dataclass
class Hyperparameters:
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
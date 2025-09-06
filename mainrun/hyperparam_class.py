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
    lr: float
    weight_decay: float
    scheduler: str # none, linear, cosine
    optimizer: str
    evals_per_epoch: float
    amp_bool: bool





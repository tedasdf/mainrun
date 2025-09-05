from mainrun.hyperparam_class import Hyperparameters
from omegaconf import OmegaConf
# memory_estimator.py
import math
cfg = OmegaConf.load("mainrun\config\model_vallos=1.2.yaml")
hparams = OmegaConf.to_container(cfg.hyperparams, resolve=True)
args = Hyperparameters(**hparams)

print(args.seed)
# ------------------------------
# Hyperparameters
# ------------------------------
vocab_size =  args.vocab_size
context_length =  args.context_length
batch_size =  args.batch_size
d_model = args.d_model
n_layer =  args.n_layer
n_head = args.n_head
bottleneck_size = args.bottleneck_size  # 0 for no bottleneck
dtype_bytes = 4  # float32 = 4 bytes

# ------------------------------
# 1. Parameter memory
# ------------------------------
# Embeddings
token_embedding = vocab_size * d_model * dtype_bytes
pos_embedding = context_length * d_model * dtype_bytes

# Attention parameters (Q,K,V,O)
attention_per_layer = 4 * d_model * d_model * dtype_bytes

# Feedforward / MLP: 2 linear layers (d_model -> 4*d_model -> d_model)
mlp_per_layer = (d_model * 4 * d_model + 4 * d_model * d_model) * dtype_bytes

# Total model parameters
param_memory = token_embedding + pos_embedding + n_layer * (attention_per_layer + mlp_per_layer)

# ------------------------------
# 2. Activations memory
# ------------------------------
# Forward activations per layer
activations_per_layer = batch_size * context_length * d_model * dtype_bytes

# Total activations for all layers
total_activations = activations_per_layer * n_layer

# Assume 2x for attention caches and backprop storage
activation_memory = total_activations * 2

# ------------------------------
# 3. Optimizer memory
# ------------------------------
# Assume SGD with momentum or Adam: ~2x parameters
optimizer_memory = param_memory * 2

# ------------------------------
# Total memory estimate
# ------------------------------
total_memory_bytes = param_memory + activation_memory + optimizer_memory

# Convert to MiB and GiB
total_memory_mib = total_memory_bytes / (1024 ** 2)
total_memory_gib = total_memory_bytes / (1024 ** 3)

print(f"Estimated GPU memory usage:")
print(f"  Parameters: {param_memory / (1024**2):.2f} MiB")
print(f"  Activations: {activation_memory / (1024**2):.2f} MiB")
print(f"  Optimizer states: {optimizer_memory / (1024**2):.2f} MiB")
print(f"  Total estimated memory: {total_memory_mib:.2f} MiB ({total_memory_gib:.2f} GiB)")

✅ Completed
- [x] Add W&B  
- [x] Restructure codebase  
- [x] Put into M3  
- [x] Use Docker to train  

⏳ Next Steps

🔧 Development / Infrastructure  
- [X] Use MDN workstation for training  
- [X] Add GitHub Action → trigger training automatically when uploading  


🤖 Transformer Architectures & Mechanisms  
- [X] Implement and test attention mechanisms:  
  - [X] Vanilla self-attention  
  - [x] Sparse attention  
  - [X] Local attention  

📚 Reading / Research  
- [ ] Read & take notes on:  
  - [ ] https://www.alphaxiv.org/abs/2505.08348  
  - [ ] https://www.mdpi.com/1999-4893/17/2/76  
  - [ ] https://link.springer.com/article/10.1007/s10915-022-01939-z  
  - [ ] https://www.alphaxiv.org/abs/2405.13218  
  - [ ] https://www.alphaxiv.org/abs/2402.19469  
  - [ ] https://www.alphaxiv.org/abs/2409.15046  
  - [ ] https://www.alphaxiv.org/abs/2501.02007  
  - [ ] https://ieeexplore.ieee.org/document/10313200  



test run val_loss 1.27 ::
hyperparameters_configured: seed=1337, epochs=7, val_frac=0.1, num_titles=100000, vocab_size=16000, context_length=256, log_file=./logs/mainrun.log, model_architecture=gpt, batch_size=128, n_layer=6, n_head=8, d_model=256, dropout=0.1, bottleneck_size=256, attention_layer=causal, lr=0.007, weight_decay=0.0, scheduler=cosine, optimizer=adagrad, evals_per_epoch=3
2025-09-04 03:21:50
device_info: device=cuda
2025-09-04 03:21:59
dataset_info: titles_count=90000, epochs=7, batches_per_epoch=33, tokens_per_epoch=1102455, vocab_size=16000
2025-09-04 03:21:59
model_configured: vocab_size=16000, block_size=256, n_layer=6, d_model=256, dropout=0.1, attn_config={'d_model': 256, 'n_head': 8, 'block_size': 256, 'dropout': 0.1}, hidden_layer=256, norm_type=pre, activation_function=gelu, init_method=xavier









hyperparameters_configured: seed=1337, epochs=7, val_frac=0.1, num_titles=100000, vocab_size=16000, context_length=256, log_file=./logs/mainrun.log, model_architecture=gpt, batch_size=128, lr=0.007, weight_decay=0.0, scheduler=cosine, optimizer=adagrad, evals_per_epoch=3 


model_configured: vocab_size=16000, block_size=256, n_layer=6, d_model=256, dropout=0.1, attn_config={'d_model': 256, 'n_head': 8, 'block_size': 256, 'dropout': 0.1, 'intermediate_dim': 64, 'attn_type': 'fixed', 'num_verts': 16, 'local_attn_ctx': 32, 'sparseblocksize': 32, 'vertsize': 64, 'n_bctx': 2}, hidden_layer=256, attention_layer=sparse, norm_type=pre, activation_function=gelu, init_method=xavier 
model_info: parameters_count=7717504 


wandb: attn_configs.sparse.intermediate_dim: 128 
wandb: attn_configs.sparse.n_bctx: 2 
wandb: attn_configs.sparse.n_head: 16 
wandb: attn_configs.sparse.num_verts: 8 
wandb: attn_configs.sparse.sparseblocksize: 64 
wandb: attn_configs.sparse.vertsize: 64 


[ 231/231] validation_step: loss=1.256678 time=48.56s 






hyperparameters_configured: seed=1337, epochs=7, val_frac=0.1, num_titles=100000, vocab_size=16000, context_length=256, log_file=./logs/mainrun.log, model_architecture=gpt, batch_size=128, lr=0.007, weight_decay=0.0, scheduler=cosine, optimizer=adagrad, evals_per_epoch=3 

model_configured: vocab_size=16000, block_size=256, n_layer=6, d_model=256, dropout=0.1, attn_config={'d_model': 256, 'n_head': 8, 'block_size': 256, 'dropout': 0.1, 'intermediate_dim': 64, 'attn_type': 'fixed', 'num_verts': 16, 'local_attn_ctx': 32, 'sparseblocksize': 32, 'vertsize': 256, 'n_bctx': 1}, hidden_layer=256, attention_layer=sparse, norm_type=pre, activation_function=gelu, init_method=xavier 
wandb: attn_configs.sparse.intermediate_dim: 64 
wandb: attn_configs.sparse.n_bctx: 1 
wandb: attn_configs.sparse.n_head: 4 
wandb: attn_configs.sparse.num_verts: 8 
wandb: attn_configs.sparse.sparseblocksize: 64 
wandb: attn_configs.sparse.vertsize: 256 

[ 209/231] validation_step: loss=1.256844 time=43.76s 
[ 220/231] validation_step: loss=1.256694 time=46.08s 
[ 231/231] validation_step: loss=1.256678 time=48.40s 





hyperparameters_configured: seed=1337, epochs=7, val_frac=0.1, num_titles=100000, vocab_size=16000, context_length=256, log_file=./logs/mainrun.log, model_architecture=gpt, batch_size=128, lr=0.007, weight_decay=0.0, scheduler=cosine, optimizer=adagrad, evals_per_epoch=3 


model_configured: vocab_size=16000, block_size=256, n_layer=6, d_model=256, dropout=0.1, attn_config={'d_model': 256, 'n_head': 8, 'block_size': 256, 'dropout': 0.1, 'intermediate_dim': 64, 'attn_type': 'fixed', 'num_verts': 16, 'local_attn_ctx': 32, 'sparseblocksize': 32, 'vertsize': 64, 'n_bctx': 2}, hidden_layer=256, attention_layer=sparse, norm_type=pre, activation_function=gelu, init_method=xavier 


wandb: attn_configs.sparse.intermediate_dim: 128 
wandb: attn_configs.sparse.n_bctx: 2 
wandb: attn_configs.sparse.n_head: 16 
wandb: attn_configs.sparse.num_verts: 8 
wandb: attn_configs.sparse.sparseblocksize: 64 
wandb: attn_configs.sparse.vertsize: 64 



[ 209/231] validation_step: loss=1.256844 time=43.90s 
[ 220/231] validation_step: loss=1.256694 time=46.23s 
[ 231/231] validation_step: loss=1.256678 time=48.56s 

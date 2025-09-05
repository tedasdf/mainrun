✅ Completed
- [x] Add W&B  
- [x] Restructure codebase  
- [x] Put into M3  
- [x] Use Docker to train  

⏳ Next Steps

🔧 Development / Infrastructure  
- [ ] Use MDN workstation for training  
- [ ] Add GitHub Action → trigger training automatically when uploading  

⚡ Model Optimization  
- [ ] Implement quantisation  
- [ ] Implement pruning  

🤖 Transformer Architectures & Mechanisms  
- [ ] Implement Causal Transformers  
- [ ] Compare Encoder-Decoder vs. Decoder-Only  
- [ ] Implement and test attention mechanisms:  
  - [ ] Vanilla self-attention  
  - [ ] Sparse attention  
  - [ ] Local attention  

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
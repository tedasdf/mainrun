#!/bin/bash

# # try out n_layer 
# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 8 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 32



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 6 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 32



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 10 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 32


# # 
# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 8 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 256 \
#     --batch_size 32



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 6 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 32



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 10 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 1024 \
#     --batch_size 32


# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 8 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 16



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 6 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 32



# echo "Running training script with specified hyperparameters..."
# python3 train.py \
#     --block_size 256 \
#     --lr 0.005 \
#     --n_layer 10 \
#     --dropout 0.15 \
#     --weight_decay 0.2 \
#     --d_model 512 \
#     --batch_size 64



# echo "Running training script with specified hyperparameters..."
# python3 train.py --batch_size=128 --dropout=0.1 --lr=0.012 --model_arhitecture=gpt --optimizer=sgd --weight_decay=0.01

wandb agent arc_agi/gpt-from-scratch/${SWEEP_ID}
# python3 train.py --batch_size=128 --context_length=256 --d_model=128 --dropout=0.1 --lr=0.01 --n_layer=8 --optimizer=adamw --weight_decay=0.1


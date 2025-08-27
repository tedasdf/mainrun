#!/bin/bash
pwd 
# for bs in 64 128 256; do
#   for lr in 0.006 0.005 0.004 0.003; do
#     for n_layer in 4 6 8; do
#       for dropout in 0.1 0.2 0.3; do
#         for weight_decay in 0.1 0.01 0.001; do
#           for d_model in 128 256 512; do
#             for batch_size in 16 32 64; do
#               echo "Running with bs=$bs, lr=$lr, n_layer=$n_layer, dropout=$dropout, weight_decay=$weight_decay, d_model=$d_model, batch_size=$batch_size"
#               python3 .\mainrune\train.py \
#                 --block_size $bs \
#                 --lr $lr \
#                 --n_layer $n_layer \
#                 --dropout $dropout \
#                 --weight_decay $weight_decay \
#                 --d_model $d_model \
#                 --batch_size $batch_size
#             done
#           done
#         done
#       done
#     done
#   done
# done

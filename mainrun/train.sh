#!/bin/bash


echo "Running with bs=$bs, lr=$lr, n_layer=$n_layer, dropout=$dropout, weight_decay=$weight_decay, d_model=$d_model, batch_size=$batch_size"
python3 train.py \
    --block_size 256 \
    --lr 0.005 \
    --n_layer 6 \
    --dropout 0.15 \
    --weight_decay 0.1 \
    --d_model 256 \
    --batch_size 32

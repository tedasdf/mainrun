#!/bin/bash

# python3 train.py --sweep --sweep_config "./config/sweep_gpt_sparse.yaml" --orig_yaml "./config/hparams_gpt_sparse.yaml"

# python3 train.py --orig_yaml config/hyperparams_unet.yaml

#  python train.py --sweep --sweep_config "./config/sweep_unet.yaml" --orig_yaml "./config/hyperparams_unet.yaml"

# sweep with unent and causal
# python train.py --sweep --sweep_config "./config/sweep_unet.yaml" --orig_yaml "./config/hyperparams_unet.yaml"


# sweep with gpt and causal didnt work before too much memory ( hopefully its fixed )
# python train.py --sweep --sweep_config "./config/sweep_gpt.yaml" --orig_yaml "./config/hyperparams.yaml"



python3 train.py --orig_yaml "./config/training.yaml"
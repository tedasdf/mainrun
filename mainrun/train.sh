#!/bin/bash

python3 train.py --sweep --sweep_config "config?sweep_gpt_sparse.yaml" --orig_yaml "config/hparams_gpt_sparse.yaml"
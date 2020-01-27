#!/bin/bash
python train_vanilla.py -opt options/train/train_vanilla_cicada_x8.json
python train.py -opt options/train/train_srim_cicada_x8.json
python train.py -opt options/train/train_srim_cicada_fine_tune_x8.json

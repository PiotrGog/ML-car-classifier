#!/bin/bash

source env/bin/activate

python ./scripts/generate_data.py \
    "./resources/pa3_images/train/car" \
    "./resources/pa3_images/train_augmentation/car" \
    8000

python ./scripts/generate_data.py \
    "./resources/pa3_images/train/other" \
    "./resources/pa3_images/train_augmentation/other" \
    8000

deactivate
#!/bin/bash

source env/bin/activate

mkdir -p ./resources/pa3_images/train_augmentation/car/
mkdir -p ./resources/pa3_images/train_augmentation/other/

echo "Start generate cars"
python ./generate_data.py \
    "./resources/pa3_images/train/car" \
    "./resources/pa3_images/train_augmentation/car" \
    16000


echo "Start generate others"
python ./generate_data.py \
    "./resources/pa3_images/train/other" \
    "./resources/pa3_images/train_augmentation/other" \
    16000

echo "Finished"

deactivate
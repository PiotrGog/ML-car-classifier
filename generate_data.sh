#!/bin/bash

source env/bin/activate

echo "Start generate cars"
python ./generate_data.py \
    "./resources/pa3_images/train/car" \
    "./resources/pa3_images/train_augmentation/car" \
    8000


echo "Start generate others"
python ./generate_data.py \
    "./resources/pa3_images/train/other" \
    "./resources/pa3_images/train_augmentation/other" \
    8000

echo "Finished"

deactivate
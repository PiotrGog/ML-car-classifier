#!/bin/bash

source ./env/bin/activate

lrs=("0.01" "0.0001")
lrs_str=("001" "00001")


for i in ${!lrs[@]};
do
    lr_param=${lrs[$i]}
    lr_param_str=${lrs_str[$i]}

    python ./train.py \
        --model simple \
        --train-dir ./resources/pa3_images/train_augmentation/ \
        --val-dir ./resources/pa3_images/validation/ \
        --model-h5 ./model.h5 \
        --model-json ./model.json \
        --epochs 20 \
        --lr $lr_param \
        --batch-size 128 \
        --img-size 64 \
        --save-history "./history"$lr_param_str".hist" \
        --save-plot "./plot"$lr_param_str".png"
        # --show-plot

done
deactivate
#!/bin/bash

source ./env/bin/activate

python ./train.py \
    --model simple \
    --train-dir ./resources/pa3_images/train/ \
    --val-dir ./resources/pa3_images/validation/ \
    --model-h5 ./model.h5 \
    --model-json ./model.json \
    --epochs 20 \
    --lr 0.001 \
    --batch-size 128 \
    --img-size 64 \
    --save-history ./history.hist \
    --save-plot ./plot.png \
    # --show-plot

deactivate
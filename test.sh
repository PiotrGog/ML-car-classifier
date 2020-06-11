#!/bin/bash

source ./env/bin/activate

python ./test.py \
    --model simple \
    --test-dir ./resources/pa3_images/validation/ \
    --model-h5 ./model.h5 \
    --model-json ./model.json \
    --batch-size 128 \
    --img-size 64

deactivate
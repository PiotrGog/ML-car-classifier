#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $DIR/env/bin/activate

python $DIR/Runme.py \
    --model simple \
    --test-dir $PWD \
    --model-h5 $DIR/model.h5 \
    --model-json $DIR/model.json \
    --batch-size 128 \
    --img-size 64

deactivate
#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pretrained_models_dir=$DIR/pretrained_models
source $DIR/env/bin/activate


models_list=("resnet" "vgg16" "simple")
lrs=("0.0001" "0.0001" "0.001")
batches_list=(400 256 128)
epochs_list=(20 20 20)
img_size_list=(244 128 126)

echo $0

i=2

if [ "$1" == "resnet" ]
then
	i=0
elif [ "$1" == "vgg16" ]
then
	i=1
elif [ "$1" == "simple" ]
then
	i=2
fi

python $DIR/Runme.py \
    --model ${models_list[$i]} \
    --test-dir $PWD \
    --model-h5 $pretrained_models_dir/${models_list[$i]}.h5 \
    --model-json $pretrained_models_dir/${models_list[$i]}.json \
    --batch-size ${batches_list[$i]} \
    --img-size ${img_size_list[$i]}

deactivate
#!/bin/bash

source ./env/bin/activate

# resnet 244 400 20 0,0001
# vgg16 128 256 20 0,0001
# simple 128 128 20 0,001

models_list=("resnet" "vgg16" "simple")
lrs=("0.0001" "0.0001" "0.001")
batches_list=(400 256 128)
epochs_list=(20 20 20)
img_size_list=(244 128 126)

pretrained_model_dir="./pretrained_models"

for i in ${!lrs[@]};
do
    lr_param=${lrs[$i]}
    model=${models_list[$i]}
    epochs=${epochs_list[$i]}
    batch=${batches_list[$i]}
    img_size=${img_size_list[$i]}

    python ./train_save_best.py \
        --model $model \
        --train-dir ./resources/pa3_images/train_augmentation/ \
        --val-dir ./resources/pa3_images/validation/ \
        --model-h5 ./$pretrained_model_dir/$model.h5 \
        --model-json ./$pretrained_model_dir/$model.json \
        --epochs $epochs \
        --lr $lr_param \
        --batch-size $batch \
        --img-size $img_size

done
deactivate
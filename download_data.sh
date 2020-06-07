#!/bin/bash

curl http://www.cse.chalmers.se/%7Erichajo/dit866/data/pa3_images.zip --output pa3_images.zip
unzip pa3_images.zip -d ./resources/
mkdir -p ./resources/pa3_images/train_augmentation/car
mkdir -p ./resources/pa3_images/train_augmentation/other
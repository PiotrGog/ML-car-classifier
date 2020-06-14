# ML-car-classifier

## Introduction

Machine learning project based on Keras and Tensorflow to classify objects on pictures. There are three deep neural network architectures implemented:

* CNN implemented from scratch named **simple**
* CNN implemented with transfer learning and fine-tuning from VGG16 net with convolutional layers weight initialized from *imagenet* named **vgg16**
* CNN implemented with transfer learning and fine-tuning from ResNet50 net with convolutional layers weight initialized from *imagenet* named **resnet**

## Pre-requirements

1. Python >= 3.6

2. pip >= 20.1.1

## Installation

### Automatic installation (Linux/MacOS)

Run ```install.sh``` script which creates Python virtual environment and installs packages.

### Manual installation

1. Create Python local virtual environment ([12. Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html))

2. Activate Python local virtual environment

3. Run ```pip install -r ./requirements.txt``` to install packages

4. Deactivate Python local virtual environment

### Getting data

Use ```download_data.sh``` script to download and unpack image dataset. Use ```generate_data.sh``` script to run data augmentation in another directory.

## Running

### Training model

To train one of implemented model use ```./train.py``` script. An example is in ```train.sh``` script. To print all available command line arguments run ```python ./train.py --help```.

### Testing model

To test one of implemented model use ```./test.py``` script. An example is in ```test.sh``` script. To print all available command line arguments run ```python ./test.py --help```.

### Predict car class

Use ```Runme.sh``` script with pretrained models. Go to directory with subdirectories called "car" (or any other directory which has car in name) and "other" (no car word in the name) and run ```Runme.sh``` script. In the result program will show the list of photos classified as car and accuracy.

## Installing packages and saving configuration

In order to install packages activate local Python virtual environment and run command: ```pip install <package_name>```

To save installed packages run command: ```pip freeze > requirements.txt```

## Coding rules

1. **Max line width:** 100

2. **Variable, function, file naming convention:** *snake_case*

## References

1. Project introduction: [http://www.cse.chalmers.se/%7Erichajo/dit866/PA3.html](http://www.cse.chalmers.se/%7Erichajo/dit866/PA3.html)

2. Image dataset: [http://www.cse.chalmers.se/%7Erichajo/dit866/data/pa3_images.zip](http://www.cse.chalmers.se/%7Erichajo/dit866/data/pa3_images.zip)

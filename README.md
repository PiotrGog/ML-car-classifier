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

### Getting pretrained models

Download pretrained models from [Google Drive](https://drive.google.com/file/d/1cjEKk15u5jrjUupMm--vTLPdQLlOS8hV/view?usp=sharing) and unpack models to *pretrained_models* folder or use ```download_models.sh``` script.

### Getting data

Use ```download_data.sh``` script to download and unpack image dataset. Use ```generate_data.sh``` script to run data augmentation in another directory.

## Running

### Predict class using ```Runme.sh``` script (with pretrained models)

Use ```Runme.sh``` script with pretrained models. Go to directory with subdirectories named "car" and "other" and run ```Runme.sh``` script. In the result program will show the list of photos classified as *car* and classification accuracy.

Example directory structure:

```bash
.
├── download_data.sh
├── .........
├── resources
│   └── pa3_images
│       └── test_samples
│           ├── car
│           │   ├── 0000.jpg
│           │   ├── 0001.jpg
│           │   ├── 0002.jpg
│           │   ├── 0003.jpg
│           │   ├── 0004.jpg
│           │   ├── 0005.jpg
│           │   └── 0006.jpg
│           └── other
│               ├── 1000.jpg
│               ├── 1001.jpg
│               ├── 1002.jpg
│               ├── 1003.jpg
│               └── 1004.jpg
├── Runme.py
├── Runme.sh
├── .........
└── train.sh
```

Go to directory ```resources/pa3_images/test_samples/``` and call:

```bash
../../../Runme.sh
```

By default ```Runme.sh``` uses *resnet* model. Call ```Runme.sh``` with model name to use another one, e.g.:

```bash
Runme.sh resnet
```

```bash
Runme.sh vgg16
```

```bash
Runme.sh simple
```

### Training model (preferred)

```train_save_best.py``` script executes given number of epochs saving model with the best validation accuracy value.

Use ```train_local_save_best.sh``` script to train all networks with the best parameters locally.

Use ```ML.ipynb``` ipython script in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) if local computation resources are not available.
In your [Google Drive](https://www.google.com/drive/) create folder named **ML_car** and place there ```ML.ipynb```, then open script with **Google Colaboratory** tool. ```ML.ipynb``` calls ```train_google_save_best.sh``` script.

**Select GPU Hardware accelerator in Runtime>Change runtime type>**.

**Run script with Runtime>Run all**.

### Training model (```train.sh``` script)

To train one of implemented model use ```./train.py``` script. An example is in ```train.sh``` script. To print all available command line arguments run ```python ./train.py --help```.

### Testing model

To test one of implemented model use ```./test.py``` script. An example is in ```test.sh``` script. To print all available command line arguments run ```python ./test.py --help```.

## Installing packages and saving configuration

In order to install packages activate local Python virtual environment and run command: ```pip install <package_name>```

To save installed packages run command: ```pip freeze > requirements.txt```

## Coding rules

1. **Max line width:** 100

2. **Variable, function, file naming convention:** *snake_case*

## References

1. Project introduction: [http://www.cse.chalmers.se/%7Erichajo/dit866/PA3.html](http://www.cse.chalmers.se/%7Erichajo/dit866/PA3.html)

2. Image dataset: [http://www.cse.chalmers.se/%7Erichajo/dit866/data/pa3_images.zip](http://www.cse.chalmers.se/%7Erichajo/dit866/data/pa3_images.zip)

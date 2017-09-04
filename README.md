# SHVN-SRGAN
Image Super Resolution using SRGAN on [Street View House Number](http://ufldl.stanford.edu/housenumbers/) (SHVN) dataset
## Getting Started

The aim is to observe how image Super Resolution improves classification accuracy of images. We have selected the SHVN cropped images dataset to test our hypothesis. We will be training a SRGAN model to super-resolve Low Resolution images, and validate its accuracy with a trained AlexNet model as well.

## Prerequisites
* [Tensorflow 1.2](https://www.tensorflow.org/install/)
* [Street View House Number Cropped Digits (32x32)](http://ufldl.stanford.edu/housenumbers/) (Minimally train and test)
* Cloned copy of this repo

## Train CNN Classifier
Run python script `SHVN_Alexnet` in command line. Use `-h` argument to see list of parsable arguments. Default arg values will be used if arguments not specified. 

The CNN model can be trained to classify images of particular resolution. For the model to be trained on original High Resolution 32x32 images, use `-s 1`. For model to be trained on images downsampled to 16x16 (4x downsampling), use `-s 4`. 
```
limwenyao:~$ python SHVN_Alexnet -s 1 -dir ~/savedir
```
If paths are not specified (saving or dataset path), the python file path will be used. Default model name saved as :
```
train_(ground_truth or bilinear)_x(downscale factor)
```

## Train SRGAN Image Super-Resolver
Run python script `SHVN_SRGAN` in command line. Use `-h` argument to see list of parsable arguments. Default arg values will be used if arguments not specified. 



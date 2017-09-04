# SHVN-SRGAN
Image Super Resolution using [SRGAN](https://arxiv.org/abs/1609.04802) on [Street View House Number](http://ufldl.stanford.edu/housenumbers/) (SHVN) dataset
## Getting Started

The aim is to observe how image Super Resolution improves classification accuracy of images. We have selected the SHVN cropped images dataset to test our hypothesis. We will be training a SRGAN model to super-resolve Low Resolution images, and validate its accuracy with a trained AlexNet model as well.

## Prerequisites
* [Tensorflow 1.2](https://www.tensorflow.org/install/)
* [Street View House Number Cropped Digits (32x32)](http://ufldl.stanford.edu/housenumbers/) (Minimally train and test)
* Cloned copy of this repo

## Naming References
Image scaling

| Scale | Resolution | Interpolation |
| :---: |:----------:| :------------:|
| 1     | 32x32      | ground_truth  |
| 4     | 16x16      | bilinear      |
| 16    | 8x8        | bilinear      |
| 64    | 4x4        | bilinear      |

## Train CNN Classifier
Run python script `SHVN_Alexnet` in command line. Use `-h` argument to see list of parsable arguments. Default arg values will be used if arguments not specified. 

The CNN model can be trained to classify images of particular resolution. For the model to be trained on original High Resolution 32x32 images, use `-s 1`. For model to be trained on images downsampled to 16x16 (4x downsampling), use `-s 4`. 
```
limwenyao:~$ python SHVN_Alexnet -s 1 -dir ~/AlexNet
```
If paths are not specified (saving or dataset path), the python file path will be used. Default model name saved as :
```
train_(Interpolation)_x(Scale)
```

## Train SRGAN Image Super-Resolver
Run python script `SHVN_SRGAN` in command line. Use `-h` argument to see list of parsable arguments. Default arg values will be used if arguments not specified. 

SRGAN trains by taking in HR/LR image pairs. Specify your target LR and HR image scale with `-lr` and `-hr` args. Specify the trained AlexNet classifier model directory `-cd` and .meta file name `-cm`. Specify a new directory to save SRGAN model and sample images produced by SRGAN `-sd`.
```
limwenyao:~$ python SHVN_SRGAN -lr 16 -hr 1 -cd ~/AlexNet -cm train_ground_truth_x1.meta -sd ~/SRGAN
```
In your save directory, a randomly selected batch of train and eval images are saved (HR and LR). At each epoch, the same batch of super-resolved images are saved (LR images fed through generator of SRGAN). The terminal also shows the prediction accuracy of training images (train_pred) and validation images (eval_pred) that were fed through the SRGAN generator followed by AlexNet classifier.

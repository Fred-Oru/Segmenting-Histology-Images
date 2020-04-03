# Glomeruli Segmentation Code

This code is an adaptation of the [Matterport implementation of Mask RCNN](https://github.com/matterport/Mask_RCNN)

The `glomerulus.py` file contains the main part of the code

## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `train` folder minus validation set)
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Resume training a model that you had trained earlier. Make sure you don't change the name of the logs folder containing the last model, since it holds the last epoch number
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Train a new model starting from specific weights file using the full `train` dataset (including validation set)
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Launch detection : it generate masks and .roi files for each image in the test set and saves them in the results folder
```
python3 glomerulus.py detect --dataset=/path/to/dataset --subset=test --weights=<last or /path/to/weights.h5>
```

## Jupyter notebooks
Several Jupyter notebooks are provided:
* 0. prepare_training_data : useful functions to prepare the train dataset
* 1. inspect_glomerulus_data.ipynb : useful functions to check the dataset content and the model parameters
* 2. train_glomerulus : bash command and tensorboard analysis
* 3. inspect_glomerulus_model : after training, useful functions to test, debug, and evaluate the model on the validation set
* 4. detect_glomerulus_test : loads the model once so that we can run detection multiple times
* 5. detect_glomerulus_bash : bash command to run detection on the test folder


## Matterport Implementation details

General explanation of the model and its components :
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

Detailed notes on the matterport code : https://ronjian.github.io/blog/2018/05/16/Understand-Mask-RCNN

Technical details about R-CNN, with enlightening explanations about the Pyramid ROI Align : https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296

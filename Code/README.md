# Glomeruli Segmentation Code

This code is an adaptation of the [Matterport implementation of Mask RCNN](https://github.com/matterport/Mask_RCNN). The `glomerulus.py` file contains the specific code for this project.

## Jupyter notebooks
0. `prepare_training_data` : useful functions to prepare the train dataset
1. `inspect_glomerulus_data` : useful functions to check the dataset content and the model parameters
2. `train_glomerulus` : bash command and tensorboard analysis
3. `inspect_glomerulus_model` : after training, useful functions to test, debug, and evaluate the model on the validation set
4. `detect_glomerulus_test` : loads the model once so that we can run detection multiple times on samples
5. `detect_glomerulus_bash` : bash command to run detection on the test folder


## Matterport Implementation details

General explanation of the model and its components :
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

Detailed notes on the matterport code : https://ronjian.github.io/blog/2018/05/16/Understand-Mask-RCNN

Technical details about R-CNN, with enlightening explanations about the Pyramid ROI Align : https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296

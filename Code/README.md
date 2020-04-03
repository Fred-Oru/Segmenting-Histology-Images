# Glomeruli Segmentation

This code is an adaptation of the [Matterport implementation of Mask RCNN](https://github.com/matterport/Mask_RCNN)

The `glomerulus.py` file contains the main part of the code

## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `train` folder minus validation set)
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Resume training a model that you had trained earlier
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Train a new model starting from specific weights file using the full `train` dataset (including validation set)
```
python3 glomerulus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

## Jupyter notebooks
Two Jupyter notebooks are provided as well: `inspect_glomerulus_data.ipynb` and `inspect_glomerulus_model.ipynb`.
They explore the dataset, run stats on it, and go through the detection process step by step.


### Matterport Implementation details

Explication générale du modèle et de ses composantes :
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

Notes explicatives sur le code : https://ronjian.github.io/blog/2018/05/16/Understand-Mask-RCNN

Détails techniques du RCNN, notamment avec des explications sur le Pyramid ROI Align : https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296

## Other Research

This folder contains the other research papers studied before choosing Mask RCNN. They could be useful in later phases of optimization.


### Segmentation of Glomeruli using DL (2019)

The authors run a CNN with a 300x300 sliding window on 2560x1920 images to predict pixel by pixel the chance of belonging to a glomerulus (this is a simpler model than the RCNN Mask but probably very long to run ...)

They start from 250 2560x1920x3 histology images that they cut into a 300x300x3 patch with a 20 pixel stride to feed a CNN.

Each patch is labelled (NPS) 'healthy or almost healthy glomerulus' or (GS) 'pathological glomerulus' or 'no glomerulus'.

CNN used: Google Inception V3 pretrained on Imagenet, re-trained to recognize one of the three classes => 90-95% accuracy

Train / valid set: they put 70% of the patients in a train set, and 30% in the valid, and *then* they did the cropping. It is important that images of patients are not in both the train and the test.

They did a 4-fold cross validation to properly estimate the model's capacity. They did data augmentation (copies with different white noise).

### Semantic Edge detection

This article could be interesting to pre-process the images by eliminating the colors to keep only the edges before passing in Mask R-CNN

It is an edge detector based on the Casenet model: the principle is to use a Resnet and extract the features at different levels in the network to make a separate convolution and deduce the edges. In the article they go a bit further by training different deep and high layers.

### Segmentation Classification HoVer-Net.pdf

The complexity of the article is due to their specific problem of cell nucleus occlusion: their problem is to separate the edges of the nuclei, even when they are superimposed. That being said, in our case, it is interesting to look at Fig2. NP branch: they manage to make a segmentation mask by using
* a Resnet50
* an upsampling/convolution/dense layer sequence
* skip connections (not specified where)

It might be possible to use this structure, less heavy than an RCNN Mask, to solve our problem.


code : https://github.com/vqdang/hover_net

### Nature Deep Learning Histology

Original Title: Automated acquisition of explainable knowledge from unannotated histopathology images

This article is not directly in line with our problem. They classify histological images as healthy or sick.

What is interesting is that they use an auto-encoder on the high-resolution images to reduce them to 100 digital features, on which they apply classical supervised learning.

Lesson learned: Auto-encoding could be a way to reduce the dimensionality of the images while keeping the important features. Maybe we can force the auto-encoder to keep only the glomerular silhouettes?

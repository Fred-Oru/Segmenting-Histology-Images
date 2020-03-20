Matterport : https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
Stanford Lecture : https://www.youtube.com/watch?v=nDPWywWRIRo
Slides: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf

also look at https://www.youtube.com/watch?v=-lXfsWP7DJ8 from 8:38 on

Image Classification :
* Question : what is the class of this thing in the picture ? (supposed to be alone)
* Answer : image + CNN layers + FC layers + softmax -> class predictions
* Limitations : works only for one thing in the picture
intéressant : les métriques de succès (IoU, mAP) https://www.coursera.org/learn/deep-learning-in-computer-vision/lecture/tvLPq/object-detection-problem

Semantic segmentation :
* Question : what and where are things in the piture ? ie. for each pixel, to what class does it belong ?
* Naive answer : slide a window on the image, feed it to Image classification Network, and tag the center pixel with this class - TOO LONG
* Good answer : **Fully Convolutional Networks for Semantic Segmentation**
  * Idea 1 : turn FC layers into 1x1 CNN layers, so that you have a FCN (Fully convolutional network). Whatever the size of the image, the output will be proportional and compute all window convolutions at once.
  * Idea 2 : the result is a reduced image. So you need to **upsample**, in order to recover the orignal size. One way to do this is kind of reverting the maxpooling of the CNN. But in fact it's easier, and more accurate to learn **transpose convolution layers**. It's a kind of fractional Max Pooling, that upscales the image.
  * Idea 3 : **Feature Pyramid Network** multiple scale upsample. The upsampling might be too coarse if the CNN is very deep make the subsampling important (like X32). To make it finer, we sum upsampling of outputs of the CNN at different stage of the network, like (((C5 upscale 2+ C4) upscale 2 + C3) upscale 2 + C2) upscale 4, instead of C5*32.
See https://www.youtube.com/watch?v=-3ylPH3BCWY 4:40 and 9:00
and original paper about FCN https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
and specifically about FCN : https://arxiv.org/pdf/1612.03144.pdf

Image Classification and localization :
* Question : same as before + where is it located ? (give me a bounding box)
* Answer : same structure, but more outputs : class predictions + (x,y,h,w) of the bounding box. It's like have the same CNN but the FC networks to predict different things. You can train them all in once by using a common loss : softmax for class prediction, MSE for bounding boxes

Mutliple object detection :
* Question : same as before, but there can be several objects => should output as many class+box as there are objects
* naiwe answer : slide a window, and try for different scales. TOO LONG
* Good answer : **Region Proposal**
  * RCNN (2014) : Generate 1000 region of interest with some algorithm. Warp them to the same size. Send them in a CNN detection and localization (adjustement to the ROI) system
  * Fast RCNN (2015) : First send the image in an FCN, then generate the ROI, warp them to the same size "ROI Pooling", send then in a FC layer and output classes and localization
  * Faster RCNN (2015) : Learn the ROI and the rest, all at once. Precisely you train RPN object / no object detection + RPN regress box coordinates + object classes + final box coordinate
   At first, it was generated with signal processing algothim. in 2014 cam the RCNN procedure,

Instance Segmentation :
* Question : what and where, at a pixel level, are things in the image ?
* Answer : **Mask R-CNN** (2017-2018)
  * Faster RCNN + a semantic segmentation on each RPN
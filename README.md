# Glomeruli Segmentation in Histology images

This repository is an Artificial Intelligence project aiming to segment [kidney glomeruli](https://en.wikipedia.org/wiki/Glomerulus_(kidney)) in histology images.   
It makes use of the [Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN).  

The trained model is used since July 2020 by some biologists at University Paris Diderot in Paris. It segments glomerulus at a speed of 2.3 seconds per image instead of 10 minutes for a human biologist. The performance as of August 2020 is **Mask IoU* = 0.894**. The margin of error has no significant effect on the surface measurements that the biologist have to perform.

(** Mask IoU is defined as Intersection over Union between the generated masks and ground truth masks. A 0.88 Mask IoU basically means that the generated mask is 88% accurate)*

![Example of segmented image](DataSamples/segmented_image.png)

# Table of Contents
<!-- TOC depthFrom:2 depthTo:3 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Business Problem](#business-problem)
- [Data specifications](#data-specifications)
- [Available Data](#available-data)
- [Performance Metrics](#performance-metrics)
- [Results Log](#results-log)

<!-- /TOC -->



## Business Problem

Biological researchers study the effects of a pathology on kidney [glomeruli](https://en.wikipedia.org/wiki/Glomerulus_(kidney)). To measure these effects, they analyse histological slices of kidneys where the pathology is revealed by colour marking. To quantify the level of kidney pathology, researchers measure the percentage of the glomerular surface area that is marked, with the help an image management software ([Fiji](http://fiji.sc/) distribution of the open source software [ImageJ](https://en.wikipedia.org/wiki/ImageJ)). Actually they :
* manually delineate the contours of the glomeruli (precisely the basal membrane of Bowman's capsule),
* apply a colour threshold to keep only the marked area in dark brown,
* measure the marked area and the total area of the glomeruli with the software.

The most tedious and time-consuming operation is the manual delineation of the glomeruli. The objective of this project was to train an artificial intelligence to automatically segment  glomeruli in a kidney histological image.

## Data specifications

The histological sections are delivered to the researchers in the form of a very high-resolution digital file, in a proprietary format that can only be viewed with a specific software. For reasons of size and format, it is not possible to make measurements directly on this image. However, it is possible to zoom in on areas of the image and export them in .jpg or .tif format.
In addition, researchers are only interested in certain areas of the image where the section is neat and shows a significant part of the glomeruli. For all these reasons, researchers start by extracting areas of interest from the high-resolution original images.

Example (see Data/sample folder) :
* R22 VEGF 2_15.0xC1.jpg to R22 VEGF 2_15.0xC5.jpg are images extracted from the same histological section of kidney no. R22.
* VEGF is one of the types of markers used. Other cuts are made with kidneys marked with PAS or IgG.
* 15 is the zoom magnification compared to the original high-resolution image. Researchers can use magnifications between 5 and 30.
* The file typically has a resolution of 1920x1018 pixels and weighs approximately 2.5 MB. However, there may be slightly different resolutions.


In ImageJ, researchers open each zoomed image, and delineate the cells using a mouse or a graphics tablet. Each delineation is saved by Fiji as a .roi file (acronym for "Region of interest"). There are usually several ROI files for each zoomed image, and they are all compressed into a .zip file. For example:
* RoiSetR22C1.zip contains seven ROIs corresponding to cells in the R22 image VEGF 2_15.0xC1.jpg
* Each ROI contains (among other things) the coordinates (x,y) of each pixel of the contour delimited by the searcher.

Once the ROIs are generated, the software can calculate the surface area of the glomerulus and the marked area inside the glomerulus from the ROI zip file.

The objective of our AI is to process zoom images of any magnification size and output a ROI zip file containing the boundaries of each glomeruli in the image.

## Available Data

For reasons of confidentiality and storage size, the github repository presents only a sample of the data (Data/sample folder). The full data set is accessible to the project contributors on a [shared google drive](https://drive.google.com/open?id=1rmJG8g-bZpiiZyb6SJd3uqtqJOa-EQ9X).

##### Volumetry:
  * size: 1355 MB
  * volumetry: 603 images with associated .roi files
  * resolution: usually 1920x1018 sometimes 1831x1058
  * three colors: VEGF, PAS, IgG
  * magnification: x15 in the majority, x20 on PSAs

##### Breakdown by type of markers:

  | marker|	% patients|	% images|	% glomeruli|
  |-|-|-|-|
  |	VEGF|37.9%	|36.6%	|41.4%|
  |	PAS	|36.8%	|40.8%	|33.2%|
  |	IgG	|25.3%	|22.6%	|25.3%|


## Performance Metrics

Performance for instance segmentation models are usually measured with [Mean Average Precision](https://medium.com/@yanfengliux/the-confusing-metrics-of-ap-and-map-for-object-detection-3113ba0386ef). However, this metrics is very technical and does not reflect the business perspective.

User interviews revealed that :
* They expect the program to detect all glomeruli
* They don't mind false postive too much (because it's easy to discard them)
* They can accept approx. 10% error on the contour (as they themselves don't draw contours perfectly)

As a consequence, the performance metrics are defined as :
* % of undetected glomeruli
* % of false positive
* Mean IoU (Intersection over Union) between generated masks and ground truth mask


## Results Log

#### V3.0 - August 2020

I trained the model on the previous train set augmented with 29 images that a biologist corrected in the previous run. Along the way, I discovered that the optimal convergence curve was was obtained by training the head of the network for 10 epochs and the full network for 25 epochs.

On the old dataset, V3 is generally performing worse than V2 : there is much more undetected glomeruli which is bad (90 instead of 23 on train set, 7 instead of 2 or 3 on valid and test set). However, it's probably something that could be corrected by fine tuning the detection threshold and on the other side, there wa much less False Positive, which is good.

On the new test set, V3 is **much better** on the new test data :
* both models have the same detection success
*	v3 makes much less False Positives (2 instead of 6 for V2)
* mIou v3 = 0,894 whereas v2 = 0,84

The biologists are very happy with results : the model processed 667 images in 25 minutes (2.3 seconds per image instead of 10 minutes on average for a human) and they just had to discard some false positives (no need to correct any segmentation)

#### V2.3 - June-July 2020
First use of the model in production.  

The model is the same as V2.2, the code was just modified in order to accept .tif format as well as .jpg format for the images. We also compiled a Fiji plugin, [ROIadjust](https://imagej.nih.gov/ij/plugins/roi-adjust.html), so that biologists could correct predicted ROIs graphically.

A biologist ran the model on around 300 pictures and decided he needed to correct only 10% of them. He compared results before and after correction and find that the difference was not significant, which means they can use the model's prediction as they come.

| Image |       |mean	   |median	|#glomerulus|
|-|-|-|-|-|
|C362A	|before |29,11812	   |29,515	|25|
|       |after	|29,6632258	 |29,731	|31|
|C362B	|before	|28,3930333	 |30,242	|30|
|       |after	|28,6028485	 |29,183	|33|
|C363A	|before	|32,7878621	 |31,45	  |29|
|       |after	|33,7888485	 |34,245	|33|


#### V2.2 - June 19th 2020
* Major change : RPN_ANCHOR_SCALES = (64, 128, 256, 512,1024)  
(anchor scale twice to better manage small and large glomerulus)

|                | Train        | Valid       | Test V2.1  |
|----------------|--------------|-------------|------------|
|#images         | 461          | 58          | 55         |
|#glomeruli      | 2936         | 359         | 348        |
|#undetected     | 23 (0.79%)   | 3 (0.84%)   |**2 (0.57%)** |
|#false positive | 362 (12.36%)  | 30 (8.36%)  | 43 (12.36%) |
|**#mean IoU on TP** |0.880    | 0.875      |**0.881**  |
|#mean AP        |0.7348        | 0.7303      | 0.7445     |

Comments:
* much better level of detection (>99% accuracy)
* high number of false positive (but not a problem)
* mIOU is improved and very similar on each set

#### V2.1 - June 13th 2020
* MASK_SHAPE scaled up to (56,56) [makes masks more precise]

|                | Train        | Valid       | Test V2.1  |
|----------------|--------------|-------------|------------|
|#images         | 461          | 58          | 55         |
|#glomeruli      | 2936         | 359         | 348        |
|#undetected     | 54 (1.84%)   | 4 (1.11%)   |**8 (2.3%)** |
|#false positive | 356 (12.13%)  | 29 (8.08%)  | 43 (12.36%) |
|**#mean IoU on TP** |0.863    | 0.858      |**0.867**  |
|#mean AP        |0.736        | 0.730      | 0.745     |

Comments:
* slight improvement on IoU and detection
* high number of false positive

#### V2.0 - June 6th 2020
* Confinement is over, the full dataset is accessible=> 3 times more images, with VGEF, IgG and PAS markers
* Dataset split into 80% train, 10% val, 10% test.
* Split done by patient to avoid data leakage
* Still a lot of undetected

|                | Train        | Valid       | Test V2.0  |
|----------------|--------------|-------------|------------|
|#images         | 461          | 58          | 55         |
|#glomeruli      | 2936         | 359         | 348        |
|#undetected     | 74 (2.52%)   | 6 (1.67%)   |**1 (0.29%)**  |
|#false positive | 243 (8.28%)  | 19 (5.29%)  | 25 (7.18%) |
|**#mean IoU on TP** |0.856    | 0.851      |**0.861**  |
|#mean AP        |0.727        | 0.720      | 0.756     |

NB : "Test V1" = performance of the previous model measured on the new test set

Comments:
* High #undetected in Train set : mostly due to PAS X20. The problem seems to be the low contrast between the glomerulus and the backgournd + the zoom difference (x20 instead of the more common x15). One IgG image is corrupted (rois not aligned with the image)
* Validation : all undetected come from PAS x20
* Test : the only undetected comes form a VGEF - small and blurry glomerulus
* this model is generally imprecise on masks
* mean IoU is not fully reproducible and may oscillate +/- 10% (seemingly due to Gpu randomness)

To do for next version:
* correct or delete the corrupted IgG image
* include contrast enhancement in data preparation
* increase the mask size parameter

#### V1 - April 9th 2020
* Still Confined => same dataset
* Major change in the model : added augmenters scale and contrast

|                | Train       | Valid       | Test |
|----------------|-------------|-------------|---------|
|#images         | 161         | 39          | 15      |
|#glomeruli      | 1161        | 277         | 106     |
|#undetected     |2 (0.17%)    | 2 (0.72%)   | 11 (10.38%)  |
|#false positive |118 (10.16%) | 19 (6.86%)  | 6 (5.66%)   |
|**#mean IoU on TP** |**0.8761**       |**0.8852**     |**0.7256**     |
|#mean AP        |0.7639       | 0.7571      | 0.5909     |

Note : on Test, one image is responsible for 6 undetected

#### V0 - March 23rd 2020
* Confined, limited access to data
* Training dataset is composed of 200 VEGF x15 images.
* Test dataset is composed of 15 IgG and CD68 x15 images -  
 **warning** : labelled by the programmer => not fully accurate
* Model trained for 40 epoch (20 head + 20 full)

|                | Train       | Valid       | Test |
|----------------|-------------|-------------|---------|
|#images         | 161         | 39          | 15 |
|#glomeruli      | 1161        | 277         | 106 |
|#undetected     |1 (0.09%)    | 1 (0.36%)   | 8 (7.55%) |
|#false positive |162 (13.95%) | 35 (12.64%) | 10 (9.43%) |
|**#mean IoU on TP** |**0.6906**       | **0.8349**      | **0.6847** |
|#mean AP        |0.7194       | 0.7070      | 0.5782 |

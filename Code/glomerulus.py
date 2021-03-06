"""
Mask R-CNN
Train on the glomeruli segmentation dataset
Author : Frederic Oru (2020)
inspired from nucleus sample written by Waleed Abdulla
as part of the MaskRCNN Matterport implementation

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier.
    # Make sure you didn't chang the name of the logs folder containing the last model, since it holds the last epoch number
    python3 glomerulus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate masks and .roi files
    python3 glomerulus.py detect --dataset=/path/to/dataset --subset=test --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
# import pickle
import cv2
import shutil
import zipfile

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save detection outputs files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_STARTS_WITH = [
    'R21',
    'R24',
    'R33',
    'R56'
]

NB_TRAIN_IMAGES = 479
NB_VALID_IMAGES = 58
NB_EPOCHS_HEAD = 20
NB_EPOCHS_FULL = 40

############################################################
#  Configurations
############################################################

class GlomerulusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "glomerulus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2
    # A 12GB GPU can typically handle 2 images of 1024x1024px #

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + glomerulus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (NB_TRAIN_IMAGES - NB_VALID_IMAGES) // IMAGES_PER_GPU
    VALIDATION_STEPS = NB_VALID_IMAGES // IMAGES_PER_GPU

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 256, 512,1024)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    ## V1
    ## MEAN_PIXEL = np.array([209.92, 203.35, 204.79])
    ## V2
    MEAN_PIXEL = np.array([190.60, 175.39, 205.24])


    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class GlomerulusInferenceConfig(GlomerulusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class GlomerulusDataset(utils.Dataset):

    def load_glomerulus(self, dataset_dir, subset):
        """Load a subset of the glomeruli dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to be loaded. Either train, val, test_xxx or run_yyy
            * train : all images in the "train" subset except ids starting with VAL_IMAGE_STARTS_WITH
            * valid : images in "train" subset starting with VAL_IMAGE_STARTS_WITH
            * other : all images in the other subset
        The 'run' folder is a plain folder containing images to run detection on
        The 'other' subset are to be organized in the following way (for perfomance tracking):
            * one folder per image (named after the name of the image)
            * one subfolder 'images' containing the image with a .jpg or .tif format
            * one subfolder 'masks' containing them masks with a .png format
            * one subfolder 'rois' containing the rois with a .roi format (not used)
        The 'result' folder will be organized as the "train" folder, with one folder
        per image in the other folder
        """
        # Add classes. We have one class.
        # Naming the dataset glomerulus, and the class glomerulus
        self.add_class("glomerulus", 1, "glomerulus")

        # Which subset?
        # "val": use hard-coded list VAL_IMAGE_STARTS_WITH
        # "train": use data from train minus the hard-coded list VAL_IMAGE_STARTS_WITH
        # else: use the data from the specified sub-directory
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if ((subset_dir == "train") or (subset.startswith("test"))):
            image_ids = next(os.walk(dataset_dir))[1] # one directory per image
            val_ids = [id for id in image_ids for start in VAL_IMAGE_STARTS_WITH if id.startswith(start) ]
            non_val_ids = [id for id in image_ids if id not in val_ids]
            image_ids = val_ids if subset == "val" else non_val_ids
        else: # run_yyy folder
            image_ids = next(os.walk(dataset_dir))[2] # files
            image_ids = [id for id in image_ids
                        if (
                            (id.endswith('.jpg') or id.endswith('.tif'))
                            and not id.startswith('.') # to avoid hidden files
                            )
                        ]
            # warning : here image_ids contain the format extension to keep track of it

        # Add images
        for image_id in image_ids:

            if ((subset_dir == "train") or (subset.startswith("test"))):
                img_dir = os.path.join(dataset_dir, image_id)
                # NB : since V3, training images can have TIF or JPG format, so we have to find the full name of img
                images_dir = os.path.join(img_dir, 'images')
                img_full_name = next(os.walk(images_dir))[2][0]
                img_path = os.path.join(images_dir, img_full_name)
            else:
                img_dir = dataset_dir
                img_path = os.path.join(img_dir, image_id)
                # images can be .jpg or .tif
                image_id = image_id[:-4]

            self.add_image(
                "glomerulus",
                image_id=image_id,
                path=img_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                if (len(m.shape) == 3): # in case the mask is an RGB image
                    m = m[:,:,0] #keep only one channel, as they are all the same
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "glomerulus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Mask to .roi utils
############################################################

def contour_from_mask(mask):
    # cv2.findContours only works on binary image
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0] #there should be only one contour

def roi_from_contour(name, cnt):
    # using roi format in read-roi package
    x = cnt[:,0,0]
    y = cnt[:,0,1]
    dict_data = {
        'type':'freehand',
        'x': list(x),
        'y': list(y),
        'n':len(x),
        'width':0,
        'name':name,
        'position':0
    }
    roi = { name : dict_data }
    return(roi)

def bytes_from_roi(roi):
# inspired from https://gist.github.com/dabasler/c4b885164db5c845cd20688f4a00cab0

    for name in roi:
        n = roi[name]['n'] # number of points
        x = roi[name]['x'] # x coordinates
        y = roi[name]['y'] # y coordinates

    x = [int(xi) for xi in x] # make sure that x is a list of python int type (not numpy.int type)
    y = [int(yi) for yi in y] # make sure that x is a list of python int type (not numpy.int type)

    data=bytearray()       # buffer

    # HEADER 1
    data += "Iout".encode()         # file always starts with 'Iout'
    data += (227).to_bytes(2,'big') # (int16) version number = 227
    data += (7).to_bytes(1,'big')   # (int8) type of roi - 7 means 'freehand'
    data += (0).to_bytes(1,'big')   # (int8) code for subtype - 0 here
    top = min(y)                    # need to convert to python int because numpy int doesn't work with to_bytes
    data += (top).to_bytes(2,'big') # (int16) lower y coordinates
    left = min(x)
    data += (left).to_bytes(2,'big')# (int16) lower x coordinates
    bottom = max(y)
    data += (bottom).to_bytes(2,'big')# (int16) upper y coordinates
    right = max(x)
    data += (right).to_bytes(2,'big') # (int16) upper x coordinates
    data += n.to_bytes(2,'big')     # (int16) total number of coordinates
    for i in range(42):             # unused bytes
        data += b'\x00'
    h2offset_index = len(data)
    data += (0).to_bytes(4,'big')   # (int32) header2 offset - set to 0 for now, will change at the end

    # COORDINATES
    for x_i in x: # (int16) array of x coordinates, with offset
        data += (x_i - left).to_bytes(2,'big')
    for y_i in y: # (int16) array of y coordinates, with offset
        data += (y_i - top).to_bytes(2,'big')

    # HEADER2
    h2offset = len(data)
    for i in range(16):             # header2 start and C,Z,T positions - unused bytes
        data += b'\x00'
    nameoffset_index = len(data)
    data += (0).to_bytes(4,'big')   # (int32) name offset - set to 0 for now, will change at the end
    data += (len(name)).to_bytes(4,'big')  # (int32) name length
    for i in range(40):             # unused bytes
        data += b'\x00'

    # FILE NAME
    nameoffset = len(data)
    for c in name:                  # name of the roi file, encoded every other byte
        data += b'\x00' + c.encode()

    # write offsets retrospectively
    data[h2offset_index:h2offset_index+4]=(h2offset).to_bytes(4,'big')
    data[nameoffset_index:nameoffset_index+4]=(nameoffset).to_bytes(4,'big')

    return data

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = GlomerulusDataset()
    dataset_train.load_glomerulus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GlomerulusDataset()
    dataset_val.load_glomerulus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.GammaContrast((0.5, 2.0),per_channel=True),
        iaa.Affine(scale=(0.5, 2.5))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=NB_EPOCHS_HEAD,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=NB_EPOCHS_FULL,
                augmentation=augmentation,
                layers='all')

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory and saves results in RESULT_DIR
    NB : 'subset' can be a single folder containing images, or a folder of single folders
    The RESULT_DIR folder is organized in the following way :
        * one folder DATE&TIME containing :
            one folder per subfolder in subset, and in this folder :
                * all original images in subset subfolder
                * one IMG_ID_roi.zip for each images in subset subfolder (zip containg detected countours in .roi format)
                * one Segmented_IMG_ID.png for each images in subset subfolder (image superposed with countours)
    """

    assert subset not in ['train','val'], "trying to run detection on train or val set => abort"
    print("Running on {}".format(dataset_dir))

    assert "/" not in subset, "subset must be a folder, not a path. Change dataset path in consequence"

    # Create Timestamped results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    timestamped_dir = "detect_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    timestamped_dir = os.path.join(RESULTS_DIR, timestamped_dir)
    os.makedirs(timestamped_dir)

    # create list of all subfolders of 'subset' (might be subset itself if it's a single folder)
    subfolders = next(os.walk(os.path.join(dataset_dir,subset)))[1]
    if subfolders:
        dataset_dir = os.path.join(dataset_dir,subset)
    else:
        subfolders = [subset]

    for subfolder in subfolders:
        # create subfolder in the result folder
        submit_dir = os.path.join(timestamped_dir, subfolder)
        os.makedirs(submit_dir)
        # Read dataset
        dataset = GlomerulusDataset()
        dataset.load_glomerulus(dataset_dir, subfolder)
        dataset.prepare()
        # Load over images

        for image_id in dataset.image_ids:
            # Load image
            image = dataset.load_image(image_id)
            img_name = dataset.image_info[image_id]["id"]

            # copy the original image
            img_path = dataset.image_info[image_id]["path"]
            img_name = os.path.basename(img_path)
            img_id = img_name[:-4]
            shutil.copyfile(img_path, os.path.join(submit_dir,img_name))

            # Detect objects
            r = model.detect([image], verbose=0)[0]

            # Save image with masks
            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=False, show_mask=False,
                title="Predictions")
            try:
                plt.savefig("{}/Segmented_{}.png".format(submit_dir, img_name ), bbox_inches='tight')
            except:
                pass

            # generate masks and extract rois
            for i in range(r['masks'].shape[2]): # shape = (h)x(w)x(number of masks)
                mask = (r['masks'][:,:,i]*255).astype('uint8')
                cnt = contour_from_mask(mask)
                mask_name = "{}-{}".format(img_id,i+1)
                roi = roi_from_contour(mask_name+'.roi',cnt)
                bytes = bytes_from_roi(roi)
                with open(os.path.join(submit_dir,mask_name+'.roi'), "wb") as file:
                    file.write(bytes)

            # Zip rois and delete original files
            with zipfile.ZipFile(os.path.join(submit_dir,img_id + '_roi.zip'), 'w') as zipObj:
                # Add multiple files to the zip
                for file in next(os.walk(submit_dir))[2]:
                   if file.endswith('.roi'):
                       zipObj.write(os.path.join(submit_dir,file),arcname=file)
                       os.remove(os.path.join(submit_dir,file))


###########################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for glomeruli counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GlomerulusConfig()
    else:
        config = GlomerulusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

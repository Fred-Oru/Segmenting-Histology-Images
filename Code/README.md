# Glomeruli Segmentation

The `glomerulus.py` file contains the main parts of the code

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

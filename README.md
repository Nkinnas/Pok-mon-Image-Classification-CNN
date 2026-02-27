# Pokemon Image Classification CNN

A convolutional neural network that classifies images of first-generation Pokemon (151 Pokemon).

## Setup

1. Download or clone this repository

2. Get the dataset from [Kaggle - Pokemon Images First Generation (17000 files)](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files/code/data)

3. Unzip the images and place them in the `Pokemon_images/` folder (it is empty by default)

4. Adjust the paths in `scripts/train.py`, `scripts/make_splits.py`, and `scripts/predict.py` to match your local setup

## Usage

1. **Run `make_splits.py`** - splits the dataset into train/val/test sets

2. **Run `train.py`** - trains the CNN model

3. **Run `predict.py`** - classifies a Pokemon image. Before running, place the image you want to classify inside the `Image_to_predict/` folder. Since this model is trained on first-generation Pokemon only, the image should be of one of the 151 original Pokemon. Also the image inside the folder must be name "predict_image.jpg"

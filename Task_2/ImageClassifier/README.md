# ImageClassifier

This project is an image classification tool built with Python. It uses machine learning algorithms to classify images into predefined categories.

It utilizes the ```ResNet50``` pretrained model and Animal 10 Kaggle dataset, available here: https://www.kaggle.com/datasets/alessiocorrado99/animals10/data

This model was trained on A100 GPU in Google Colab.

The resulting model is saved into the ```image_classifier.weights.h5``` file.

## Installation

1. To install the required packages use a virtual environment and run:
    ```sh
    pip install -r requirements.txt
    ```
2. Download the Animal 10 dataset from Kaggle and put the ```raw-img``` folder from it into this directory

## Usage

1. Preprocess the data:
    ```sh
    python CreateImageClassifier.py
    ```

## Additional comments

This folder also contains a jypiter notebook ```Task2_1.ipynb``` made in Google Colab with the analysis of dataset and results as well as description of bits of the code from this folder.


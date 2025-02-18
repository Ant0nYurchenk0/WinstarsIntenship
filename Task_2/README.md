# Task 2. Named entity recognition + image classification

Winstars internship Task 2

## Description

In this task, you will work on building your ML pipeline that consists of 2 models responsible for
totally different tasks. The main goal is to understand what the user is asking (NLP) and check if
he is correct or not (Computer Vision).
You will need to:
* find or collect an animal classification/detection dataset that contains at least 10
classes of animals.
* train NER model for extracting animal titles from the text. Please use some
transformer-based model (not LLM).
* Train the animal classification model on your dataset.
* Build a pipeline that takes as inputs the text message and the image.

In general, the flow should be the following:
1. The user provides a text similar to “There is a cow in the picture.” and an image that
contains any animal.
2. Your pipeline should decide if it is true or not and provide a boolean value as the output.
You should take care that the text input will not be the same as in the example, and the
user can ask it in a different way

## Installation

1. To install the required packages use a virtual environment and run:
    ```sh
    pip install -r requirements.txt
    ```
2. This project relies on models from ```ImageClassifier``` and ```NamedEntityRecognizer``` so it is necessary to generate ```image_classifier.h5``` and ```ner.h5``` models by running respective projects.

## Usage

1. Run the python file with parameters:
    ```sh
    python main.py <path-to-image> <sentence>
    ```

## Structure of the project
This project is organized as follows:
```

Task_2/
├── main.py
|
├── ModelPipeline.py # Class that combines image classifier and named entity recognizer
|
└── ImageClassifier/
|   ├── AnimalImageClassifier.py
|   ├── CreateImageClassifier.py
|   ├── DataPreProcessor.py
|   ├── ModelImplementation.py
|   ├── Translate.py
|   └── image_classifier.h5
|
└── NamedEntityRecognizer
|   ├── AnimalNameEntityRecognizer.py
|   ├── CreateNamedEnityRecognizer.py
|   ├── DataGenerator.py
|   ├── DataPreProcessor.py
|   └── ner.h5
|
└── README.md
```

## Additional comments

There might be a mismatch in versions of Keras for image classifier and named entity recognizer. This is because ```TFBertModel``` relies on the version of keras used in the transformers library, which is outdated at this point.

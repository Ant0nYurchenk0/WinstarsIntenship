# Task 1. Image classification + OOP

Winstars intenship Task 1

## Description

In this task, you need to use a publicly available simple MNIST dataset and build 3 classification models around it. It should be the following models:

1. Random Forest;
2. Feed-Forward Neural Network;
3. Convolutional Neural Network;
Each model should be a separate class that implements  MnistClassifierInterface  with 2 abstract methods - train and predict. Finally, each of your three models should be hidden under another  MnistClassifier  class.  MnistClassifer  takes an algorithm as an input parameter. Possible values for the algorithm are:  cnn ,  rf , and  nn  for the three models described above. The solution should contain:

* Interface for models called  MnistClassifierInterface .
* 3 classes (1 for each model) that implement  MnistClassifierInterface .
* MnistClassifier , which takes as an input parameter the name of the algorithm and provides predictions with exactly the same structure (inputs and outputs) not depending on the selected algorithm.

## Installation

Better use a virtual environment for running this program as there are many version-specific dependencies to be installed.

To install them use:

```bash
pip install -r requirements.txt
```

## Usage

To run the program use:
```python
python main.py
```

## Structure of the project

This project is organized as follows:

```
│
│
├── Classifiers/                        # Directory containing the model classes
│   ├── RFClassifier.py                 # Random Forest model implementation
│   ├── FFNNClassifier.py               # Feed-Forward Neural Network implementation
│   └── CNNClassifier.py                # Convolutional Neural Network implementation
│   └── MnistClassifierInterface.py     # MnistClassifierInterface definition
│   └── MnistClassifier.py              # Wrapper class for all classifiers
|
├── Helpers/                            # Directory containing helper classes
│   └── bcolors.py                      # Enum with color codes for terminal
|
├── main.py                             # Main script to run the program
│
├── requirements.txt                    # File listing the dependencies
│
└── README.md                           # Project documentation
```

## Additional comments

There is also a jupyter notebook ```Task1.ipynb``` made in Google Colab with the results and descreptions of executed bits of the code from this project.

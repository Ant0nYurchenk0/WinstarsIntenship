import numpy as np
from typing import Tuple
from Classifiers.RFClassifier import RFClassifier
from Classifiers.FFNNClassifier import FFNNClassifier
from Classifiers.CNNClassifier import CNNClassifier


class MnistClassifier:
    def __init__(self, algorithm: str):
        match (algorithm):
            case "rf":
                self._classifier = RFClassifier()
            case "nn":
                self._classifier = FFNNClassifier()
            case "cnn":
                self._classifier = CNNClassifier()
            case _:
                raise ValueError("Invalid algorithm")
        self._predictions = np.ndarray(0)
        self._accuracy = 0

    def classify(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        self._classifier.train(train_images, train_labels)
        self._predictions = self._classifier.predict(test_images)
        self._accuracy = self._classifier.evaluate_accuracy(test_labels)
        return (self._predictions, self._accuracy)

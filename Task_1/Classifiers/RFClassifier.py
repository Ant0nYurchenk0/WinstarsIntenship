import numpy as np
from Classifiers.MnistClassifierInterface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RFClassifier(MnistClassifierInterface):
    def __init__(self):
        self._predictor = RandomForestClassifier()
        super()

    def train(self, train_images: np.ndarray, train_labels: np.ndarray) -> None:
        self._train_images = [image.flatten() for image in train_images]
        self._train_labels = train_labels
        self._predictor.fit(self._train_images, self._train_labels)

    def predict(self, test_images: np.ndarray) -> np.ndarray:
        self._test_images = [image.flatten() for image in test_images]
        self._predictions = self._predictor.predict(self._test_images)
        return self._predictions

    def evaluate_accuracy(self, test_labels: np.ndarray) -> float:
        self._test_labels = test_labels
        return accuracy_score(self._test_labels, self._predictions)

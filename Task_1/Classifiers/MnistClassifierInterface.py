import numpy as np
from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    def __init__(self):
        self._train_images = np.ndarray(0)
        self._train_labels = np.ndarray(0)
        self._test_images = np.ndarray(0)
        self._test_labels = np.ndarray(0)
        self._predictions = np.ndarray(0)

    @abstractmethod
    def train(self, train_images: np.ndarray, train_labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, test_images: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_accuracy(self, test_labels: np.ndarray) -> float:
        pass

import numpy as np
from Classifiers.MnistClassifierInterface import MnistClassifierInterface
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # type: ignore


class FFNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.epochs = 5
        self._predictor = Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(128, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )
        self._predictor.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()],
        )
        super()

    def train(self, train_images: np.ndarray, train_labels: np.ndarray) -> None:
        self._train_images = train_images
        self._train_labels = train_labels
        self._predictor.fit(
            self._train_images, self._train_labels, epochs=self.epochs, verbose=0
        )

    def predict(self, test_images: np.ndarray) -> np.ndarray:
        self._test_images = test_images
        return self._predictor.predict(self._test_images)

    def evaluate_accuracy(self, test_labels: np.ndarray) -> float:
        self._test_labels = test_labels
        _, test_acc = self._predictor.evaluate(self._test_images, self._test_labels)
        return test_acc

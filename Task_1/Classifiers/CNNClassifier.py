from Classifiers.FFNNClassifier import FFNNClassifier
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # type: ignore


class CNNClassifier(FFNNClassifier):
    def __init__(self):
        self.epochs = 5
        self._predictor = Sequential(
            [
                Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(2, 2),
                Flatten(),
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

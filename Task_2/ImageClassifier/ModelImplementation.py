import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


class ModelImplementation:
    def __init__(self, num_classes: int, train_epochs: int, input_tensor: tf.Tensor):
        self._data_preprocessed = False
        self._model_compiled = False
        self._model_trained = False
        self.train_epochs = train_epochs

        base = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,  # Input(shape=(128, 128, 3))
        )

        for layer in base.layers:
            if not layer.name.startswith("conv5_"):
                layer.trainable = False

        model = base.output
        model = Flatten(name="flatten")(model)
        model = Dense(1024, activation="relu")(model)
        model = Dropout(0.5)(model)
        model = Dense(num_classes, activation="softmax")(model)

        self.model = Model(inputs=base.input, outputs=model)

        self.model.summary()

    def preprocess_data(
        self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset
    ) -> None:
        def preprocess(image, label):
            image = preprocess_input(image)
            return image, label

        self.train_ds = train_ds.map(preprocess)
        self.val_ds = val_ds.map(preprocess)

        self._data_preprocessed = True

    def compile_model(self) -> None:
        self.model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()],
        )

        self._model_compiled = True

    def train_model(self) -> tf.keras.callbacks.History:
        assert self._data_preprocessed, "Data not preprocessed"
        assert self._model_compiled, "Model not compiled"

        self._history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=self.train_epochs
        )
        self._model_trained = True
        return self._history

    def plot_accuracy(self) -> None:
        """Plots the training and validation loss and accuracy from a history object"""

        acc = self._history.history["sparse_categorical_accuracy"]
        val_acc = self._history.history["val_sparse_categorical_accuracy"]
        loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]

        epochs = range(len(acc))

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, acc, "bo", label="Training accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

        plt.show()

    def predict(self, image: tf.Tensor) -> str:
        assert self._model_trained, "Model not trained"
        assert (
            image.shape
            == self.model.get_config()["layers"][0]["config"]["batch_shape"][1:]
        ), "Image shape not correct"
        img_batch = np.expand_dims(image, axis=0)

        predictions = self.model.predict(img_batch)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def save(self, path: str) -> None:
        save_path = os.path.join(path, "image_classifier.h5")
        self.model.save(save_path)

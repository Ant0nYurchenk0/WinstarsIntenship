import math
import tensorflow as tf
import matplotlib.pyplot as plt  # type: ignore
from typing import Tuple
from DataPreProcessor import DataPreProcessor
from ModelImplementation import ModelImplementation
from tensorflow.keras.layers import Input  # type: ignore


class AnimalImageClassifier:
    def __init__(self):
        self._data_preprocessor = None
        self._model = None
        self._train_ds = None
        self._val_ds = None
        self._data_loaded = False
        self._model_trained = False

    def load_data(
        self, data_dir: str, translate: dict[str, str], visualize: bool = False
    ) -> Tuple[tf.data.Dataset, dict[str, int]]:
        self._data_preprocessor = DataPreProcessor(data_dir)
        self._train_ds, self._val_ds = self._data_preprocessor.get_train_test_split(
            validation_split=0.2, image_size=(128, 128), seed=42, batch_size=32
        )
        if translate:
            self._data_preprocessor.add_translation(translate)

        data_sample = self._data_preprocessor.get_data_sample(
            visualize=visualize, num_batches=1
        )
        class_distrib = self._data_preprocessor.get_class_distribution(
            visualize=visualize
        )
        self._data_loaded = True
        return data_sample, class_distrib

    def train_model(
        self, train_epochs: int, plot_history: bool = False
    ) -> tf.keras.callbacks.History:
        assert self._data_loaded, "Data not loaded"
        self._model = ModelImplementation(
            num_classes=len(self._data_preprocessor.get_class_names()),
            train_epochs=train_epochs,
            input_tensor=Input(shape=(128, 128, 3)),
        )
        self._model.preprocess_data(self._train_ds, self._val_ds)
        self._model.compile_model()
        history = self._model.train_model()
        if plot_history:
            self._model.plot_accuracy()

        self._model_trained = True
        return history

    def save_model(self, model_path: str) -> None:
        assert self._model_trained, "Model not trained"
        self._model.save(model_path)

    def visualize_classification(
        self, num_batches: int, visualize_num: int = 9
    ) -> None:
        plt.figure(figsize=(10, 10))

        for images, _ in self._val_ds.take(num_batches):
            for i in range(9):
                height = int(math.sqrt(visualize_num))
                ax = plt.subplot(height, visualize_num // height, i + 1)
                img = images[i].numpy().astype("uint8")
                plt.imshow(img)
                plt.title(self._model.predict(img))
                plt.axis("off")

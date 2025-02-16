import math
import tensorflow as tf
import matplotlib.pyplot as plt  # type: ignore
from typing import Tuple


class DataPreProcessor:
    def __init__(self, data_dir: str):
        self._data_dir = data_dir

    def get_train_test_split(
        self,
        validation_split: float,
        image_size: Tuple[int, int],
        seed: int,
        batch_size: int,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        assert 0 < validation_split < 1, "Validation split must be between 0 and 1"
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self._data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self._data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )
        return self.train_ds, self.val_ds

    def add_translation(self, translation_dict: dict) -> None:
        self._translation_dict = translation_dict

    def get_class_names(self, translate: bool = False) -> list[str]:
        if translate and self._translation_dict:
            return [self._translation_dict[name] for name in self.train_ds.class_names]
        return self.train_ds.class_names

    def get_data_sample(
        self, visualize: bool, num_batches: int, visualize_num: int = 9
    ) -> tf.data.Dataset:
        assert self.train_ds, "Train dataset not found"
        assert self.val_ds, "Validation dataset not found"

        image_sample = self.train_ds.take(num_batches)

        if visualize:

            class_names = self.train_ds.class_names
            plt.figure(figsize=(10, 10))

            for images, labels in image_sample:
                for i in range(visualize_num):
                    height = int(math.sqrt(visualize_num))
                    ax = plt.subplot(height, visualize_num // height, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    if self._translation_dict:
                        plt.title(
                            self._translation_dict[class_names[labels[i].numpy()]]
                        )
                    else:
                        plt.title(class_names[labels[i].numpy()])
                    plt.axis("off")

        return image_sample

    def get_class_distribution(self, visualize: bool) -> dict[str, int]:
        assert self.train_ds, "Train dataset not found"

        class_names = self.train_ds.class_names

        label_counts = {class_name: 0 for class_name in class_names}

        for _, labels in self.train_ds.unbatch():
            label_counts[class_names[int(labels.numpy())]] += 1

        if visualize:
            plt.figure(figsize=(8, 5))
            if self._translation_dict:
                plt.bar(
                    [self._translation_dict[name] for name in label_counts.keys()],
                    label_counts.values(),
                )
            else:
                plt.bar(
                    label_counts.keys(),
                    label_counts.values(),
                )
            plt.xlabel("Class")
            plt.ylabel("Number of Images")
            plt.title("Class Distribution in Training Data")
            plt.xticks(rotation=45)
            plt.show()
        return label_counts

import os
import tensorflow as tf
from transformers import TFBertModel  # type: ignore


class AnimalNameEntityRecognizer:
    def __init__(self, num_labels: int = 2, max_len: int = 64):
        bert_model = TFBertModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased", num_labels=num_labels
        )
        self._max_len = max_len
        self._model = self._create_model(bert_model)
        self._model_trained = False

    def train(
        self,
        input_ids,
        attention_mask,
        train_tag,
        val_input_ids,
        val_attention_mask,
        test_tag,
        epochs: int = 1,
    ) -> tf.keras.callbacks.History:
        self._history = self._model.fit(
            [input_ids, attention_mask],
            train_tag,
            validation_data=([val_input_ids, val_attention_mask], test_tag),
            epochs=epochs,
            batch_size=32,
        )
        self._model_trained = True
        return self._history

    def save(self, filepath: str) -> None:
        assert self._model_trained, "Model has not been trained yet."
        path = os.path.join(filepath, "ner.h5")
        self._model.save(path)

    def _create_model(self, bert_model: TFBertModel) -> tf.keras.Model:
        def custom_loss(y_true, y_pred):
            # Compute the sparse categorical crossentropy loss per token.
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            # For tokens where y_true is 0 ("B-ANIMAL"), multiply the loss by a factor (e.g. 100)
            animal_weight = tf.where(tf.equal(y_true, 0), 100.0, 1.0)
            weighted_loss = loss * animal_weight
            return tf.reduce_mean(weighted_loss)

        input_ids = tf.keras.Input(shape=(self._max_len,), dtype="int32")
        attention_masks = tf.keras.Input(shape=(self._max_len,), dtype="int32")
        bert_output = bert_model(
            input_ids, attention_mask=attention_masks, return_dict=True
        )
        embedding = tf.keras.layers.Dropout(0.3)(bert_output[0])
        output = tf.keras.layers.Dense(2, activation="softmax")(embedding)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks], outputs=[output]
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(lr=0.00001),
            loss=custom_loss,
            metrics=["accuracy"],
        )
        return model

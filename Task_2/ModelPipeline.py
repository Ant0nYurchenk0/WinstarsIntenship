import numpy as np
import tensorflow as tf
import transformers


class ModelPipeline:
    def __init__(self, image_classifier_path: str, ner_path: str):
        self._image_classifier = tf.keras.models.load_model(image_classifier_path)
        self._ner = tf.keras.models.load_model(
            ner_path, custom_objects={"TFBertModel": transformers.TFBertModel}
        )

    def predict_image(self, image: tf.Tensor) -> int:
        resized_img = tf.image.resize(image, [128, 128])
        img_batch = np.expand_dims(resized_img, axis=0)

        predictions = self._image_classifier.predict(img_batch)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def get_animal_token(self, sentence: str, max_len: int) -> str:
        tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_len,
            is_split_into_words=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].reshape(1, max_len)
        attention_mask = encoded["attention_mask"].reshape(1, max_len)
        predictions = self._ner.predict([input_ids, attention_mask])
        pred_with_pad = np.argmax(predictions, axis=-1)
        return sentence(list(x == 0 for x in pred_with_pad).index(True))

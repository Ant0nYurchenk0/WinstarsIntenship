import numpy as np
import tensorflow as tf
import torch
import transformers
from transformers import BertForTokenClassification


class ModelPipeline:
    def __init__(self, image_classifier_path: str, ner_path: str, num_labels: int = 3):
        self._image_classifier = tf.keras.models.load_model(image_classifier_path)

        # Load PyTorch NER model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ner = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self._ner.load_state_dict(torch.load(ner_path, map_location=self._device, weights_only=True))
        self._ner.to(self._device)
        self._ner.eval()

    def predict_image(self, image: tf.Tensor) -> int:
        resized_img = tf.image.resize(image, [128, 128])
        img_batch = np.expand_dims(resized_img, axis=0)

        predictions = self._image_classifier.predict(img_batch)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def get_animal_token(self, sentence: str, max_len: int) -> str:
        tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")

        # Split sentence into words for tokenization
        words = sentence.split()

        encoded = tokenizer(
            words,
            add_special_tokens=True,
            max_length=max_len,
            is_split_into_words=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._ner(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        predictions_np = predictions.cpu().numpy()[0]
        word_ids = encoded.word_ids(batch_index=0)

        # Find the first word that is predicted as B-ANIMAL (class 0)
        for idx, (pred, word_id) in enumerate(zip(predictions_np, word_ids)):
            if pred == 0 and word_id is not None:
                return words[word_id]

        return ""

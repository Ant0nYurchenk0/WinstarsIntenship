import numpy as np
import pandas as pd
from typing import Tuple
from sklearn import preprocessing
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

MAX_LEN = 64

class DataPreprocessor:
    def __init__(self, max_len: int = 64):
        self._max_len = max_len
        self._tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self._data_loaded = False
        self._enc_label = None
        self._animal_id = None
        self._o_id = None
        self._pad_id = None

    def load_data(self, filepath: str) -> None:
        df = pd.read_csv(filepath, encoding="latin-1")
        df["Sentence Number"] = df["Sentence Number"].ffill()

        self._enc_label = preprocessing.LabelEncoder()

        df["Label"] = self._enc_label.fit_transform(df["Label"])

        self._sentences = df.groupby("Sentence Number")["Word"].apply(list).values
        self._tag = df.groupby("Sentence Number")["Label"].apply(list).values

        self._animal_id = self._enc_label.transform(["B-ANIMAL"])[0]
        self._o_id = self._enc_label.transform(["O"])[0]
        self._pad_id = max(self._animal_id, self._o_id) + 1

        self._data_loaded = True

    def get_train_test(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._data_loaded, "Data has not been loaded yet."
        X_train, X_test, y_train, y_test = train_test_split(
            self._sentences, self._tag, random_state=42, test_size=0.1
        )
        input_ids, attention_mask, labels = self._tokenize(X_train, y_train)
        val_input_ids, val_attention_mask, val_labels = self._tokenize(X_test, y_test)
        return (
            input_ids,
            attention_mask,
            labels,
            val_input_ids,
            val_attention_mask,
            val_labels,
        )

    def get_label_encoder(self):
        return self._enc_label

    def get_pad_id(self):
        return self._pad_id

    def get_animal_id(self):
        return self._animal_id

    def get_o_id(self):
        return self._o_id

    def _tokenize(self, sentences, tags, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        n = len(sentences)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_sents = list(sentences[start:end])
            batch_tags = tags[start:end]

            encoded = self._tokenizer(
                batch_sents,
                is_split_into_words=True,
                add_special_tokens=True,
                max_length=self._max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )

            batch_input_ids = encoded["input_ids"]
            batch_attention = encoded["attention_mask"]

            batch_labels = []
            for i, word_labels in enumerate(batch_tags):
                word_ids = encoded.word_ids(batch_index=i)

                token_labels = []
                for wid in word_ids:
                    if wid is None:
                        token_labels.append(self._pad_id)
                    else:
                        token_labels.append(word_labels[wid])

                batch_labels.append(token_labels)

            all_input_ids.append(np.array(batch_input_ids))
            all_attention_masks.append(np.array(batch_attention))
            all_labels.append(np.array(batch_labels))

        input_ids = np.concatenate(all_input_ids, axis=0)
        attention_masks = np.concatenate(all_attention_masks, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return input_ids, attention_masks, labels

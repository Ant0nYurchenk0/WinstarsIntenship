import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn import preprocessing
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, max_len: int = 64):
        self._max_len = max_len
        self._tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self._data_loaded = False

    def load_data(self, filepath: str) -> None:
        df = pd.read_csv(filepath, encoding="latin-1")

        df.loc[:, "Sentence Number"] = df["Sentence Number"].fillna(method="ffill")

        self._enc_label = preprocessing.LabelEncoder()

        df.loc[:, "Label"] = self._enc_label.fit_transform(df["Label"])

        self._sentences = df.groupby("Sentence Number")["Word"].apply(list).values
        self._tag = df.groupby("Sentence Number")["Label"].apply(list).values
        self._data_loaded = True

    def get_train_test(
        self,
    ) -> Tuple[np.vstack, np.vstack, np.array, np.vstack, np.vstack, np.array]:
        assert self._data_loaded, "Data has not been loaded yet."
        X_train, X_test, y_train, y_test = train_test_split(
            self._sentences, self._tag, random_state=42, test_size=0.1
        )
        input_ids, attention_mask = self._tokenize(X_train)
        val_input_ids, val_attention_mask = self._tokenize(X_test)
        train_tag = self._padding(y_train)
        test_tag = self._padding(y_test)
        return (
            input_ids,
            attention_mask,
            train_tag,
            val_input_ids,
            val_attention_mask,
            test_tag,
        )

    def _padding(self, data: np.array) -> np.array:
        tags = list()
        for i in range(len(data)):
            tags.append(np.array(data[i] + [1] * (self._max_len - len(data[i]))))
        return np.array(tags)

    def _tokenize(self, data: np.array) -> Tuple[np.vstack, np.vstack]:
        input_ids = list()
        attention_mask = list()
        for i in tqdm(range(len(data))):
            encoded = self._tokenizer.encode_plus(
                data[i],
                add_special_tokens=True,
                max_length=self._max_len,
                is_split_into_words=True,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

            input_ids.append(encoded["input_ids"])
            attention_mask.append(encoded["attention_mask"])
        return np.vstack(input_ids), np.vstack(attention_mask)

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast


class AnimalNameEntityRecognizer:
    def __init__(self, num_labels: int = 3, max_len: int = 64):
        self._max_len = max_len
        self._num_labels = num_labels
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self._model.to(self._device)
        self._model_trained = False
        self._tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def train(
        self,
        input_ids,
        attention_mask,
        labels,
        val_input_ids,
        val_attention_mask,
        val_labels,
        pad_id,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
    ):
        # Convert to tensors
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)

        val_input_ids_t = torch.tensor(val_input_ids, dtype=torch.long)
        val_attention_mask_t = torch.tensor(val_attention_mask, dtype=torch.long)
        val_labels_t = torch.tensor(val_labels, dtype=torch.long)

        # Create datasets
        train_dataset = TensorDataset(input_ids_t, attention_mask_t, labels_t)
        val_dataset = TensorDataset(val_input_ids_t, val_attention_mask_t, val_labels_t)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Set up loss and optimizer
        class_weights = torch.tensor([1.0, 100.0, 0.0], device=self._device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_id)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids_b, attention_mask_b, labels_b = [t.to(self._device) for t in batch]
                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids_b, attention_mask=attention_mask_b)
                logits = outputs.logits
                loss = criterion(logits.view(-1, self._num_labels), labels_b.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            self._model.eval()
            val_acc = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_b, attention_mask_b, labels_b = [t.to(self._device) for t in batch]
                    outputs = self._model(input_ids=input_ids_b, attention_mask=attention_mask_b)
                    logits = outputs.logits
                    val_acc += self._custom_accuracy_pytorch(logits, labels_b, pad_id)
                    n_batches += 1

            val_acc /= max(n_batches, 1)
            print(f"Epoch {epoch+1}: train loss={avg_train_loss:.4f}, val masked acc={val_acc:.4f}")

        self._model_trained = True

    def predict(self, val_input_ids, val_attention_mask):
        self._model.eval()
        with torch.no_grad():
            input_ids_t = torch.tensor(val_input_ids, dtype=torch.long).unsqueeze(0).to(self._device)
            attention_t = torch.tensor(val_attention_mask, dtype=torch.long).unsqueeze(0).to(self._device)
            outputs = self._model(input_ids=input_ids_t, attention_mask=attention_t)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    def save(self, filepath: str) -> None:
        assert self._model_trained, "Model has not been trained yet."
        path = os.path.join(filepath, "ner.pt")
        torch.save(self._model.state_dict(), path)

    def load(self, filepath: str) -> None:
        path = os.path.join(filepath, "ner.pt")
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.to(self._device)
        self._model.eval()
        self._model_trained = True

    def _custom_accuracy_pytorch(self, logits, labels, pad_id):
        preds = torch.argmax(logits, dim=-1)
        mask = labels != pad_id
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct / total

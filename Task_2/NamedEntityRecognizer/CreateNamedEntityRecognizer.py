from AnimalNameEntityRecognizer import AnimalNameEntityRecognizer
from DataGenerator import DataGenerator
from DataPreprocessor import DataPreprocessor

MAX_LEN = 64

if __name__ == "__main__":
    data_gen = DataGenerator()
    data_gen.generate_data("./ner_animal_generated_dataset.csv", 5000)

    data_proc = DataPreprocessor(max_len=MAX_LEN)
    data_proc.load_data("./ner_animal_generated_dataset.csv")
    (
        input_ids,
        attention_mask,
        labels,
        val_input_ids,
        val_attention_mask,
        val_labels,
    ) = data_proc.get_train_test()

    pad_id = data_proc.get_pad_id()

    ner = AnimalNameEntityRecognizer(num_labels=3, max_len=MAX_LEN)
    ner.train(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        val_input_ids=val_input_ids,
        val_attention_mask=val_attention_mask,
        val_labels=val_labels,
        pad_id=pad_id,
        epochs=3,
    )

    ner.save(".")

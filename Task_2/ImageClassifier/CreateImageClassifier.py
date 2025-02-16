from Translate import translate
from AnimalImageClassifier import AnimalImageClassifier

if __name__ == "__main__":
    animal_image_classifier = AnimalImageClassifier()
    animal_image_classifier.load_data(
        data_dir="raw-img", translate=translate, visualize=True
    )
    animal_image_classifier.train_model(train_epochs=1, plot_history=True)
    animal_image_classifier.visualize_classification(num_batches=1)
    animal_image_classifier.save_model(".")

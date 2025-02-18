import sys
import PIL.Image
import tensorflow as tf
from ModelPipeline import ModelPipeline

MAX_LEN = 64

image_classes = [
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "spider",
    "squirrel",
]

if __name__ == "__main__":
    mp = ModelPipeline(
        "./ImageClassifier/image_classifier.h5",
        "./NamedEntityRecognizer/ner.h5",
    )

    image_path = sys.argv[1]
    image_loaded = tf.keras.utils.load_img(image_path)
    predicted_class = image_classes[mp.predict_image(image_loaded)]

    sentence = sys.argv[2]
    animal_token = mp.get_animal_token(sentence, MAX_LEN)

    print(predicted_class == animal_token)

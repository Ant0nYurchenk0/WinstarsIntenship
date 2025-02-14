from Helper import bcolors
from Classifiers import MnistClassifier
from tensorflow.keras.datasets import mnist  # type: ignore

SPLIT_PERCENTAGE = 0.8


if __name__ == "__main__":

    (images, labels), (_, _) = mnist.load_data()
    images = images / 255  # normalising images

    print(f"images are of shape: {images.shape} and labels: {labels.shape}")

    size = images.shape[0]
    split = int(size * SPLIT_PERCENTAGE)
    # Subsample the images
    train_images = images[:split]
    train_labels = labels[:split]

    test_images = images[split:]
    test_labels = labels[split:]

    RFClassifier = MnistClassifier("rf")
    _, RFAcc = RFClassifier.classify(
        train_images, train_labels, test_images, test_labels
    )
    print(
        bcolors.OKGREEN + "Random forest accuracy score: " + str(RFAcc) + bcolors.ENDC
    )

    FFNNClassifier = MnistClassifier("nn")
    _, FFNNAcc = FFNNClassifier.classify(
        train_images, train_labels, test_images, test_labels
    )
    print(
        bcolors.OKGREEN
        + "Fast-forward neural network accuracy score: "
        + str(FFNNAcc)
        + bcolors.ENDC
    )

    CNNClassifier = MnistClassifier("cnn")
    _, CNNAcc = CNNClassifier.classify(
        train_images, train_labels, test_images, test_labels
    )
    print(
        bcolors.OKGREEN
        + "Convolutional neural network accuracy score: "
        + str(CNNAcc)
        + bcolors.ENDC
    )

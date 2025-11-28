import random
import pandas as pd
import string


class DataGenerator:
    def __init__(self):
        self._animal_names = [
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
        self._beginnings = [
            "Once upon a time, a {animal} wandered into a mysterious forest.",
            "In a quiet village, a {animal} discovered an ancient secret.",
            "A curious {animal} stumbled upon a hidden cave.",
            "Long ago, a {animal} set off on a grand adventure.",
            "A lonely {animal} roamed the vast plains in search of something special.",
            "One evening, a {animal} found itself in an enchanted garden.",
            "Deep in the jungle, a {animal} heard a strange sound.",
            "A {animal} in the meadow noticed something glowing in the distance.",
            "Under the bright moon, a {animal} felt a strange pull towards the river.",
            "A {animal} in the desert uncovered a long-lost relic."
        ]

        self._middles = [
            "It met a wise old {animal} who shared a mysterious riddle.",
            "A sudden storm forced it to seek shelter in a hidden cavern.",
            "The {animal} found a map leading to a legendary treasure.",
            "An unexpected friend, a talking bird, guided the {animal} along the way.",
            "It had to solve a puzzle to continue its journey.",
            "A mischievous creature tried to trick the {animal} out of its findings.",
            "The path was blocked by a giant boulder, but a kind {animal} helped move it.",
            "A magical pond reflected the {animal}'s deepest dreams.",
            "The {animal} discovered an ancient book filled with forgotten wisdom.",
            "A hidden passage led the {animal} into a secret underground world."
        ]

        self._endings = [
            "At last, the {animal} found what it had been searching for all along.",
            "It returned home, wiser and braver than before.",
            "The journey changed the {animal} forever, filling its heart with joy.",
            "A newfound friendship made the adventure truly special.",
            "The {animal} realized that the real treasure was the memories made.",
            "With the mystery solved, the {animal} could finally rest.",
            "The enchanted land bid the {animal} farewell as it continued its journey.",
            "Having learned an important lesson, the {animal} shared its story with others.",
            "The {animal} knew it would return one day for another grand adventure.",
            "As the sun set, the {animal} smiled, knowing its adventure was only the beginning."
        ]

    def generate_data(self, filepath: str, num_of_sentences: int) -> None:
        assert num_of_sentences > 0, "Number of sentences must be greater than 0."
        assert num_of_sentences < len(self._animal_names) * len(self._beginnings) * len(
            self._middles
        ) * len(
            self._endings
        ), "Number of sentences must be less than the total number of possible sentences."

        # Generate unique sentences
        unique_sentences = set()
        while len(unique_sentences) < 5000:
            animal = random.choice(self._animal_names)
            sentence = f"{random.choice(self._beginnings)} {random.choice(self._middles)} {random.choice(self._endings)}".format(
                animal=animal
            )

            if sentence not in unique_sentences:
                unique_sentences.add(sentence)

        # Restructure the dataset
        restructured_data = []
        sentence_id = 1

        for sentence in unique_sentences:
            words = sentence.split()
            labels = [
                "B-ANIMAL" if word.lower() in self._animal_names else "O"
                for word in words
            ]

            for word, label in zip(words, labels):
                restructured_data.append(
                    (sentence_id, word.strip(string.punctuation), label)
                )

            sentence_id += 1

        # Create a DataFrame
        df_unique_sentences = pd.DataFrame(
            restructured_data, columns=["Sentence Number", "Word", "Label"]
        )

        # Save the dataset to CSV
        file_path_unique_sentences = filepath
        df_unique_sentences.to_csv(file_path_unique_sentences, index=False)

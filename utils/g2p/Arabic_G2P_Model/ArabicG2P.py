import json

from dp.phonemizer import Phonemizer
from phonemizer.separator import Separator

SEARCH_SPACE_PATH = "./utils/g2p/Arabic_G2P_Model/Data/DataSet.json"


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)  # Load the JSON data from the file
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")  # Handle potential JSON decoding errors


class ArabicG2P:
    def __init__(self):
        self.phonemizer = Phonemizer.from_checkpoint("./utils/g2p/Arabic_G2P_Model/checkpoints/best_model.pt")
        self.SearchSpace = load_data(SEARCH_SPACE_PATH)
        self.separator = Separator(word="_", syllable="-", phone="|")

    def G2P(self, sentence: str) -> str:
        words = sentence.split()
        Phoneme_results = []
        for word in words:
            if word in self.SearchSpace:
                Phoneme_results.append(self.SearchSpace[word])
            else:
                Phoneme_results.append(self.phonemizer(word, lang='ar'))
        return ' '.join(Phoneme_results)

    def _customize_text(self,text, word_separator, letter_separator):
        # Split the text into words
        words = text.split()

        # Initialize an empty list to store modified words
        modified_words = []

        # Iterate over each word in the list of words
        for word in words:
            # Split the word into letters
            letters = list(word)

            # Add letter separator between each letter
            modified_word = letter_separator.join(letters)

            # Add the modified word to the list
            modified_words.append(modified_word)

        # Join the modified words with word separator to form the final text
        modified_text = word_separator.join(modified_words)

        return modified_text

import json

from dp.phonemizer import Phonemizer

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

    def G2P(self, sentence: str) -> str:
        words = sentence.split()
        Phoneme_results = []
        for word in words:
            if word in self.SearchSpace:
                Phoneme_results.append(self.SearchSpace[word])
            else:
                Phoneme_results.append(self.phonemizer(word, lang='ar'))
        return ' '.join(Phoneme_results)

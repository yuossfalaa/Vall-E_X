from pathlib import Path
from dp.phonemizer import Phonemizer


class ArabicG2P:
    def __init__(self):

        script_dir = Path(__file__).resolve().parent
        rel_path = "checkpoints/best_model.pt"
        abs_file_path = script_dir / rel_path
        self.phonemizer = Phonemizer.from_checkpoint(abs_file_path)

    def G2P(self, sentence: str) -> str:
        return self.phonemizer(sentence, lang='ar')

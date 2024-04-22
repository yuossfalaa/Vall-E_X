from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached
from macros import lang2token

from utils.g2p import PhonemeBpeTokenizer

if __name__ == '__main__':
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    lang_token = lang2token['ar']
    text="0"
    phoneme_tokens, lang = text_tokenizer.tokenize(
        f"_{lang_token +  text + lang_token}".strip())

    print(phoneme_tokens)

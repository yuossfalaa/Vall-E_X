from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

from data import TextTokenCollater, get_text_token_collater
from macros import lang2token

from utils.g2p import PhonemeBpeTokenizer

if __name__ == '__main__':
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    lang_token = lang2token['ar']
    text="0"
    phoneme_tokens, lang = text_tokenizer.tokenize(
        f"_{lang_token +  text + lang_token}".strip())
    text_collater = get_text_token_collater()
    text_tokens, enroll_x_lens = text_collater(
        [
            phoneme_tokens
        ]
    )
    print(phoneme_tokens)

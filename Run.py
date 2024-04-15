from utils.g2p import PhonemeBpeTokenizer
from macros import lang2token

if __name__ == '__main__':
    lang_token_ar = lang2token["ar"]
    lang_token_en = lang2token["en"]

    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    phoneme_tokens, lang = text_tokenizer.tokenize(
        f"_{lang_token_ar}سلام عليكم{lang_token_ar}".strip())
    print(phoneme_tokens)
    phoneme_tokens, lang = text_tokenizer.tokenize(
        f"_{lang_token_en}hi there{lang_token_en}".strip())
    print(phoneme_tokens)




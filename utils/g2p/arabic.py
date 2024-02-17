import re
import string

import num2words

from utils.g2p.Arabic_G2P_Model.ArabicG2P import ArabicG2P

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


def normalize_numbers(text):
    pattern = r"\d+"
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace(match, num2words.num2words(match, lang='ar'))
    return text


def normalize_arabic(text):
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_extra_space(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.\s+", ".", text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def arabic_cleaner(text):
    text = normalize_numbers(text)
    text = normalize_arabic(text)
    text = remove_punctuations(text)
    text = remove_extra_space(text)
    return text


def arabic_to_ipa(text):
    text = arabic_cleaner(text)
    arabicG2P = ArabicG2P()
    return arabicG2P.G2P(text)


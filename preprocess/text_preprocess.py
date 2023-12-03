""" Module de fonctions de preprocessing """

import unicodedata


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()


def clean_text(text):
    # Remplace les caractères non imprimables par une chaîne vide
    cleaned_text = ''.join(char if char.isprintable() else ' ' for char in text)
    return cleaned_text


def clean_and_flatten_text(text):
    text = remove_accents(text)
    text = " ".join(text.split("\n"))
    return text

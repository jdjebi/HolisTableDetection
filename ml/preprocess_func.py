import re

import pandas as pd
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords


def clean_text(text):
    porter = PorterStemmer()
    stop_words = set(stopwords.words())
    text = str(text)
    text = text.lower()
    text = re.sub('(<.*?>)|(\n)|([^\w\s\.\,])|([_])|([.])|([,])|(\s\s+)|([ا-ي])', '', text)
    cleaned_text = ' '.join([porter.stem(i) for i in word_tokenize(text) if i not in stop_words])
    return cleaned_text


def cleaning_pdf_interest(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['text'] = df_copy[col_name].apply(clean_text)
    return df_copy

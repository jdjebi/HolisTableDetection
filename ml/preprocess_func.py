import re

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


def cleaning_pdf_interest(df, col_name):
    cleaned_column = df[col_name].apply(clean_text)
    return cleaned_column

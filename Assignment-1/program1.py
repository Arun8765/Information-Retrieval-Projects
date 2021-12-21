from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk import pos_tag
import re
import string
import os
from sys import getsizeof
import unicodedata


def preprocess(data):
    data = data.lower()
    data = re.sub(r'\d+', '', data)
    data = re.sub(r'\n', ' ', data)
    data = re.sub(r'[^A-Za-z]+', ' ', data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = data.strip()

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(data)
    result = [i for i in tokens if i not in stop_words]
    result = [unicodedata.normalize('NFKD', i).encode('ascii', 'ignore').decode('utf-8', 'ignore') for i in result]

    stemmer = PorterStemmer()
    new_result = [stemmer.stem(i) for i in result]
    new_result = [i for i in new_result if i not in stop_words]

    return new_result


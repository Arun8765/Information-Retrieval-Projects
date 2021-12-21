import os,sys
import re, string, unicodedata
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.translate(str.maketrans("","",string.punctuation))
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        new_word = re.sub(r'\d+','',word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    stop_words = set(stopwords.words("english"))
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
def lexical_analysis(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    #words = remove_punctuation(words)
    words = remove_numbers(words)
    return words

vocabulary = []

for filename in os.listdir(os.getcwd() + "/Document_Preprocessing"):
    with open(os.path.join(os.getcwd() + "/Document_Preprocessing", filename), "r") as rf:
        print("size of", filename, ":", os.stat(os.getcwd() + "/Document_Preprocessing/" + filename).st_size, "bytes")

        processed_doc_name = filename

        sample = rf.read()

        sample = strip_html(sample)
        # removal of punctuations
        sample = sample.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = word_tokenize(sample)

        words = lexical_analysis(words)
        with open(os.path.join(os.getcwd() + "/processed_docs", processed_doc_name), "w") as wf:
            n = wf.write(" ".join(words))
        print("size after lexical analysis:", os.stat(os.getcwd() + "/processed_docs/" + processed_doc_name).st_size,
              "bytes")

        words = remove_stopwords(words)
        with open(os.path.join(os.getcwd() + "/processed_docs", processed_doc_name), "w") as wf:
            n = wf.write(" ".join(words))
        print("size after removing stopwords:", os.stat(os.getcwd() + "/processed_docs/" + processed_doc_name).st_size,
              "bytes")

        words = stem_words(words)
        with open(os.path.join(os.getcwd() + "/processed_docs", processed_doc_name), "w") as wf:
            n = wf.write(" ".join(words))
        print("size after stemming:", os.stat(os.getcwd() + "/processed_docs/" + processed_doc_name).st_size, "bytes")

        words = remove_stopwords(words)
        with open(os.path.join(os.getcwd() + "/processed_docs", processed_doc_name), "w") as wf:
            n = wf.write(" ".join(words))
        print("size after removing stop words once again after stemming:",
              os.stat(os.getcwd() + "/processed_docs/" + processed_doc_name).st_size, "bytes")

        vocabulary = vocabulary + words
        print("\n\n")
vocabulary = list(set(vocabulary))
vocabulary.sort()
with open(os.path.join(os.getcwd(),"vocabulary.txt"),"w") as wf:
    wf.write(" ".join(vocabulary))
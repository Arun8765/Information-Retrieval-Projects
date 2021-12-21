import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

xfiles = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt', 'T10.txt']


# Tokenizing the text, return token list


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    return nltk.word_tokenize(" ".join(tokens))


# Stopwords removal, return list
def stop_words_remove(tokens):
    stop = stopwords.words('english')
    new_tokens = [i for i in tokens if i not in stop]
    return new_tokens

def remove_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        new_word = re.sub(r'\d+','',word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

# Stemming of tokens, return list
def stem_tokens(new_tokens):
    ps = PorterStemmer()
    stemmed = []
    for i in new_tokens:
        stemmed.append(ps.stem(i))
    return stemmed


inv_file = r'Inverted.csv'


def get_relevance(n, nw):
    return (n - nw + 0.5) / (nw + 0.5)


def get_probability_matrix(n, df, toks):
    prob_matrix = {}

    for i in toks:
        nw = df.loc[i, 'Occurences'].count(')')

        prob_matrix[i] = [nw, get_relevance(n, nw)]
    return prob_matrix


def get_query_tokens(query):
    tokens = preprocess(query.lower())
    tokens = remove_numbers(tokens)
    tokens = stop_words_remove(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def get_conditional_probability(qtok, inv_file):
    prob_matrix = {}

    df = pd.read_csv(inv_file)
    toks = list(df['Tokens'])

    df.set_index('Tokens', inplace=True)

    word_matrix = get_probability_matrix(len(xfiles), df, toks)

    # print(word_matrix)
    for i in xfiles:
        flag = False
        val = 1
        prob_matrix[i] = 0

        for j in qtok:
            if j in toks:
                if i in df.loc[j, 'Occurences']:
                    flag = True
                    val *= word_matrix[j][1]
        prob_matrix[i] = val if flag else 0

    return prob_matrix


print("Program 4: \n\tProbablistic Model Implementation")


print("Enter your query here :", end=" ")
rel_docs = get_conditional_probability(get_query_tokens(input()), inv_file)

rel_docs = {k: "{0:.5f}".format(v) for k, v in sorted(rel_docs.items(), key=lambda item: item[1], reverse=True)}

# print(vect)
print("The documents in the order of relevance to the query are as follows: ")
print(pd.DataFrame(rel_docs.items(), columns=['File', 'Relevance']))

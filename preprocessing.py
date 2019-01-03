import pandas as pd
import re, nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def take_data_to_shower(tweet):
    noises = ['URL', 'USER', '\'ve', 'n\'t', '\'s', '\'m']

    for noise in noises:
        tweet = tweet.replace(noise, '')

    return re.sub(r'[^a-zA-Z]', ' ', tweet)


def tokenize(tweet):
    return tweet.split(' ')


def remove_stop_words(tokens):
    clean_tokens = []
    stopWords = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopWords:
            if token.replace(' ', '') != '':
                clean_tokens.append(token)
    return clean_tokens


def stem_and_lem(tokens):
    clean_tokens = []
    for token in tokens:
        token = lancaster_stemmer.stem(token)
        token = wordnet_lemmatizer.lemmatize(token)
        clean_tokens.append(token)
    return clean_tokens

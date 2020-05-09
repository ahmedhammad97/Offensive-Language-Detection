from os import listdir
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfid(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors

def tfid_test(train_vectors, test_vectors):
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in train_vectors]
    vectorizer = vectorizer.fit(untokenized_data)

    untokenized_data =[' '.join(tweet) for tweet in test_vectors]
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors


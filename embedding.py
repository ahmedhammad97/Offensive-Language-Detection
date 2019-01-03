from os import listdir
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np

def build_model(text_vector, retrain=False):
    model = None
    if 'dict.model' not in listdir('./') or retrain:
        print("Training The Model..Please Be Patient...")
        model = Word2Vec(text_vector, size=80, window=15, min_count=1, workers=5)
        model.train(text_vector, total_examples=len(text_vector), epochs=15)
        print("Model Trained And Saved Successfully...")
    else:
        print("Loading Model...")
        model = Word2Vec.load('./dict.model')
        print("Loaded Successfully...")

    return model


def vectorize(model, text_vector):
    vector = model.wv
    vectors = [np.array([vector[word] for word in tweet if word in model]).flatten() for tweet in tqdm(text_vector,'Vectorizing Phase I...')]
    max_length = np.max([len(vector) for vector in vectors])
    result_vectors = [np.array(vector.tolist()+[0 for _ in range(max_length-len(vector))]) for vector in tqdm(vectors,'Vectorizing Phase II...')]
    return result_vectors

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

# def build_model(text_vector, retrain=False):
#     model = None
#     if 'dict.model' not in listdir('./') or retrain:
#         print("Training The Model..Please Be Patient...")
#         model = Word2Vec(text_vector, size=80, window=15, min_count=10, workers=5)
#         model.train(text_vector, total_examples=len(text_vector), epochs=15)
#         model.save('./dict.model')
#         print("Model Trained And Saved Successfully...")
#     else:
#         print("Loading Model...")
#         model = Word2Vec.load('./dict.model')
#         print("Loaded Successfully...")
#
#     return model
#
#
# def vectorize(model, text_vector):
#     vector = model.wv
#     vectors = [np.array([vector[word] for word in tweet if word in model]).flatten() for tweet in tqdm(text_vector,'Vectorizing Phase I...')]
#     print(vectors[400:500])
#     max_length = np.max([len(vector) for vector in vectors])
#     result_vectors = [np.array(vector.tolist()+[0 for _ in range(max_length-len(vector))]) for vector in tqdm(vectors,'Vectorizing Phase II...')]
#     return result_vectors

from os import listdir
from gensim.models import Word2Vec

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

    #Tesing only
    print("TESTING ONLY PRINTS")
    print(model.wv.most_similar(positive="shit"))

    return model

import pandas as pd
import numpy as np
from tqdm import tqdm
import preprocessing, embedding
import copy
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

directory = "datasets/training-v1/offenseval-training-v1.tsv"
print("Reading Dataset...")
train_data = pd.read_csv(directory, sep='\t', header=0)

tweets = train_data[["id", "tweet"]]
subtask_a_labels = train_data[["id", "subtask_a"]]
subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["id", "subtask_b"]]
subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["id", "subtask_c"]]

clean_tweets = copy.deepcopy(tweets)

tqdm.pandas(desc="Cleaning Data Phase I...")
clean_tweets['tweet'] = tweets['tweet'].progress_apply(preprocessing.take_data_to_shower)

tqdm.pandas(desc="Tokenizing Data...")
clean_tweets['tokens'] = clean_tweets['tweet'].progress_apply(preprocessing.tokenize)

tqdm.pandas(desc="Cleaning Data Phase II...")
clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(preprocessing.remove_stop_words)

tqdm.pandas(desc="Stemming And Lemmatizing")
clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(preprocessing.stem_and_lem)

text_vector = clean_tweets['tokens'].tolist()
model = embedding.build_model(text_vector)

vectors = embedding.vectorize(model, text_vector) # Numerical Vectors
labels = subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

# Random Splitting With Ratio 3 : 1
train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels, test_size=0.25)

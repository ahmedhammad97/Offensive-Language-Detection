#######################################
## Author: Ahmed Hammad
## License: MIT
## Email: hammad97official@gmail.com
## Website: www.ahmedhammad97.com
#######################################

import pandas as pd
import numpy as np
from tqdm import tqdm
import preprocessing, embedding, helper, classifying
import copy


train_directory = "datasets/training-v1/offenseval-training-v1.tsv"
print("Reading Dataset...")
train_data = pd.read_csv(train_directory, sep='\t', header=0)

tweets = train_data[["tweet"]]
subtask_a_labels = train_data[["subtask_a"]]
subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]

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

vectors_a = embedding.tfid(text_vector) # Numerical Vectors A
labels_a = subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

vectors_b = helper.get_vectors(vectors_a, labels_a, "OFF") # Numerical Vectors B
labels_b = subtask_b_labels['subtask_b'].values.tolist() # Subtask B Labels

vectors_c = helper.get_vectors(vectors_b, labels_b, "TIN") # Numerical Vectors C
labels_c = subtask_c_labels['subtask_c'].values.tolist() # Subtask C Labels

print("\nBuilding Model Subtask A...")
classifying.classify(vectors_a[:], labels_a[:], text_vector, "A", "MNB")

print("\nBuilding Model Subtask B...")
classifying.classify(vectors_b[:], labels_b[:], text_vector, "B", "SVM")

print("\nBuilding Model Subtask C...")
classifying.classify(vectors_c[:], labels_c[:], text_vector, "C", "RF")

# You can choose from the classifiers {MNB, KNN, SVM, DT, RF, LR}
# You can also try only a subset of the data for quick classification:
#   vectors_a[1000:3000], labels_a[1000:3000]
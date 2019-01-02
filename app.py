import pandas as pd
import numpy as np
from tqdm import tqdm
import preprocessing

directory = "datasets/training-v1/offenseval-training-v1.tsv"
train_data = pd.read_csv(directory, sep='\t', header=0)

tweets = train_data[["tweet"]]
subtask_a_labels = train_data[["subtask_a"]]
subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]

clean_tweets = preprocessing.take_data_to_shower(tweets)

tfile = open('output.txt', 'a')
tfile.write(clean_tweets.to_string())
tfile.close()

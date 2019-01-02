import pandas as pd
import numpy as np
import preprocessor as p
from tqdm import tqdm
import re

directory = "datasets/training-v1/offenseval-training-v1.tsv"
train_data = pd.read_csv(directory, sep='\t', header=0)

tweets = train_data[["tweet"]]
subtask_a_labels = train_data[["subtask_a"]]
subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER)
clean_tweets = pd.DataFrame(columns=["tweet"])
for tweet in tqdm(tweets['tweet'], "Cleaning Data"):
    noises = ['#', '_', 'URL', ',', '.', '"', "'", '?', '!', '+', '=', '*']
    noises.extend(['-', '(', ')', ']', '[', '&', '$', '\\', '/', ':', '%', ';'])
    noises.extend(['1','2','3','4','5','6','7','8','9'])
    clean_tweet = p.clean(tweet)
    for noise in noises:
        clean_tweet = clean_tweet.replace(noise, '')

    emoji_pattern = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

    clean_tweet = emoji_pattern.sub(r'', clean_tweet) # no emoji

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    clean_tweet = emoji_pattern.sub(r'', clean_tweet) # no emoji

    clean_tweets = clean_tweets.append({'tweet': clean_tweet}, ignore_index=True)

tfile = open('output.txt', 'a')
tfile.write(clean_tweets.to_string())
tfile.close()

import pandas as pd
import preprocessor as p
import re
from tqdm import tqdm

def take_data_to_shower(tweets):
    clean_tweets = pd.DataFrame(columns=["tweet"])

    # Preprocessor settings
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER)

    for tweet in tqdm(tweets['tweet'], "Cleaning Data"):
        clean_tweet = p.clean(tweet) # Removes Mentions, Smiles, Numbers, URLs
        clean_tweet = remove_noise(clean_tweet) # Removes Punctuation and Undersired Chars
        clean_tweet = remove_emojis(clean_tweet)
        clean_tweets = clean_tweets.append({'tweet': clean_tweet}, ignore_index=True)

    return clean_tweets


def remove_noise(clean_tweet):
    noises = ['#', '_', 'URL', ',', '.', '"', "'", '?', '!', '+', '=', '*']
    noises.extend(['-', '(', ')', ']', '[', '&', '$', '\\', '/', ':', '%', ';'])
    noises.extend(['1','2','3','4','5','6','7','8','9'])

    for noise in noises:
        clean_tweet = clean_tweet.replace(noise, '')

    return clean_tweet


def remove_emojis(clean_tweet):
    emoji_pattern = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
    clean_tweet = emoji_pattern.sub(r'', clean_tweet)

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    clean_tweet = emoji_pattern.sub(r'', clean_tweet)

    return clean_tweet

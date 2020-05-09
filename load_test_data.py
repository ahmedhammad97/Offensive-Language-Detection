import pandas as pd
import preprocessing, embedding, helper, classifying
import copy

def load(train_vector, task='A'):
    directory = "datasets/trial-data/offenseval-trial.txt"
    print("Preparing Test Data...")
    data = pd.read_csv(directory, sep='\t', header=None)
    data.columns = ["tweet", "subtask_a", "subtask_b", "subtask_c"]

    tweets = data[["tweet"]]
    subtask_a_labels = data[["subtask_a"]]
    subtask_b_labels = data.query("subtask_a == 'OFF'")[["subtask_b"]]
    subtask_c_labels = data.query("subtask_b == 'TIN'")[["subtask_c"]]

    clean_tweets = copy.deepcopy(tweets)

    clean_tweets['tweet'] = tweets['tweet'].apply(preprocessing.take_data_to_shower)

    clean_tweets['tokens'] = clean_tweets['tweet'].apply(preprocessing.tokenize)

    clean_tweets['tokens'] = clean_tweets['tokens'].apply(preprocessing.remove_stop_words)

    clean_tweets['tokens'] = clean_tweets['tokens'].apply(preprocessing.stem_and_lem)

    text_vector = clean_tweets['tokens'].tolist()

    vectors_a = embedding.tfid_test(train_vector, text_vector) # Numerical Vectors A
    labels_a = subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

    vectors_b = helper.get_vectors(vectors_a, labels_a, "OFF") # Numerical Vectors B
    labels_b = subtask_b_labels['subtask_b'].values.tolist() # Subtask B Labels

    vectors_c = helper.get_vectors(vectors_b, labels_b, "TIN") # Numerical Vectors C
    labels_c = subtask_c_labels['subtask_c'].values.tolist() # Subtask C Labels

    if(task=='A' or task=='a'):
        return vectors_a, labels_a
    elif(task=='B' or task=='b'):
        return vectors_b, labels_b
    elif(task=='C' or task=='c'):
        return vectors_c, labels_c
    else:
        print("Wrong Subtask!")
        return None

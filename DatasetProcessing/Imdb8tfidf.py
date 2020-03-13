import os
import timeit
import itertools
import numpy as np
from gensim.models import Word2Vec
import gensim
import sys

def num(s):
    try:
        return int(s)
    except ValueError:
        return 5

from DatasetProcessing.Lib import processText
min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)

def readData(directory, data, data_rating,readall=True):
    nrows=20
    min_length = 3

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory+filename, "r", encoding="latin1")
            reviewtext=f.read()
            ratingtext = ((filename.split("."))[0].split("_"))[1]  # split on underscores

            rating = num(ratingtext)
            tokens,status = tokenize(reviewtext)
            if(status==False):continue
            if (int(rating) in [5, 6] and len(tokens) < min_length): continue

            data_rating.append(rating)
            data.append(" ".join(tokens))

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1

def read(home_dir, output_dir):
    input_file1 = home_dir + "train/pos/"
    input_file2 = home_dir + "train/neg/"
    input_file3 = home_dir + "test/pos/"
    input_file4 = home_dir + "test/neg/"

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)
    data=[]
    data_rating = []

    print("Started Reading data")
    start_reading = timeit.default_timer()
    readData(input_file1, data, data_rating)
    readData(input_file2, data, data_rating)
    readData(input_file3, data, data_rating)
    readData(input_file4, data, data_rating)

    from DatasetProcessing.Lib import save_labels_txt, tf_idf_result

    save_labels_txt(data_rating, output_dir, dataset_name="imdb8_tfidf")

    TF_IDF_config = {
        'max_df': 0.5,
        'min_df': 3,
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }

    data_vector = tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="imdb8_tfidf")

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",data_vector.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print("Total:",len(data_rating))
    return (data, data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["imdb8tfidf"]["input_path"],dataset_path["imdb8tfidf"]["output_path"])
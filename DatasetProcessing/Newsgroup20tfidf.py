from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import timeit
import numpy as np
import pickle

from DatasetProcessing.Lib import processText

min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)

def readData(output_dir):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    categories=list(newsgroups_train.target_names)
    print(categories)

    trsize=len(newsgroups_train.data)
    testsize=len(newsgroups_test.data)

    print("Training size :",trsize)
    print("Test size :", testsize)

    data=[]
    data_rating=[]

    print("Started filtering text")
    for i in range(len(newsgroups_train.data)):
        (tokens,status)=tokenize(newsgroups_train.data[i])
        if(status):
            data_rating.append(newsgroups_train.target[i])
            data.append(" ".join(tokens))
    print("Filtering ended")

    for i in range(len(newsgroups_test.data)):
        (tokens,status)=tokenize(newsgroups_test.data[i])
        if(status):
            data_rating.append(newsgroups_test.target[i])
            data.append(" ".join(tokens))

    return (data,data_rating)

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()
    (data, data_rating)=readData(output_dir)

    from DatasetProcessing.Lib import save_labels_txt, tf_idf_result

    save_labels_txt(data_rating, output_dir, dataset_name="newsgroup20_tfidf")

    TF_IDF_config = {
        'max_df': 0.5,
        'min_df': 3,
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }

    data_vector = tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="newsgroup20_tfidf")

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",data_vector.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["newsgroup20tfidf"]["output_path"])
import os
import timeit
import numpy as np
import csv

from DatasetProcessing.Lib import processText
import gzip

min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)
    if(len(filtered_tokens)<min_length):
        return ("",False)
    return (filtered_tokens,True)

def num(s):
    try:
        return int(s)
    except ValueError:
        return -1

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def readData(filename,data,data_rating,readall=True):
    nrows=2000
    for review in parse(filename):
        # print(review)
        # data.append([review['reviewerID'],review['asin'],review['reviewText'],review['overall']])
        # data.append(review['reviewText'])
        tokens,status=tokenize(review['reviewText'])
        rating=num(review['overall'])
        if(status==False or rating==3):continue

        data.append(" ".join(tokens))
        data_rating.append(rating)

        if (readall == False):
            if (nrows < 0):
                break
            nrows -= 1


def read(home_dir,output_dir):
    input_file = home_dir + "aggressive_dedup.json.gz"
    output_file = home_dir + 'amazon_review_5k.mtx'
    output_label = home_dir + 'amazon_review_5k.label'
    word_mapping = home_dir + 'amazon_word_5k.json'

    print(input_file)

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()

    data=[]
    data_rating=[]

    readData(input_file, data, data_rating)

    from DatasetProcessing.Lib import save_labels_txt, tf_idf_result

    save_labels_txt(data_rating, output_dir, dataset_name="amazon4_tfidf")

    TF_IDF_config = {
        'max_df': 0.6,
        'min_df': 10,
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }

    data_vector = tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="amazon4_tfidf")

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",data_vector.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["amazon4tfidf"]["input_path"], dataset_path["amazon4tfidf"]["output_path"])
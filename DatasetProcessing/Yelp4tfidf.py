import os
import timeit
import numpy as np
import csv

from DatasetProcessing.Lib import processText
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

def readData(filename,data,data_rating,readall=True):
    nrows = 20

    with open(filename, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            rating = num(row['stars'])
            tokens,status = tokenize(row['text'])

            if(status==False):continue
            if (rating== 3):continue
            data_rating.append(rating)  # 0 is class label
            data.append(" ".join(tokens))  # 1 is title 2 is abstract

            if readall==False:
                if (nrows < 0):
                    break
                nrows-=1

def read(home_dir,output_dir):
    input_file1 = home_dir + "yelp_review.csv"

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()

    data=[]
    data_rating=[]

    readData(input_file1, data, data_rating)

    from DatasetProcessing.Lib import save_labels_txt, tf_idf_result

    save_labels_txt(data_rating, output_dir, dataset_name="yelp4_tfidf")

    TF_IDF_config = {
        'max_df': 0.25,
        'min_df': 5,
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }

    data_vector = tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="yelp4_tfidf")

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",data_vector.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["yelp4tfidf"]["input_path"], dataset_path["yelp4tfidf"]["output_path"])
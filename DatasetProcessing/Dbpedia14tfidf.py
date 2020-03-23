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
    nrows=20
    with open(filename,newline='',encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            #print(row)
            category=num(row[0])
            if(category==-1):continue
            title=row[1]
            description=row[2]
            #print(title+description)
            tokens,status=tokenize(title+" "+description)
            #print(tokens)

            if (status == False): continue
            data.append(" ".join(tokens))
            data_rating.append(category)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1

def read(home_dir,output_dir):
    input_file1 = home_dir + "train.csv"
    input_file2 = home_dir + "test.csv"

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()

    data=[]
    data_rating=[]

    readData(input_file1, data,data_rating)
    readData(input_file2, data,data_rating)

    from DatasetProcessing.Lib import save_labels_txt, tf_idf_result

    save_labels_txt(data_rating, output_dir, dataset_name="dbpedia14_tfidf")

    TF_IDF_config = {
        'max_df': 0.5,
        'min_df': 10,
        'max_features': 10000,
        'ngram': (1, 1)  # min max range
    }

    data_vector = tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="dbpedia14_tfidf")

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",data_vector.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["dbpedia14tfidf"]["input_path"], dataset_path["dbpedia14tfidf"]["output_path"])
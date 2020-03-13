import os
import timeit
from nltk.corpus import reuters
import itertools
import numpy as np
from gensim.models import Word2Vec
import gensim
import sys
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

from DatasetProcessing.Path import load_model
model=load_model("GLOVE")

def readData(filename,data_vector,data_rating,readall=True):
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
            vocab_tokens = [word for word in tokens if word in model.vocab]
            if (len(vocab_tokens) < min_length):
                print(tokens)
                print(vocab_tokens)
                continue

            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector.append(vector)
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

    data_vector=[]
    data_rating=[]

    readData(input_file1, data_vector,data_rating)
    readData(input_file2, data_vector,data_rating)

    from DatasetProcessing.Lib import save_labels_txt, save_data_vector_list_mtx
    save_labels_txt(data_rating, output_dir, dataset_name="dbpedia14_w2v")
    save_data_vector_list_mtx(data_vector, output_dir, dataset_name="dbpedia14_w2v")

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print("Total:", len(data_rating))
    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",np.array(data_vector))

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["dbpedia14w2v"]["input_path"], dataset_path["dbpedia14w2v"]["output_path"])
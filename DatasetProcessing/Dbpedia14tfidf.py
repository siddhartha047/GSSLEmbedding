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

def save_stats(output_dir,data_vector, data_rating):
    label_file_name = output_dir + 'dbpedia14_labels.txt'
    with open(label_file_name, 'wb') as f:
        np.savetxt(f, [len(data_rating)], fmt='%d')
    with open(label_file_name, 'a+') as f:
        np.savetxt(f, data_rating, "%d")

    category_map = dict()
    for category in data_rating:
        if(category in category_map.keys()):
            category_map[category] = category_map[category] + 1
        else:
            category_map[category] = 0

    print(category_map)
    with open(output_dir+"categories_all.txt","w") as f:
        for k,v in category_map.items():
            f.write('%s,%d\n'%(k,v))

    m=len(data_rating)
    n=300

    header = np.array([[m, n, m * n]])

    filename=output_dir+"dbpedia14_w2v_vector.mtx"

    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f.write("%d %d %f\n" % (i, j, data_vector[i - 1][j - 1]))

    return (data_vector,data_rating)


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

    np.save(output_dir + "data_rating_np", data_rating)
    save_stats(output_dir,data_vector,data_rating)
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["dbpediatfidf"]["input_path"], dataset_path["dbpediatfidf"]["output_path"])
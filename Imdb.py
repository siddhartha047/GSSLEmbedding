import timeit
import os
import numpy as np
from Lib import *
from Word2Vec import *
import json as jsn
from scipy import io
import pickle

start = timeit.default_timer()
model=load_model(MODEL_NAME)
stop = timeit.default_timer()
print('Model loading time: ', stop - start)

def readData(directory,data,data_vector,data_rating,minCharLength,readall=False):
    nrows=20

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory+filename, "r", encoding="latin1")
            reviewtext=f.read();
            ratingtext = ((filename.split("."))[0].split("_"))[1]  # split on underscores

            rating = num(ratingtext)
            review = processText(reviewtext)

            if (len(review) >= minCharLength and int(rating) != 3):
                data.append(review)
                data_vector.append(apply_embedding(review, model))
                data_rating.append(rating)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1

def main():
    # home_dir = "/global/cscratch1/sd/sferdou/amazon_data/"
    home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/Imdb/aclImdb/";
    input_file1 = home_dir + "train/pos/"
    input_file2 = home_dir + "train/neg/"
    input_file3 = home_dir + "test/pos/"
    input_file4 = home_dir + "test/neg/"

    output_file = home_dir + 'imdb_review.mtx'
    output_label = home_dir + 'imdb_review.label'
    output_data = home_dir + 'imdb_data'

    data=[]
    data_vector=[]
    data_rating=[]

    #ignores rating 3, review with text length less than140
    #to read all pass True
    minCharLength=20
    readall=False

    print("Started Reading data")
    start_reading = timeit.default_timer()
    readData(input_file1, data, data_vector, data_rating, minCharLength,readall)
    readData(input_file2, data, data_vector, data_rating, minCharLength, readall)
    readData(input_file3, data, data_vector, data_rating, minCharLength, readall)
    readData(input_file4, data, data_vector, data_rating, minCharLength, readall)
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    # data=['The cat sat on the mat.','The ate ate dog # @ ate my homework 123','hibijibi']
    # data = [processText(text) for text in data]
    # data_rating=np.array([1,2,3])

    print("Data count: ",len(data))
    print("Vector count: ",len(data_vector))
    print("Rating count: ", len(data_rating))

    print("Started Writing data")
    pickle.dump(data_vector,open(output_data,"wb"))
    io.mmwrite(output_file, data_vector, comment="imdb review data")

    f = open(output_label, 'w')
    f.write("%d\n" % len(data_rating))

    print("Writing Class label")
    with open(output_label, 'a') as f:
        for item in data_rating:
            f.write("%s\n" % item)
    f.close()

if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
import timeit
import os
import numpy as np
from Lib import *
from Word2Vec import *

start = timeit.default_timer()
model=load_model(MODEL_NAME)
stop = timeit.default_timer()
print('Model loading time: ', stop - start)

def readData(filename,data,data_vector,data_rating,minCharLength,readall=False):
    nrows=20
    with open(filename,encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            rating=num(row['stars'])
            review = processText(row['text'])

            if(int(rating)!=3):
                data.append(" ".join(review))
                data_vector.append(apply_embedding(review, model))
                data_rating.append(rating)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1


def main():
    # home_dir = "/global/cscratch1/sd/sferdou/amazon_data/"
    home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/Yelp/"
    input_file = home_dir + "yelp_review.csv"
    output_file = home_dir + 'yelp_review.mtx'
    output_label = home_dir + 'yelp_review.label'
    output_data = home_dir + 'yelp_data'

    data=[]
    data_vector=[]
    data_rating=[]

    print("Started Reading data")
    start_reading = timeit.default_timer()
    readData(input_file, data, data_vector, data_rating, minCharLength=20, readall=True)
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print("Data count: ",len(data))
    print("Vector count: ",len(data_vector))
    print("Rating count: ", len(data_rating))

    save_data(data,data_vector,data_rating,output_file,output_label,output_data,comment="yelp review vector")

if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
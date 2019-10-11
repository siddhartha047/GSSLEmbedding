import timeit
import os
import numpy as np
from Lib import *
from Word2Vec import *

start = timeit.default_timer()
model=load_model(MODEL_NAME)
stop = timeit.default_timer()
print('Model loading time: ', stop - start)

def readData(fileName,data,data_vector,data_rating,minCharLength,readall=False):
    nrows = 20

    with open(fileName, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            if len(row) == 3:
                label=num(row[0])
                text=processText(row[1] + row[2])
                data_rating.append(label)  # 0 is class label
                data_vector.append(apply_embedding(text, model))
                data.append(" ".join(text))  # 1 is title 2 is abstract
                #print(" ".join(text))
            else:
                print("improper format\n")

            if readall==False:
                if (nrows < 0):
                    break
                nrows-=1

def main():
    # home_dir = "/global/cscratch1/sd/sferdou/amazon_data/"
    home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/DBpedia/dbpedia_csv/"

    inputFile1 = home_dir + "train.csv"
    inputFile2 = home_dir + "test.csv"

    output_file = home_dir + 'dbpedia_description.mtx'
    output_label = home_dir + 'dbpedia_class.label'
    output_data = home_dir + 'dbpedia_data'

    data=[]
    data_vector=[]
    data_rating=[]

    print("Started Reading data")
    start_reading = timeit.default_timer()
    readData(inputFile1, data, data_vector, data_rating, minCharLength=140, readall=True)  # False: read first 20 data
    readData(inputFile2, data, data_vector, data_rating, minCharLength=140, readall=True)  # False: read first 20
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print("Data count: ",len(data))
    print("Vector count: ",len(data_vector))
    print("Rating count: ", len(data_rating))

    save_data(data,data_vector,data_rating,output_file,output_label,output_data,comment="dbpedia review vector")

if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
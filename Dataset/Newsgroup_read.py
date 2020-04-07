from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import timeit
import numpy as np
import pickle

from Dataset.Lib import processText

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


def read(home_dir,output_dir,load_saved):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data = []
    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        (data,data_rating)=readData(output_dir)
        stop_reading = timeit.default_timer()
        print('Time to process: ', stop_reading - start_reading)
    else:
        print("Loading Saved data")
        data = np.load(output_dir + "data_np.npy",allow_pickle=True)
        data_rating = np.load(output_dir + "data_rating_np.npy", allow_pickle=True)
        print("Loading Done")

    from Dataset.Lib import save_data_rating_numpy
    save_data_rating_numpy(output_dir,data,data_rating)

    return (data,data_rating)

if __name__ == '__main__':
    read("", "/Users/siddharthashankardas/Purdue/Dataset/Newsgroup/", False)
    #readData("",[], [], [], 10, False)
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import timeit
import numpy as np
import pickle

from Dataset.Lib import processText

def tokenize(text):
    min_length = 3

    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (" ".join(filtered_tokens),True)

def readData(output_dir,readall,minWordLength):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    categories=list(newsgroups_train.target_names)

    trsize=len(newsgroups_train.data)
    testsize=len(newsgroups_test.data)

    print("Training size :",trsize)
    print("Test size :", testsize)


    data=[]
    data_rating=[]

    print("Started filtering text")
    index=0
    for i in range(len(newsgroups_train.data)):
        (text,status)=tokenize(newsgroups_train.data[i])
        if(status):
            data.append(text)
            data_rating.append(newsgroups_train.target[i])
            index += 1
    print("Filtering ended")
    train_size=index
    for i in range(len(newsgroups_test.data)):
        (text,status)=tokenize(newsgroups_test.data[i])
        if(status):
            data.append(text)
            data_rating.append(newsgroups_test.target[i])
            index += 1
    print("Filtering ended")
    test_size=index-train_size

    print("New train size ", train_size, " Removed :", trsize-train_size)
    print("New test size ", test_size, " Removed :",testsize-test_size)

    print("Total example remove: ",trsize+testsize-train_size-test_size)


    print("Total ",len(data))
    print("Total rating ",len(data_rating))

    categories_to_index= dict(zip(categories, range(len(categories))))
    index_to_categories = np.array(categories)

    print(categories_to_index)
    print(index_to_categories)

    f = open(output_dir+"categories_to_index.pkl", "wb")
    pickle.dump(categories_to_index, f)
    f.close()
    np.save(output_dir + 'index_to_categories', index_to_categories)

    train_index = np.array(range(train_size))
    test_index = np.array(range(train_size,index))

    np.save(output_dir+'train_index',train_index)
    np.save(output_dir + 'test_index', test_index)

    return (np.array(data), np.array(data_rating))


def read(home_dir,output_dir,load_saved):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data = []
    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True
    minWordLength = 10
    readall = False

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        (data,data_rating)=readData(output_dir,readall,minWordLength)
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
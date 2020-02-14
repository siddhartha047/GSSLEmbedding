from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import timeit
import numpy as np
import pickle

def readData(output_dir):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    categories=list(newsgroups_train.target_names)

    print("Training size :",len(newsgroups_train.data))
    print("Test size :", len(newsgroups_test.data))

    data=newsgroups_train.data
    data.extend(newsgroups_test.data)

    data_rating=newsgroups_train.target
    data_rating=np.append(data_rating,newsgroups_test.target)

    print("Total ",len(data))
    print("Total rating ",data_rating.shape)

    categories_to_index= dict(zip(categories, range(len(categories))))
    index_to_categories = np.array(categories)

    print(categories_to_index)
    print(index_to_categories)

    f = open(output_dir+"categories_to_index.pkl", "wb")
    pickle.dump(categories_to_index, f)
    f.close()
    np.save(output_dir + 'index_to_categories', index_to_categories)

    train_index= np.array(range(len(newsgroups_train.data)))
    test_index = np.array(range(len(newsgroups_train.data),len(newsgroups_train.data)+len(newsgroups_test.data)))
    np.save(output_dir+'train_index',train_index)
    np.save(output_dir + 'test_index', test_index)

    return (data, data_rating)


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
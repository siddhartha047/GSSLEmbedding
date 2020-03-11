import os
import timeit
import itertools
import numpy as np
from gensim.models import Word2Vec
import gensim
import sys
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import timeit
import numpy as np
import pickle


def load_model(model_name):
    from DatasetProcessing.Path import pretrained_model
    if (model_name == "GLOVE"):
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(pretrained_model[model_name]["path"]), binary=False,
                                                                encoding="ISO-8859-1")
    elif (model_name == "GOOGLE"):
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(pretrained_model[model_name]["path"]), binary=True)

    else:
        print("Model not implemented yet")
        sys.exit(0)

    return model

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

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data_rating = []
    print("Started Reading data")
    start_reading = timeit.default_timer()
    (data_vector, data_rating)=readData(output_dir, data_rating)
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    np.save(output_dir + "data_vector_np", data_vector)
    np.save(output_dir + "data_rating_np", data_rating)

    print(data_vector.shape)

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["newsgroup"]["output_path"])
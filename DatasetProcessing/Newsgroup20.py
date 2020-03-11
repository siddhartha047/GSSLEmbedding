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

from DatasetProcessing.Lib import processText

min_length = 3

def tokenize(text):

    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)


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

def readData(output_dir):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    categories=list(newsgroups_train.target_names)
    print(categories)

    trsize=len(newsgroups_train.data)
    testsize=len(newsgroups_test.data)

    print("Training size :",trsize)
    print("Test size :", testsize)

    model=load_model("GOOGLE")

    data=[]
    data_vector=[]
    data_rating=[]

    print("Started filtering text")
    index=0
    for i in range(len(newsgroups_train.data)):
        (tokens,status)=tokenize(newsgroups_train.data[i])
        if(status):
            vocab_tokens = [word for word in tokens if word in model.vocab]
            if(len(vocab_tokens)<min_length):
                print(tokens)
                print(vocab_tokens)
                continue
            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector.append(vector)
            data_rating.append(newsgroups_train.target[i])
            data.append(" ".join(tokens))
            index += 1
    print("Filtering ended")
    train_size=index

    for i in range(len(newsgroups_test.data)):
        (tokens,status)=tokenize(newsgroups_test.data[i])
        if(status):
            vocab_tokens = [word for word in tokens if word in model.vocab]
            if (len(vocab_tokens) < min_length):
                print(tokens)
                print(vocab_tokens)
                continue
            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector.append(vector)
            data_rating.append(newsgroups_test.target[i])
            data.append(" ".join(tokens))
            index += 1
    print("Filtering ended")

    test_size=index-train_size
    print("New train size ", train_size, " Removed :", trsize-train_size)
    print("New test size ", test_size, " Removed :",testsize-test_size)
    print("Total example remove: ",trsize+testsize-train_size-test_size)
    print("Total rating ",len(data_rating))

    train_index = np.array(range(train_size))
    test_index = np.array(range(train_size,index))

    np.savetxt(output_dir+'train_index.txt',train_index)
    np.savetxt(output_dir + 'test_index.txt', test_index)

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

    filename=output_dir+"newsgroup20_w2v_vector.mtx"

    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f.write("%d %d %f\n" % (i, j, data_vector[i - 1][j - 1]))

    label_file_name = output_dir + 'newsgroup20_labels.txt'
    with open(label_file_name, 'wb') as f:
        np.savetxt(f, [len(data_rating)], fmt='%d')
    with open(label_file_name, 'a+') as f:
        np.savetxt(f, data_rating, "%d")

    return (data,data_vector,data_rating)

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()
    (data, data_vector, data_rating)=readData(output_dir)
    np.save(output_dir + "data_np.npy", data)
    np.save(output_dir + "data_rating_np", data_rating)

    # data=np.load(output_dir + "data_np.npy")

    TF_IDF_config = {
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }
    from DatasetProcessing.Lib import tf_idf_result
    tf_idf_result(data, TF_IDF_config, output_dir, dataset_name="newsgroup20")

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["newsgroup"]["output_path"])
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
    data_vector=[]
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

    label_file_name = output_dir + 'newsgroup20_labels.txt'
    with open(label_file_name, 'wb') as f:
        np.savetxt(f, [len(data_rating)], fmt='%d')
    with open(label_file_name, 'a+') as f:
        np.savetxt(f, data_rating, "%d")

    return (data,data_vector,data_rating)


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/

def tf_idf(docs):
    TF_IDF_config = {
        'max_features': 5000,
        'ngram': (1, 2)  # min max range
    }
    #tokenizer=tokenize
    tfidf = TfidfVectorizer( min_df=3,
                        max_df=0.90, max_features=TF_IDF_config['max_features'],
                        use_idf=True, sublinear_tf=True, ngram_range=TF_IDF_config['ngram'],
                        norm='l2');
    tfidf.fit(docs)


    X = tfidf.fit_transform(docs)
    print(tfidf.get_feature_names())
    print(X.shape)
    print(type(X).__name__)

    return X;

def tf_idf_result(data,output_dir,dataset_name):
    print(data[0:5])
    data_vector=tf_idf(data)

    m,n = data_vector.shape

    non_zero=int(data_vector.count_nonzero())
    print(non_zero)
    nz=0

    header = np.array([[m, n, non_zero]])
    filename = output_dir + dataset_name+"_tf_idf_vector.mtx"

    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for row, col in zip(*data_vector.nonzero()):
            val = data_vector[row, col]
            f.write("%d %d %f\n" % (row+1, col+1, val))
            nz+=1

    if(nz==non_zero):
        print("Verified")

    return data_vector

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()
    (data, data_vector, data_rating)=readData(output_dir)

    X=tf_idf_result(data, output_dir, "newsgroup20")
    print(X.shape)

    np.save(output_dir + "data_np.npy", data)
    np.save(output_dir + "data_rating_np.npy", data_rating)
    np.save(output_dir+"data_vector_np.npy", X.todense())

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["newsgroup20tfidf"]["output_path"])
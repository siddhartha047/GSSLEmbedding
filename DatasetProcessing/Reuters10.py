import os
import timeit
from nltk.corpus import reuters
import itertools
import numpy as np
from gensim.models import Word2Vec
import gensim
import sys


import nltk
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')

from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

cachedStopWords = stopwords.words("english")

from DatasetProcessing.Lib import processText
min_length = 3

def tokenize(text):
    text=" ".join(text)
    filtered_tokens=processText(text)
    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)

from DatasetProcessing.Path import load_model
model=load_model("GOOGLE")

def readData(output_dir, data_rating, minWordLength=10, readall=True):
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"),documents));
    print(str(len(train_docs)) + " total train documents originally");

    test_docs = list(filter(lambda doc: doc.startswith("test"),documents));
    print(str(len(test_docs)) + " total test documents originally");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories")
    print(categories)
    category_map= dict(zip(categories, np.zeros(len(categories),dtype=int)))
    print(category_map)

    for doc in train_docs:
        if (len(reuters.categories(doc)) > 1): continue
        category=reuters.categories(doc)[0]
        category_map[category]=category_map[category]+1
    for doc in test_docs:
        if (len(reuters.categories(doc)) > 1): continue
        category=reuters.categories(doc)[0]
        category_map[category]=category_map[category]+1

    print(category_map)
    category_map_sorted={k: v for k, v in sorted(category_map.items(), key=lambda item: item[1],reverse=True)}
    print(category_map_sorted)

    category_map_10=dict(itertools.islice(category_map_sorted.items(), 10))
    print(category_map_10)

    with open(output_dir+"original_categories_all.txt","w") as f:
        for k,v in category_map_sorted.items():
            f.write('%s,%d\n'%(k,v))

    model=load_model("GOOGLE")
    i=0
    filename = output_dir + "reuters10_w2v_vector.mtx"

    data_vector=[]
    data=[]
    with open(filename, 'a+') as f:
        for doc in train_docs:
            if (len(reuters.categories(doc)) > 1): continue
            category = reuters.categories(doc)[0]
            if(category not in category_map_10.keys()): continue
            (tokens,status)=tokenize(reuters.words(doc))
            if(status==False):continue
            vocab_tokens = [word for word in tokens if word in model.vocab]
            if(len(vocab_tokens)<min_length):
                print(tokens)
                print(vocab_tokens)
                continue
            i += 1
            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector.append(vector)
            data.append(" ".join(tokens))
            data_rating.append(category)
            if(readall==False and i>minWordLength):
                break

        train_size = i
        print("Training documents: ",train_size)

        i = 0
        for doc in test_docs:
            if (len(reuters.categories(doc)) > 1): continue
            category = reuters.categories(doc)[0]
            if (category not in category_map_10.keys()): continue
            (tokens, status) = tokenize(reuters.words(doc))
            if (status == False): continue
            vocab_tokens = [word for word in tokens if word in model.vocab]
            if (len(vocab_tokens) < min_length ):
                print(tokens)
                print(vocab_tokens)
                continue

            i += 1
            vector = np.mean(model[vocab_tokens], axis=0)
            data.append(" ".join(tokens))
            data_vector.append(vector)
            data_rating.append(category)

            if (readall == False and i > minWordLength):
                break

        test_size = i
        print("Test documents: ",test_size)
        print("Total documents: ",train_size+test_size)

    train_index= np.array(range(train_size))
    test_index = np.array(range(train_size,train_size+test_size))

    if(len(data_rating)!=train_size+test_size):
        print("Error here")
        print("Ratings: ",len(data_rating))
        sys.exit(0)

    np.savetxt(output_dir+'train_index.txt',train_index,"%d")
    np.savetxt(output_dir + 'test_index.txt', test_index,"%d")

    category_map = dict()

    for category in data_rating:
        if (category in category_map.keys()):
            category_map[category] = category_map[category] + 1
        else:
            category_map[category] = 0

    print(category_map)
    with open(output_dir + "categories_top10.txt", "w") as f:
        for k, v in category_map.items():
            f.write('%s,%d\n' % (k, v))

    m=len(data_rating)
    n=300

    header = np.array([[m, n, m * n]])
    filename = output_dir + "reuters10_vector.mtx"

    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f.write("%d %d %f\n" % (i, j, data_vector[i - 1][j - 1]))


    label_file_name=output_dir +'reuters10_labels.txt'
    with open(label_file_name, 'wb') as f:
        np.savetxt(f, [len(data_rating)], fmt='%d')
    with open(label_file_name, 'a+') as f:
        np.savetxt(f,data_rating,"%s")

    return (data,data_vector, data_rating)

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data_rating = []
    print("Started Reading data")
    start_reading = timeit.default_timer()

    (data, data_vector, data_rating)=readData(output_dir, data_rating)
    np.save(output_dir + "data_np.npy", data)
    np.save(output_dir + "data_rating_np", data_rating)

    #data=np.load(output_dir + "data_np.npy")

    TF_IDF_config = {
        'max_features': 5000,
        'ngram': (1, 1)  # min max range
    }
    from DatasetProcessing.Lib import tf_idf_result
    tf_idf_result(data,TF_IDF_config,output_dir,dataset_name="reuters10")
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["reuters10"]["output_path"])
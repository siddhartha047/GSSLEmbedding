import os
import timeit
from nltk.corpus import reuters
import itertools
import numpy as np
import sys


import nltk
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')

from DatasetProcessing.Lib import processText

min_length = 3

def tokenize(text):
    text=" ".join(text)
    filtered_tokens=processText(text)
    if(len(filtered_tokens)<min_length):
        return ("",False)

    return (filtered_tokens,True)

from DatasetProcessing.Path import load_model
model=load_model("GOOGLE")

def readData(output_dir, data_rating, minWordLength=10, readall=False):
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")

    train_docs = list(filter(lambda doc: doc.startswith("train"),documents))
    print(str(len(train_docs)) + " total train documents originally")

    test_docs = list(filter(lambda doc: doc.startswith("test"),documents))
    print(str(len(test_docs)) + " total test documents originally")

    # List of categories
    categories = reuters.categories()
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
    keys = list(category_map_10.keys())
    category_map_index=dict(zip(keys,range(len(keys))))
    print(category_map_index)

    with open(output_dir+"original_categories_all.txt","w") as f:
        for k,v in category_map_sorted.items():
            f.write('%s,%d\n'%(k,v))
    i=0

    data_vector=[]
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
        vector = np.mean(model.wv[vocab_tokens], axis=0)
        data_vector.append(vector)
        data_rating.append(category_map_index[category])
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
        vector = np.mean(model.wv[vocab_tokens], axis=0)
        data_vector.append(vector)
        data_rating.append(category_map_index[category])

        if (readall == False and i > minWordLength):
            break

    test_size = i

    print("Test documents: ",test_size)
    print("Total documents: ",train_size+test_size)

    if(len(data_rating)!=train_size+test_size):
        print("Error here")
        print("Ratings: ",len(data_rating))
        sys.exit(0)

    from DatasetProcessing.Lib import save_labels_txt, save_data_vector_list_mtx
    save_labels_txt(data_rating,output_dir,dataset_name="reuters10_w2v")
    save_data_vector_list_mtx(data_vector,output_dir,dataset_name="reuters10_w2v")

    return (data_vector, data_rating)

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data_rating = []
    print("Started Reading data")
    start_reading = timeit.default_timer()

    (data_vector, data_rating)=readData(output_dir, data_rating)

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",np.array(data_vector))

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["reuters10w2v"]["output_path"])
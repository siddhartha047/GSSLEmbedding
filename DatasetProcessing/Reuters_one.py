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

def tokenize(text):
    min_length = 10
    words = map(lambda word: word.lower(), text);
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));

    return " ".join(filtered_tokens)

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


def readData(output_dir, data_rating, minWordLength=10, readall=True):
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"),documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"),documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories")
    print(categories)
    category_map= dict(zip(categories, np.zeros(len(categories),dtype=int)))
    print(category_map)

    total=0
    for doc in train_docs:
        if (len(reuters.categories(doc)) > 1): continue
        total +=1
        category=reuters.categories(doc)[0]
        category_map[category]=category_map[category]+1
    for doc in test_docs:
        if (len(reuters.categories(doc)) > 1): continue
        total += 1
        category=reuters.categories(doc)[0]
        category_map[category]=category_map[category]+1

    print(category_map)
    category_map_sorted={k: v for k, v in sorted(category_map.items(), key=lambda item: item[1],reverse=True)}
    print(category_map_sorted)

    category_map_10=dict(itertools.islice(category_map_sorted.items(), 10))
    print(category_map_10)

    with open(output_dir+"categories_all.txt","w") as f:
        for k,v in category_map_sorted.items():
            f.write('%s,%d\n'%(k,v))
    with open(output_dir+"categories_top10.txt","w") as f:
        for k,v in category_map_10.items():
            f.write('%s,%d\n'%(k,v))

    print("Total documents will be use: ",total)

    model=load_model("GOOGLE")
    i=0
    index=0

    filename = output_dir + "reuters10_vector.mtx"
    m=total
    n=300

    #data_vector=[]
    data_vector = np.zeros((m, n), dtype=float)

    header = np.array([[m, n, m * n]])
    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    index=0
    with open(filename, 'a+') as f:
        for doc in train_docs:
            if (len(reuters.categories(doc)) > 1): continue
            category = reuters.categories(doc)[0]
            if(category not in category_map_10.keys()): continue
            index+=1
            i+=1
            data_rating.append(category)
            tokens=tokenize(reuters.words(doc))
            vocab_tokens = [word for word in tokens if word in model.vocab]
            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector[index-1]=vector
            for j in range(1,n+1):
                f.write("%d %d %f\n" % (index, j, vector[j - 1]))

            if(readall==False and i>minWordLength):
                break

        train_size = i
        print("Training documents: ",train_size)

        i = 0
        for doc in test_docs:
            if (len(reuters.categories(doc)) > 1): continue
            category = reuters.categories(doc)[0]
            if (category not in category_map_10.keys()): continue
            i += 1
            data_rating.append(category)

            tokens = tokenize(reuters.words(doc))
            vocab_tokens = [word for word in tokens if word in model.vocab]
            vector = np.mean(model[vocab_tokens], axis=0)

            data_vector[index - 1] = vector
            for j in range(1, n + 1):
                f.write("%d %d %f\n" % (index, j, vector[j - 1]))

            if (readall == False and i > minWordLength):
                break

    #print(data_vector)

    test_size = i
    print("Test documents: ",test_size)
    print("Total documents: ",train_size+test_size)

    train_index= np.array(range(len(train_docs)))
    test_index = np.array(range(len(train_docs),len(train_docs)+len(test_docs)))
    np.savetxt(output_dir+'train_index.txt',train_index,"%d")
    np.savetxt(output_dir + 'test_index.txt', test_index,"%d")
    np.savetxt(output_dir +'reuters10_labels.txt',data_rating,"%s")

    return (data_vector, data_rating)


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
    read(dataset_path["reuters10"]["output_path"])
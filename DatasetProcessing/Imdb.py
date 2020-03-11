import os
import timeit
import itertools
import numpy as np
from gensim.models import Word2Vec
import gensim
import sys
import nltk

from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

cachedStopWords = stopwords.words("english")


def num(s):
    try:
        return int(s)
    except ValueError:
        return 5

from DatasetProcessing.Lib import load_model
model=load_model("GOOGLE")

from DatasetProcessing.Lib import processText
def tokenize(text):
    return processText(text)

def readData(directory,data_vector, data_rating,readall=True):
    nrows=20
    min_length = 10

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory+filename, "r", encoding="latin1")
            reviewtext=f.read()
            ratingtext = ((filename.split("."))[0].split("_"))[1]  # split on underscores

            rating = num(ratingtext)
            tokens = tokenize(reviewtext)

            if (int(rating) in [5, 6] and len(tokens) < min_length): continue

            vocab_tokens = [word for word in tokens if word in model.vocab]
            if (len(vocab_tokens) < min_length):
                print(tokens)
                print(vocab_tokens)
                continue

            vector = np.mean(model[vocab_tokens], axis=0)
            data_vector.append(vector)
            data_rating.append(rating)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1


def save_stats(output_dir,data_vector, data_rating):

    np.savetxt(output_dir+"imdb8_labels.txt",data_rating)

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

    filename=output_dir+"imdb8_vector.mtx"

    with open(filename, 'wb') as f:
        np.savetxt(f, header, fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f.write("%d %d %f\n" % (i, j, data_vector[i - 1][j - 1]))

    return (data_vector,data_rating)

def read(home_dir, output_dir):
    input_file1 = home_dir + "train/pos/"
    input_file2 = home_dir + "train/neg/"
    input_file3 = home_dir + "test/pos/"
    input_file4 = home_dir + "test/neg/"

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data_vector = []
    data_rating = []

    print("Started Reading data")
    start_reading = timeit.default_timer()
    readData(input_file1, data_vector, data_rating)
    readData(input_file2, data_vector, data_rating)

    train_size=len(data_rating)
    train_index = np.array(train_size)
    np.savetxt(output_dir+'train_index.txt',train_index)

    readData(input_file3, data_vector, data_rating)
    readData(input_file4, data_vector, data_rating)

    test_size=len(data_rating)-train_size
    test_index = np.array(test_size)
    np.savetxt(output_dir + 'test_index.txt', test_index)

    save_stats(output_dir,data_vector,data_rating)
    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print(data_vector)
    print(data_rating)

    # np.save(output_dir + "data_vector_np", data_vector)
    # np.save(output_dir + "data_rating_np", data_rating)

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["imdb"]["input_path"],dataset_path["imdb"]["output_path"])
from Dataset.Lib import *
import os
import timeit
from nltk.corpus import reuters
import sys
import itertools

import nltk
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')

#resources
#https://martin-thoma.com/nlp-reuters/
#process the documents into following steps

from Dataset.Lib import processText
min_length = 3

def tokenize(text):
    text=" ".join(text)
    filtered_tokens=processText(text)
    if(len(filtered_tokens)<min_length):
        return ("",False)

    return (filtered_tokens,True)

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
    keys = list(category_map_10.keys())
    category_map_index=dict(zip(keys,range(len(keys))))
    print(category_map_index)

    with open(output_dir+"original_categories_all.txt","w") as f:
        for k,v in category_map_sorted.items():
            f.write('%s,%d\n'%(k,v))
    i=0
    data=[]
    for doc in train_docs:
        if (len(reuters.categories(doc)) > 1): continue
        category = reuters.categories(doc)[0]
        if(category not in category_map_10.keys()): continue
        (tokens,status)=tokenize(reuters.words(doc))
        if(status==False):continue
        i += 1
        data.append(" ".join(tokens))
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
        i += 1
        data.append(" ".join(tokens))
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

    return data

def read(home_dir,output_dir,load_saved):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)


    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True
    minWordLength = 10
    readall = True

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        data=readData(output_dir, data_rating, minWordLength, readall)
        stop_reading = timeit.default_timer()
        print('Time to process: ', stop_reading - start_reading)
    else:
        print("Loading Saved data")
        data = np.load(output_dir + "data_np.npy",allow_pickle=True)
        data_rating = np.load(output_dir + "data_rating_np.npy",allow_pickle=True)
        print("Loading Done")

    from Dataset.Lib import save_data_rating_numpy
    save_data_rating_numpy(output_dir, data, data_rating)

    return (data,data_rating)

if __name__ == '__main__':
    read("", "/Users/siddharthashankardas/Purdue/Dataset/Reuters/", False)
    #readData("",[], [], [], 10, False)




# # Words for a document
# document_id = category_docs[0]
# document_words = reuters.words(category_docs[0]);

# #print(categories)
#
# # Documents in a category
# category_docs = reuters.fileids("acq");
#
# # Words for a document
# document_id = category_docs[0]
# document_words = reuters.words(category_docs[0]);
# print(document_words);
#
# # Raw document
# print(reuters.raw(document_id));
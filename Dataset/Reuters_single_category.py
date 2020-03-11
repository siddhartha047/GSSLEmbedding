from Dataset.Lib import *
import os
import timeit
from nltk.corpus import reuters

import nltk
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')

#resources
#https://martin-thoma.com/nlp-reuters/
#process the documents into following steps

from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
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

def readData(output_dir,data, data_rating, minWordLength, readall):
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"),documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"),documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");
    print(categories)

    categories_to_index= dict(zip(categories, range(len(categories))))
    index_to_categories = np.array(categories)

    print(categories_to_index)
    print(index_to_categories)

    # f = open(output_dir+"categories_to_index.pkl", "wb")
    # pickle.dump(categories_to_index, f)
    # f.close()
    # np.save(output_dir + 'index_to_categories', index_to_categories)

    i=0
    for doc in train_docs:
        if (len(reuters.categories(doc)) > 1): continue
        i+=1
        data_rating.append("".join(reuters.categories(doc)))
        data.append(tokenize(reuters.words(doc)))
        if(readall==False and i>minWordLength):
            break

    train_size = i
    print("Training documents: ",train_size)

    i=0
    for doc in test_docs:
        if (len(reuters.categories(doc)) > 1): continue
        i+=1
        data_rating.append("".join(reuters.categories(doc)))
        data.append(tokenize(reuters.words(doc)))
        if(readall==False and i>minWordLength):
            break

    test_size = i
    print("Test documents: ",test_size)

    print("Total documents: ",train_size+test_size)

    train_index= np.array(range(len(train_docs)))
    test_index = np.array(range(len(train_docs),len(train_docs)+len(test_docs)))
    np.savetxt(output_dir+'train_index.txt',train_index)
    np.savetxt(output_dir + 'test_index.txt', test_index)
    np.savetxt(output_dir +'labels.txt',data_rating,"%s")

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
    readall = True

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        (data, data_rating)=readData(output_dir,data, data_rating, minWordLength, readall)
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
    read("", "/Users/siddharthashankardas/Purdue/Dataset/Reuters_onecat/", False)
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
import numpy as np
from sklearn.datasets import fetch_20newsgroups
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

from DatasetProcessing.Path import load_model
model=load_model("GOOGLE")

def readData(output_dir):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    categories=list(newsgroups_train.target_names)
    print(categories)

    trsize=len(newsgroups_train.data)
    testsize=len(newsgroups_test.data)

    print("Training size :",trsize)
    print("Test size :", testsize)

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
            vector = np.mean(model.wv[vocab_tokens], axis=0)
            data_vector.append(vector)
            data_rating.append(newsgroups_train.target[i])
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
            vector = np.mean(model.wv[vocab_tokens], axis=0)
            data_vector.append(vector)
            data_rating.append(newsgroups_test.target[i])
            index += 1
    print("Filtering ended")

    test_size=index-train_size
    print("New train size ", train_size, " Removed :", trsize-train_size)
    print("New test size ", test_size, " Removed :",testsize-test_size)
    print("Total example remove: ",trsize+testsize-train_size-test_size)
    print("Total rating ",len(data_rating))

    if(len(data_rating)==(train_size+test_size)):
        print("Data size correct")
    else:
        print("something wrong")

    from DatasetProcessing.Lib import save_labels_txt,save_data_vector_list_mtx
    save_labels_txt(data_rating,output_dir,dataset_name="newsgroup20_w2v")
    save_data_vector_list_mtx(data_vector,output_dir,dataset_name="newsgroup20_w2v")

    return (data_vector,data_rating)

def read(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Started Reading data")
    start_reading = timeit.default_timer()
    (data_vector, data_rating)=readData(output_dir)

    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",np.array(data_vector))

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["newsgroup20w2v"]["output_path"])
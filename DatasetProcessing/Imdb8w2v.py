import os
import timeit
import itertools
import numpy as np
import sys

def num(s):
    try:
        return int(s)
    except ValueError:
        return 5

from DatasetProcessing.Path import load_model
model=load_model("GLOVE")

from DatasetProcessing.Lib import processText
min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)

def readData(directory, data_vector, data_rating,readall=True):
    nrows=20
    min_length = 3

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory+filename, "r", encoding="latin1")
            reviewtext=f.read()
            ratingtext = ((filename.split("."))[0].split("_"))[1]  # split on underscores

            rating = num(ratingtext)
            tokens,status = tokenize(reviewtext)
            if(status==False):continue
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
    readData(input_file3, data_vector, data_rating)
    readData(input_file4, data_vector, data_rating)

    from DatasetProcessing.Lib import save_labels_txt, save_data_vector_list_mtx
    save_labels_txt(data_rating, output_dir, dataset_name="imdb8_w2v")
    save_data_vector_list_mtx(data_vector, output_dir, dataset_name="imdb8_w2v")

    stop_reading = timeit.default_timer()
    print('Time to process: ', stop_reading - start_reading)

    print("Total:",len(data_rating))
    # np.save(output_dir + "data_rating_np.npy", data_rating)
    # np.save(output_dir+"data_vector_np.npy",np.array(data_vector))

    return (data_vector,data_rating)

if __name__ == '__main__':
    from DatasetProcessing.Path import dataset_path
    read(dataset_path["imdb8w2v"]["input_path"],dataset_path["imdb8w2v"]["output_path"])
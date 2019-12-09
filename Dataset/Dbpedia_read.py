from Dataset.Lib import *
import os
import timeit


def readData(filename,data,data_vector,data_rating,minWordLength,readall=False):
    nrows=20
    with open(filename,encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            rating=num(row['stars'])
            review = processText(row['text'])

            if(int(rating)!=3 and len(review)>minWordLength):
                data.append(" ".join(review))
                data_rating.append(rating)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1



def read(home_dir,output_dir,load_saved):
    print(os.uname())

    input_file1 = home_dir + "train.csv"
    input_file2 = home_dir + "test.csv"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    data_vector = []
    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True
    minWordLength = 10
    readall = False

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        readData(input_file1, data, data_vector, data_rating, minWordLength, readall)
        readData(input_file2, data, data_vector, data_rating, minWordLength, readall)
        stop_reading = timeit.default_timer()
        print('Time to process: ', stop_reading - start_reading)
    else:
        print("Loading Saved data")
        data = np.load(output_dir + "data_np.npy")
        data_rating = np.load(output_dir + "data_rating_np.npy")
        data_vector = np.load(output_dir + "data_vector_np.npy")
        print("Loading Done")

    return (data,data_rating,data_vector)
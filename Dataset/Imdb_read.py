from Dataset.Lib import *
import os
import timeit

def readData(directory,data,data_rating,minWordLength,readall=False):
    nrows=20

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory+filename, "r", encoding="latin1")
            reviewtext=f.read()
            ratingtext = ((filename.split("."))[0].split("_"))[1]  # split on underscores

            rating = num(ratingtext)
            review = processTextParagraph(reviewtext)

            if (int(rating) != 5.0 and len(review)>minWordLength):
                data.append(review)
                data_rating.append(rating)

            if (readall == False):
                if (nrows < 0):
                    break
                nrows -= 1



def read(home_dir,output_dir,load_saved):
    print(os.uname())

    input_file1 = home_dir + "train/pos/"
    input_file2 = home_dir + "train/neg/"
    input_file3 = home_dir + "test/pos/"
    input_file4 = home_dir + "test/neg/"

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
        readData(input_file1, data, data_rating, minWordLength, readall)
        readData(input_file2, data, data_rating, minWordLength, readall)
        readData(input_file3, data, data_rating, minWordLength, readall)
        readData(input_file4, data, data_rating, minWordLength, readall)
        stop_reading = timeit.default_timer()
        print('Time to process: ', stop_reading - start_reading)
    else:
        print("Loading Saved data")
        data = np.load(output_dir + "data_np.npy")
        data_rating = np.load(output_dir + "data_rating_np.npy")
        print("Loading Done")

    from Dataset.Lib import save_data_rating_numpy
    save_data_rating_numpy(output_dir, data, data_rating)

    return (data,data_rating)
from Dataset.Lib import *
import os
import timeit
import csv

def readData(fileName,data,data_rating,minWordLength,readall=False):
    nrows = 20

    with open(fileName, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            if len(row) == 3:
                label=num(row[0])
                text=processText(row[1] + row[2])

                if(len(text)>minWordLength):
                    data_rating.append(label)  # 0 is class label
                    data.append(" ".join(text))  # 1 is title 2 is abstract
                    #print(" ".join(text))
            else:
                print("improper format\n")

            if readall==False:
                if (nrows < 0):
                    break
                nrows-=1



def read(home_dir,output_dir,load_saved):
    print(os.uname())

    input_file1 = home_dir + "yelp_review.csv"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True
    minWordLength = 10
    readall = False

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        readData(input_file1, data, data_rating, minWordLength, readall)
        stop_reading = timeit.default_timer()
        print('Time to process: ', stop_reading - start_reading)
    else:
        print("Loading Saved data")
        data = np.load(output_dir + "data_np.npy")
        data_rating = np.load(output_dir + "data_rating_np.npy")
        print("Loading Done")

    return (data,data_rating)
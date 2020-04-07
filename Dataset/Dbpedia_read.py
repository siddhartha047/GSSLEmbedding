from Dataset.Lib import *
import os
import timeit
import csv


min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)
    if(len(filtered_tokens)<min_length):
        return ("",False)
    return (filtered_tokens,True)

def num(s):
    try:
        return int(s)
    except ValueError:
        return -1

def readData(filename,data,data_rating,readall=True):
    nrows=20
    with open(filename,newline='',encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            #print(row)
            category=num(row[0])
            if(category==-1):continue
            title=row[1]
            description=row[2]
            #print(title+description)
            tokens,status=tokenize(title+" "+description)
            #print(tokens)

            if (status == False): continue
            data.append(" ".join(tokens))
            data_rating.append(category)

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
    data_rating = []

    # ignores rating 3, review with text length less than140
    # to read all pass True
    minWordLength = 10
    readall = False

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        readData(input_file1, data, data_rating, readall)
        readData(input_file2, data, data_rating, readall)
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
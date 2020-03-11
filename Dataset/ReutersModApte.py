from Dataset.Lib import *
import os
import timeit
import scipy.io


def readData(home_dir, output_dir,data, data_rating, minWordLength, readall):
    # List of documents
    print(home_dir)
    print(output_dir)
    mat = scipy.io.loadmat(home_dir+'Reuters21578.mat')

    print(mat.fe)


    return (data, data_rating)


def read(home_dir,output_dir,load_saved):
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    data = []
    data_rating = []

    minWordLength = 10
    readall = True

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        (data, data_rating)=readData(home_dir,output_dir,data, data_rating, minWordLength, readall)
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

    read("/Users/siddharthashankardas/Purdue/Dataset/ReutersModApte/", "/Users/siddharthashankardas/Purdue/Dataset/ReutersModApte/", False)

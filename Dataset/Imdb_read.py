from Dataset.Lib import *
import os
import timeit

def num(s):
    try:
        return int(s)
    except ValueError:
        return 5

from Dataset.Lib import processText
min_length = 3

def tokenize(text):
    filtered_tokens=processText(text)

    if(len(filtered_tokens)<min_length):
        # print(text)
        # print(filtered_tokens)
        return ("",False)

    return (filtered_tokens,True)

def readData(directory, data, data_rating,readall=True):
    nrows=5000
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

            data_rating.append(rating)
            data.append(" ".join(tokens))

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
    readall = False

    if (load_saved==False or os.path.exists(output_dir + "data_np.npy") == False):
        print("Started Reading data")
        start_reading = timeit.default_timer()
        readData(input_file1, data, data_rating, readall)
        readData(input_file2, data, data_rating, readall)
        readData(input_file3, data, data_rating, readall)
        readData(input_file4, data, data_rating, readall)
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


def data_G_imdb(config):
    import scipy as sp
    import scipy.sparse
    from collections import namedtuple
    import networkx as nx

    dataset_name=config['dataset_name']
    output_dir=config['input_path']
    graph_dir=config['graph_algorithm']+"/"
    k = 5
    train_percent=config['train']
    val_percent=config['val']


    print("Loading Saved data")
    data_vector = np.load(output_dir + "data_vector_np.npy")
    data_rating = np.load(output_dir + "data_rating_np.npy").astype(int)
    data_graph=sp.sparse.load_npz(output_dir+graph_dir+"graph_knn_"+str(k)+".npz")
    print("Loading Done")

    # print(data_vector.shape)
    # print(data_rating)
    # print(data_graph)

    data_rating= np.array([int(i>5) for i in data_rating])
    unique_rating=np.unique(data_rating)
    # print(data_rating)
    # print(unique_rating)
    n = len(data_rating)
    n_class=len(unique_rating)

    one_hot_label = np.zeros((n, n_class), dtype=float)
    for i in range(n):
        one_hot_label[i,data_rating[i]] = 1.0

    portion= int(n / 4)
    training_size=int(n*train_percent/4)
    val_size = int(n * val_percent / 4)

    train_index=[]
    val_index=[]

    for i in range(4):
        train_index.extend(range(i*portion,i*portion+training_size))
        val_index.extend(range(i*portion+training_size,i * portion + training_size+val_size))

    test_index=[i for i in range(n) if (i not in train_index) and (i not in val_index)]
    # print(train_index)
    # print(val_index)
    # print(test_index)
    #
    # print(n)
    # print(len(train_index))
    # print(len(val_index))
    # print(len(test_index))

    graph=nx.from_scipy_sparse_matrix(data_graph,parallel_edges=False)

    # print(graph.edges())
    # print(graph[0][163])

    label_mask = np.zeros(n, dtype=float)
    for i in train_index:
        label_mask[i] = 1

    Dataset = namedtuple('Dataset', field_names=['train_index', 'test_index', 'val_index', 'Graph', 'Feature', 'Label',
                                                 'Label_mask', 'classname'])

    data = Dataset(train_index=train_index, test_index=test_index, val_index=val_index, Graph=graph,
                   Feature=data_graph.todense(), Label=one_hot_label, Label_mask=label_mask, classname=unique_rating)

    return data

if __name__ == '__main__':
    config={"dataset_name":"imdb",
            "input_path":"/Users/siddharthashankardas/Purdue/Dataset/Imdb/aclImdb/TF_IDF/",
            "graph_algorithm":"knn",
            "train": 0.1,
            "val": 0.2};
    data_G_imdb(config)
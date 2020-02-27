import os
import scipy.sparse
from sklearn.metrics import  pairwise_distances
import multiprocessing
import numpy as np
import sys

def csr2graph(dataset_info,GRAPH_config,config):
    data_dir = dataset_info['output_path'] + GRAPH_config['method'] + '/'
    print(data_dir)
    output_dir = data_dir + '/complete/'

    if not os.path.exists(output_dir):
        print("Creating directory: ", output_dir)
        os.makedirs(output_dir)

    print("Loading Saved data")
    data_vector = scipy.sparse.load_npz(data_dir + "data_vector.npz")
    print("Loading Done....")
    print(data_vector.shape)

    worker = multiprocessing.cpu_count()
    print("number of processors: ", worker)

    print("using Metric ",config["metric"])

    W = pairwise_distances(data_vector, n_jobs=worker, metric=config["metric"])

    print("Adjusting weight: (1-W)")
    W = 1 - W
    print(W.shape)
    np.savetxt(output_dir+dataset_info['name']+"_weights.txt", W)
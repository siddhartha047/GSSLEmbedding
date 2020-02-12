import numpy as np
import scipy as sp
from scipy import io
from sklearn.neighbors import kneighbors_graph
import sys
import os
import timeit


# from scipy.sparse import csr_matrix
# from graphviz import Digraph,Graph


def knn(x,k,mode='distance',metric='cosine',include_self=True):
    A = kneighbors_graph(x, k, mode=mode, metric=metric,include_self=include_self)

    if (mode == 'distance'):
        # A_others.data = 1.0 / (1.0 + A_others.data)  # poincare
        A.data = 1.0 - A.data
        A.eliminate_zeros()

    return A

def knn_single(data_vector,data_rating,k,KNN_config,GRAPH_config,output_dir):
    print("Constructing graph using Knn with k= ", k)

    if (data_rating.shape[0] < k):
        print("k={0} has to be smaller than N={1}".format(k, N))
        return False

    A = knn(data_vector, k, KNN_config['mode'], KNN_config['metric'], KNN_config['include_self'])
    print('Saving graph ----', GRAPH_config['saving_format'], ' format')

    if ('numpy' in GRAPH_config['saving_format']):
        sp.sparse.save_npz(output_dir + 'graph_knn_' + str(k) + '.npz', A)
    if ('mat' in GRAPH_config['saving_format']):
        io.savemat(output_dir + 'graph_knn_' + str(k) + '.mat', mdict={'data': A})
    if ('mtx' in GRAPH_config['saving_format']):
        io.mmwrite(output_dir + 'graph_knn_' + str(k) + '.mtx', A, comment='Sparse Graph')
    if ('gephi' in GRAPH_config['saving_format']):
        save_gephi_graph(output_dir, A, data_rating, k,GRAPH_config['multi_label'])
    if ('txt' in GRAPH_config['saving_format']):
        print("text format saving is not implemented yet")
        # np.savetxt(output_dir+'graph_knn_'+str(k)+'.txt', A, delimiter='\t')

    print('Graph saving Done for ',k)

    return True

def kmatch_wrapper(args):
    return knn_single(*args)

def KNN_construction(dataset_info,GRAPH_config,KNN_config):
    data_dir = dataset_info['output_path'] + GRAPH_config['method'] + '/'
    print(data_dir)
    output_dir=data_dir+'/knn/'

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Loading Saved data")
    #data = np.load(output_dir + "data_np.npy")
    print(data_dir + "data_rating_np.npy")
    data_rating = np.load(data_dir + "data_rating_np.npy",allow_pickle=True)
    data_vector = np.load(data_dir + "data_vector_np.npy",allow_pickle=True)
    print("Loading Done....")
    print("Data vector shape: ",data_vector.shape)
    print("Data rating shape: ",data_rating.shape)

    start_time = timeit.timeit()

    from multiprocessing import Pool
    k_num = len(KNN_config['k'])
    p = Pool(k_num)
    arguments = [(data_vector,data_rating,k,KNN_config,GRAPH_config,output_dir) for k in  KNN_config['k']]
    print(p.map(kmatch_wrapper, arguments))

    #(knn_single(data_vector,data_rating,k,KNN_config,GRAPH_config,output_dir) for k in KNN_config['k'])

    end_time = timeit.timeit()

    print("Total time: ", end_time - start_time)

def save_gephi_graph(output_dir,A,y,k,multi_label=False):
    import networkx as nx

    labels=[]
    if(multi_label):
        nY= [" ".join(row) for row in y]
        labels = dict(zip(range(len(y)), nY))
    else:
        labels = dict(zip(range(len(y)), y))

    G = nx.from_scipy_sparse_matrix(A)
    # print(G.edges())
    # G=G.to_directed()
    # print(G.edges())

    nx.set_node_attributes(G, labels, 'labels')
    print("Writing gephi")
    nx.write_gexf(G, output_dir+'graph_knn_'+str(k)+'.gexf')

    return

if __name__ == '__main__':


    x = np.array([[1, 2],
                  [2, 4],
                  [4, 6],
                  [1, 0],
                  [2, 0],
                  [3, 0],
                  [-1, -2],
                  [-2, -4],
                  [-4,-6]])

    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]

    k=3
    mode='distance'
    metric='cosine'
    include_self=True
    A=knn(x,k,mode,metric,include_self)

    print(A)

    save_gephi_graph("",A,y,k)


    #KNN_construction()
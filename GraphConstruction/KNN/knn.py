import numpy as np
import scipy as sp
from scipy import io
from sklearn.neighbors import kneighbors_graph
import sys

# from scipy.sparse import csr_matrix
# from graphviz import Digraph,Graph


def knn(x,k,mode='distance',metric='cosine',include_self=True):
    A = kneighbors_graph(x, k, mode=mode, metric=metric,include_self=include_self)

    if (mode == 'distance'):
        # A_others.data = 1.0 / (1.0 + A_others.data)  # poincare
        A.data = 1.0 - A.data
        A.eliminate_zeros()

    return A

def KNN_construction(dataset_info,GRAPH_config,KNN_config,save_gephi=False):
    dataset_name = dataset_info['name']
    home_dir = dataset_info['path']
    output_dir = dataset_info['output_path'] + GRAPH_config['method'] + '/'
    print(output_dir)

    print("Loading Saved data")
    #data = np.load(output_dir + "data_np.npy")
    data_rating = np.load(output_dir + "data_rating_np.npy")
    data_vector = np.load(output_dir + "data_vector_np.npy")
    print("Loading Done....")
    print("Data vector shape: ",data_vector.shape)

    print("Constructing graph using Knn with k= ",KNN_config['k'])
    A=knn(data_vector,KNN_config['k'],KNN_config['mode'],KNN_config['metric'],KNN_config['include_self'])

    print('Saving graph ----',GRAPH_config['saving_format'],' format')

    if(GRAPH_config['saving_format']=='numpy'):
        sp.sparse.save_npz(output_dir+'graph_knn_'+str(KNN_config['k'])+'.npz', A)
    elif(GRAPH_config['saving_format']=='mat'):
        io.savemat(output_dir+'graph_knn_'+str(KNN_config['k'])+'.mat', mdict={'data': A})
    elif (GRAPH_config['saving_format'] == 'mtx'):
        io.mmwrite(output_dir+'graph_knn_'+str(KNN_config['k']), A, comment='Sparse Graph')
    else:
        print("This graph saving format is not implemented yet")
        sys.exit(0)
    print('Graph saving Done')


    if(save_gephi==True):
        save_gephi_graph(output_dir,A,data_rating,KNN_config['k'])

    return A

def save_gephi_graph(output_dir,A,y,k):
    import networkx as nx

    labels = dict(zip(range(len(y)), y))
    G = nx.from_scipy_sparse_matrix(A)
    # print(G.edges())
    # G=G.to_directed()
    # print(G.edges())

    nx.set_node_attributes(G, labels, 'labels')
    print("Writing gephi")
    nx.write_gexf(G, output_dir+'graph_'+str(k)+'.gexf')

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
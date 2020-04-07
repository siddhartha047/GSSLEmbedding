import numpy as np
from scipy.sparse import  csr_matrix,save_npz
from sklearn.metrics import  pairwise_distances
import multiprocessing
import sys

def text2mtx(weight_input_filename,edge_input_filename,edge_output_filename):
    data=np.loadtxt(weight_input_filename)
    #data=np.random.rand(8293,8293)

    print(data.shape)
    dm,dn=data.shape

    edges=np.loadtxt(edge_input_filename).astype(int)
    #edges=np.array(list(zip(np.random.randint(0,8293,100),np.random.randint(0,8293,100))))
    print(edges.shape)
    m,n=edges.shape
    edge_weight=np.zeros((m+1,3))
    print(edge_weight.shape)
    edge_weight[0, 0] = dm
    edge_weight[0, 1] = dm
    edge_weight[0, 2] = m

    for i in range(m):
        edge_weight[i + 1, 0] = edges[i, 0]+1
        edge_weight[i + 1, 1] = edges[i, 1]+1
        edge_weight[i + 1, 2] = data[edges[i,0], edges[i,1]]

    print(edge_weight)
    np.savetxt(edge_output_filename,edge_weight,fmt='%d %d %lf')


    return
if __name__ == '__main__':
    weight_input_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/reuters_full_weights_cosine.txt'
    edge_input_filename = '/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/full_reuters_full_weights_20.txt'
    edge_output_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/full_reuters_full_weights_20.mtx'
    text2mtx(weight_input_filename,edge_input_filename,edge_output_filename)
import numpy as np
from scipy.sparse import  csr_matrix,save_npz
from sklearn.metrics import  pairwise_distances
import multiprocessing

def text2mtx(weight_input_filename,edge_input_filename,edge_output_filename):
    data=np.loadtxt(weight_input_filename)
    print(data.shape)
    dm,dn=data.shape

    edges=np.loadtxt(edge_input_filename).astype(int)

    print(edges[1,0])
    print(edges[1, 1])
    print(data[edges[1,0],edges[1,1]])



    return
if __name__ == '__main__':
    weight_input_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/reuters_full_weights.txt'
    edge_input_filename = '/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/snyder_reuters_full_weights_10.txt'
    edge_output_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/snyder_reuters_full_edge_weights_10.mtx'
    text2mtx(weight_input_filename,edge_input_filename,edge_output_filename)
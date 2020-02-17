import numpy as np
from scipy.sparse import  csr_matrix
from sklearn.metrics import  pairwise_distances
import multiprocessing

def mtxtotext(input_filename,output_filename):
    #print(io.mmread(input_filename))
    data=np.loadtxt(input_filename)
    header=data[0,:].astype(int)
    row_ind=data[1:,0].astype(int)-1
    col_ind=data[1:,1].astype(int)-1
    value=data[1:,2].astype(np.float64)

    print(row_ind[0:20])
    print(col_ind[0:20])
    print(value[0:5])

    print(header)

    print(row_ind.shape)
    print(col_ind.shape)
    print(value.shape)

    print(max(row_ind))
    print(max(col_ind))

    X=csr_matrix((value, (row_ind, col_ind)), shape = (header[0], header[1]),dtype=np.float64)
    print(X.shape)

    worker=multiprocessing.cpu_count()

    print("number of processors: ",worker)

    W=pairwise_distances(X,n_jobs=worker, metric="euclidean")
    print(W.shape)
    np.savetxt(output_filename,W)

    return
if __name__ == '__main__':
    input_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/reuters_full_tfidf.mtx'
    output_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/reuters_full_weights_euclidean.txt'
    mtxtotext(input_filename,output_filename)
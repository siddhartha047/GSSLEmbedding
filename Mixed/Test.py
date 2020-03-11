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


def array2mtx(data_vector):
    print(data_vector)
    filename='data_vector.mtx'

    m, n = data_vector.shape
    header=np.array([[m, n, m * n]])

    with open(filename, 'wb') as f:
        np.savetxt(f,header , fmt='%d %d %d')

    with open(filename, 'a+') as f:
        for i in range(1,m+1):
            for j in range(1,n+1):
                f.write("%d %d %f\n" %(i,j,data_vector[i-1,j-1]))

    return

def mydist(x1,x2):
    # d= np.sum(np.power(x1-x2,2))
    # return np.sqrt(d)
    # n1= np.sqrt(np.sum(np.power(x1,2)))
    # n2 = np.sqrt(np.sum(np.power(x2, 2)))
    # d= np.dot(x1,x2)/(n1*n2)
    # return 1-d

    n1= np.sqrt(np.sum(np.power(x1,2)))
    n2 = np.sqrt(np.sum(np.power(x2, 2)))
    d= np.sqrt(np.sum(np.power(x1-x2,2)))

    return np.arccosh(
        1 + 2 * ((d ** 2) / ((n1 ** 2) * (n2 ** 2)))
    )



def testKNN():
    np.random.seed(123)
    import sklearn
    from sklearn.neighbors import kneighbors_graph
    x=np.random.rand(10,2)

    k=2
    mode="distance"
    metric=mydist
    include_self=True

    A = kneighbors_graph(x, k, mode=mode, metric=metric, include_self=include_self)

    print(A)


if __name__ == '__main__':
    # weight_input_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/reuters_full_weights.txt'
    # edge_input_filename = '/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/snyder_reuters_full_weights_10.txt'
    # edge_output_filename='/Users/siddharthashankardas/Purdue/Dataset/ReutersMTX/snyder_reuters_full_edge_weights_10.mtx'
    # text2mtx(weight_input_filename,edge_input_filename,edge_output_filename)

    # data_vector=np.array([[1.0, 2.0],[3.0,4.0]])
    # array2mtx(data_vector)
    testKNN()

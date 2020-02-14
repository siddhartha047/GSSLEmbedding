import os
import sys
import subprocess
import numpy as np
import scipy as sp
from scipy import io
from Path import executable
import timeit

def read_txt(output_edges):
    with open(output_edges, 'r') as f:
        data = f.readlines()
        data=[list(map(int,(line.strip().split()))) for line in data]
        data = np.array(data)
        return data

def write_mtx(output_edges):
    data=read_txt(output_edges)
    #print(data)
    print("Edges: ",data.shape)
    newfileName=os.path.splitext(output_edges)[0]+".mtx"
    io.mmwrite(newfileName, data, comment="edge list")
    #print(read_mtx(newfileName))

def read_mtx(output_edges):
    data=io.mmread(output_edges)
    return data


def bmatch_weight_matrix(weight_matrix,b_degree,output_edges,N,Cache,max_iterations=-1,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    args=''
    if(max_iterations==-1):
        args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose))
    else:
        args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges, "-c", str(Cache), "-v",str(verbose),"-i",str(max_iterations))
    # popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    # popen.wait()
    # output = popen.stdout.read()
    # print(output.decode())
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()
    print(return_code)

def bmatch_descriptor(feature_matrix,b_degree,output_edges,N,Cache,Dimension,max_iterations=-1,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    print("Maximum iteration: ",max_iterations)
    args=""
    if(max_iterations==-1):
        args = (executable, "-x", feature_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose),"-t","1","-D",str(Dimension))
    else:
        args = (executable, "-x", feature_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose),"-t","1","-D",str(Dimension),"-i",str(max_iterations))
    # popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    # return_code=popen.wait()
    # output = popen.stdout.read()
    # print(output.decode())

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()
    print(return_code)

    return return_code


def bmatch_single(output_dir,b,data_dir,GRAPH_config,data_rating,D,N,max_iterations):
    print("Constructing graph using b-matching with b= ", b)

    if(N<=b):
        print("b={0} has to be smaller than N={1}".format(b,N))
        return False

    b_degree = output_dir + 'b_degree_' + str(b) + '.txt'
    degree = np.ones(N, int) * b
    np.savetxt(b_degree, degree, delimiter='\n', fmt='%d')
    feature_matrix = data_dir + 'data_vector_txt.txt'
    output_edges = output_dir + 'graph_bmatch_' + str(b) + '.txt'
    #Cache = b
    #N=1000
    Cache = min(b,N-1)
    #Cache = N-1
    Dimension = D
    return_code=bmatch_descriptor(feature_matrix, b_degree, output_edges, N, Cache, Dimension, max_iterations, verbose=1)

    if(return_code!=0):
        return False

    print('Saving graph ----', GRAPH_config['saving_format'], ' format')
    data = read_txt(output_edges)

    if ('numpy' in GRAPH_config['saving_format']):
        np.save(output_dir + 'graph_bmatch_' + str(b) + '.npy', data)
    if ('mat' in GRAPH_config['saving_format']):
        io.savemat(output_dir + 'graph_bmatch_' + str(b) + '.mat', mdict={'data': data})
    if ('mtx' in GRAPH_config['saving_format']):
        io.mmwrite(output_dir + 'graph_bmatch_' + str(b) + '.mtx', data, comment='Edge list')
    if ('gephi' in GRAPH_config['saving_format']):
        save_gephi_graph(output_dir, data, data_rating, b,GRAPH_config['multi_label'])

    print('Graph saving Done for ',b)

    if os.path.exists(b_degree):
        os.remove(b_degree)
        print("Degree file deleted")
    else:
        print("degree file does not exist")

    return True

def bmatch_wrapper(args):
    return bmatch_single(*args)

def bmatch_construction(dataset_info,GRAPH_config,bMatching_config):
    data_dir = dataset_info['output_path'] + GRAPH_config['method'] + '/'
    print(data_dir)
    output_dir=data_dir+'/bmatch/'

    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Loading Saved data")
    data_rating = np.load(data_dir + "data_rating_np.npy",allow_pickle=True)
    data_vector = np.load(data_dir + "data_vector_np.npy",allow_pickle=True)
    print("Data rating shape :",data_rating.shape)
    print("Data vector shape :", data_vector.shape)
    print("Loading Done....")

    N=data_rating.shape[0]
    D=data_vector.shape[1]

    start_time=timeit.timeit()

    from multiprocessing import Pool
    b_num=len(bMatching_config)
    p = Pool(b_num)
    arguments = [(output_dir, bMatching_config['b'][biter], data_dir, GRAPH_config, data_rating, D, N,bMatching_config['max_iterations'][biter]) for biter in range(len(bMatching_config['b']))]
    print(p.map(bmatch_wrapper, arguments))

    #bmatch_single(output_dir,b,data_dir,GRAPH_config,data_rating,D,N)) for b in bMatching_config['b']

    end_time = timeit.timeit()

    print("Total time: ",end_time-start_time)

def save_gephi_graph(output_dir,A,y,k,multi_label=False):
    import networkx as nx

    labels=[]
    if(multi_label):
        nY= [" ".join(row) for row in y]
        labels = dict(zip(range(len(y)), nY))
    else:
        labels = dict(zip(range(len(y)), y))

    #G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_edgelist(A)
    # print(G.edges())
    # G=G.to_directed()
    # print(G.edges())

    nx.set_node_attributes(G, labels, 'labels')
    print("Writing gephi")
    nx.write_gexf(G, output_dir+'graph_knn_'+str(k)+'.gexf')

    return



def bmatch_test():
    b_degree='test/uni_example_degrees.txt'
    weight_matrix='test/uni_example_weights.txt'
    output_edges='test/uni_example_ssolution.txt'

    #Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    #os.system('Release/BMatchingSolver -w 1test_data/uni_example_weights.txt -d 1test_data/uni_example_degrees.txt -n 10 -o 1test_data/uni_example_solution.txt -c 5 -v 1')


    args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", "10", "-o", output_edges,"-c","5","-v","1")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output.decode())


if __name__ == '__main__':

   # b_degree = 'test/uni_example_degrees.txt'
   # weight_matrix = 'test/uni_example_weights.txt'
   # output_edges = 'test/uni_example_ssolution.txt'
   # N=10
   # Cache=10 #keep it degree-1 (of course less than N)
   # bmatch_weight_matrix(weight_matrix,b_degree,output_edges,N,Cache)

   b_degree = 'test/test_degree.txt'
   feature_matrix = 'test/test_feature.txt'
   output_edges = 'test/test_solution.txt'
   N = 5
   Cache = 5  # keep it degree-1 (of course less than N)
   Dimension=2
   bmatch_descriptor(feature_matrix, b_degree, output_edges, N, Cache,Dimension)

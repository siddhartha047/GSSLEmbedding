import os
import sys
import subprocess
import numpy as np
import scipy as sp
from scipy import io
from DatasetProcessing.Path import executable,dataset_path
import timeit
from multiprocessing import Pool

loaded=False
M=np.array([])

def load_weights(filename):
    global loaded, M
    if(loaded==False):
        print("loading first time")
        M = np.loadtxt(filename)
        print(M.shape)
        loaded=True
    else:
        print("used preloaded")
    return M

def load_file(filename):
    M = np.loadtxt(filename)
    print(M.shape)
    return M

def read_txt(output_edges):
    with open(output_edges, 'r') as f:
        data = f.readlines()
        data=[list(map(int,(line.strip().split()))) for line in data]
        data = np.array(data)
        return data

def create_degree_file(b, N,degree_file):
    degree = np.ones(N, int) * b
    np.savetxt(degree_file, degree, delimiter='\n', fmt='%d')

def cosine_distance(x1,x2):
    n1= np.sqrt(np.sum(np.power(x1,2)))
    n2 = np.sqrt(np.sum(np.power(x2, 2)))
    d= np.dot(x1,x2)/(n1*n2)
    return 1-d

def text2mtx(output_dir,weight_file,edge_input_filename,edge_output_filename,labels,b,use_weight=True,save_gephi=True):
    W=load_weights(weight_file)
    dm,dn=W.shape
    print(dm,dn)

    edges=np.loadtxt(edge_input_filename).astype(int)
    print(edges.shape)
    m,n=edges.shape

    with open(edge_output_filename,'w+') as f:
        np.savetxt(f,np.array([[dm, dm, m]]),fmt='%d %d %d')

    edge_weight=np.zeros((m,3))
    print(edge_weight.shape)

    for i in range(m):
        edge_weight[i, 0] = edges[i, 0]+1
        edge_weight[i, 1] = edges[i, 1]+1

        if(use_weight==True):
            edge_weight[i, 2] = W[edges[i,0], edges[i,1]]
        else:
            edge_weight[i, 2] = cosine_distance(W[edges[i, 0],:], W[edges[i, 1],:])

    #print(edge_weight)
    with open(edge_output_filename,'a+') as f:
        np.savetxt(f,edge_weight,fmt='%d %d %f')


    if(save_gephi==True):
        edge_list=read_txt(edge_input_filename)
        save_gephi_graph(output_dir,edge_list,labels,b)

    return True

def save_gephi_graph(output_dir, edge_weight, y, k, multi_label=False):

    import networkx as nx

    labels = []
    if (multi_label):
        nY = [" ".join(row) for row in y]
        labels = dict(zip(range(len(y)), nY))
    else:
        labels = dict(zip(range(len(y)), y))

    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_edgelist(edge_weight)
    # print(G.edges())
    # G=G.to_directed()
    # print(G.edges())

    nx.set_node_attributes(G, labels, 'labels')
    print("Writing gephi")
    nx.write_gexf(G, output_dir + 'graph_bmatch_' + str(k) + '.gexf')

    return

def bmatch_weight_matrix(weight_matrix,b_degree,output_edges,N,Cache,output_log,max_iterations=-1,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    print("Max iteration: ",max_iterations)
    if(max_iterations==-1):
        args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose))
    else:
        args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges, "-c", str(Cache), "-v",str(verbose),"-i",str(max_iterations))

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    f=open(output_log,'w+')
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
        f.write("%s" % (stdout_line))
    popen.stdout.close()
    f.close()
    return_code = popen.wait()
    print(return_code,end='')

    if (return_code != 0):
        return False

    return True

def bmatch_descriptor(feature_matrix,b_degree,output_edges,N,Cache,Dimension,output_log,max_iterations=-1,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    print("Maximum iteration: ",max_iterations)
    args=""
    if(max_iterations==-1):
        args = (executable, "-x", feature_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose),"-t","1","-D",str(Dimension))
    else:
        args = (executable, "-x", feature_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose),"-t","1","-D",str(Dimension),"-i",str(max_iterations))

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    f = open(output_log, 'w+')
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
        f.write("%s" % (stdout_line))
    popen.stdout.close()
    f.close()
    return_code = popen.wait()
    print(return_code, end='')

    if (return_code != 0):
        return False

    return True

def bmatch_weight_wrapper(args):
    return bmatch_weight_matrix(*args)

def bmatch_feature_wrapper(args):
    return bmatch_descriptor(*args)


def mtx_wrapper(args):
    return text2mtx(*args)

if __name__ == '__main__':
   dataset='newsgroup20tfidf'

   dataset_name=dataset_path[dataset]['name']
   use_weight=True

   input_dir=dataset_path[dataset]['output_path']
   output_dir=input_dir+'bmatch/'

   if not os.path.exists(output_dir):
       print("Creating directory: ", output_dir)
       os.makedirs(output_dir)

   if (use_weight):
       weight_file=input_dir+dataset_name+'_weights.txt'
   else:
       weight_file = input_dir + dataset_name + '_feature.txt'

   label_file=input_dir+dataset_name+'_labels.txt'

   labels=load_file(label_file).astype(int)
   N=labels[0]
   labels=labels[1:]

   Dimension=-1
   if (use_weight==False):
       M=load_weights(weight_file)
       N,Dimension=M.shape
       print("Examples {0} Dimension: {1}".format(N,Dimension))

   bs = [5, 10, 20, 30, 40, 50]
   iterations = [-1, -1,-1,-1,-1,-1]
   # bs = [5]
   # iterations = [-1]
   # N=20

   arguments=[]
   for b,iteration in zip(bs,iterations):
       degree_file=output_dir+'b_degree_'+str(b)+'.txt'
       output_edges=output_dir+dataset_name+'_edges_'+str(b)+'.txt'
       output_edges_mtx=output_dir+dataset_name+'_edges_'+str(b)+'.mtx'
       output_log=output_dir+dataset_name+'_log_'+str(b)+'.txt'

       create_degree_file(b, N,degree_file)
       print("Using N: ",N)
       Cache=max(10,b) #keep it degree-1 (of course less than N)
       max_iterations = iteration
       verbose = 1
       if(use_weight):
           arguments.insert(len(arguments),(weight_file,degree_file,output_edges,N,Cache,output_log,iteration,verbose))
       else:
           if(Dimension==-1):
               print("Incorrect Dimension")
               sys.exit(0)

           arguments.insert(len(arguments),
                            (weight_file, degree_file, output_edges, N, Cache, Dimension ,output_log, iteration, verbose))

   print(arguments)

   p = Pool(len(bs))

   if(use_weight):
       print(p.map(bmatch_weight_wrapper, arguments))
   else:
       print(p.map(bmatch_feature_wrapper, arguments))

   arguments=[]

   for b,iteration in zip(bs,iterations):
       degree_file = output_dir + 'b_degree_' + str(b) + '.txt'
       output_edges = output_dir + dataset_name + '_edges_' + str(b) + '.txt'
       output_edges_mtx = output_dir + dataset_name + '_edges_' + str(b) + '.mtx'
       arguments.insert(len(arguments), (output_dir, weight_file, output_edges, output_edges_mtx,labels,b,use_weight))

   print(arguments)

   if(use_weight):
       load_weights(weight_file)

   p = Pool(len(bs))
   print(p.map(mtx_wrapper, arguments))

   # for b,iteration in zip(bs,iterations):
   #     degree_file=output_dir+'b_degree_'+str(b)+'.txt'
   #     output_edges=output_dir+dataset_name+'_weights_'+str(b)+'.txt'
   #     output_edges_mtx=output_dir+dataset_name+'_weights_'+str(b)+'.mtx'
   #     output_log=output_dir+dataset_name+'_log_'+str(b)+'.txt'
   #
   #     create_degree_file(b, N,degree_file)
   #
   #     N=20
   #     Cache=10 #keep it degree-1 (of course less than N)
   #
   #     if(use_weight):
   #         bmatch_weight_matrix(weight_file,degree_file,output_edges,N,Cache,output_log,max_iterations=iteration,verbose=1)
   #
   #     text2mtx(output_dir, weight_file, output_edges, output_edges_mtx,labels,b,use_weight)


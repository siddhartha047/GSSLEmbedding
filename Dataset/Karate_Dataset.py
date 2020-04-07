from collections import namedtuple
from networkx import read_edgelist,set_node_attributes
from pandas import read_csv,Series
from numpy import array
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#dataset source
#https://github.com/TobiasSkovgaardJepsen/posts/tree/master/HowToDoDeepLearningOnGraphsWithGraphConvolutionalNetworks/Part2_SemiSupervisedLearningWithSpectralGraphConvolutions

l_markersize = 7

def process_karate_club(filepath):

    Dataset = namedtuple('Dataset', field_names=['x_train', 'y_train', 'x_test', 'y_test', 'Graph', 'Feature','Label'])
    network=read_edgelist(filepath+'karate.edgelist',nodetype=int)

    nx.set_edge_attributes(network,1.0,name='weight')
    #network.edges.data('weight', default=1.0)
    #print(network.edges(data=True))
    #print(network.get_edge_data(0, 5)['weight'])

    attributes=read_csv(filepath+'karate.attributes.csv',index_col=['node'])

    #print(attributes)
    #print(attributes.columns.values)
    #print(Series(attributes['role'],index=attributes.index).to_dict())


    for attribute in attributes.columns.values:
        set_node_attributes(network,values=Series(attributes[attribute],index=attributes.index).to_dict(),name=attribute)

    # print(network.nodes)
    # print(network.edges)
    # print(network.neighbors(0))
    # print(network[0])
    # print(network.nodes.data())
    # print(network.nodes[0])

    train=[(node,int(data['role']=='Administrator')) for node,data in network.nodes(data=True) if data['role'] in {'Instructor','Administrator'}]


    x_train,y_train=map(array,zip(*train))

    test = [(node, int(data['community'] == 'Administrator')) for node, data in network.nodes(data=True) if
             data['role']=='Member']

    #print(test)

    x_test, y_test=map(array,zip(*test))

    # print(x_train,y_train)
    # print(x_test,y_test)
    # print(to_numpy_matrix(network))
    # sp_graph=to_scipy_sparse_matrix(network)
    n=network.number_of_nodes()

    features=np.zeros((n,n))
    i = 0
    for node in range(n):
        features[i,node]=1.0
        i += 1

    # features=nx.convert_matrix.to_numpy_matrix(network)
    # features=features+np.eye(34)

    # print(features)
    # sys.exit(0)

    labels = dict([(node,int(data['community'] == 'Administrator')) for node, data in network.nodes(data=True)])

    list_label=np.zeros((n),dtype='int')
    i = 0
    for node in range(n):
        list_label[node]=labels[node]
        i += 1

    # #customize:
    #
    # x_train = [p for p in range(0, 34)]
    # np.random.seed(1)
    # x_test=np.random.choice(34, 17,replace=False)
    #
    # for x in x_test:
    #     x_train.remove(x)
    #
    # y_train=list_label[x_train]
    # y_test=list_label[x_test]

    # print(x_test)
    # print(x_train)
    # print(x_train.shape)
    # print(x_test.shape)

    return Dataset(x_train, y_train, x_test, y_test, network,features,list_label)

def plot_karate_club(data):
    # https://networkx.github.io/documentation/networkx-1.9/examples/drawing/labels_and_colors.html
    G = data.Graph

    # G = nx.Graph()
    # for edge, weight in data.edges_weights.items():
    #     (u,v)=edge
    #     G.add_edge(u,v,weight=weight)

    pos = nx.spring_layout(G)

    #nodes=list(data.GraphAdj.keys())
    #print(nodes)

    #x_train_nodes = [item for sublist in data.x_train for item in sublist]
    x_train_nodes=data.x_train
    y_train_nodes=list(data.y_train)

    x_train_ad=[ x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i]==1] #admin
    x_train_ins = [x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i] == 0]  # instructor

    #x_test_nodes =  [item for sublist in data.x_test for item in sublist]
    x_test_nodes=data.x_test
    y_test_nodes = list(data.y_test)

    x_test_ad=[ x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i]==1] #admin
    x_test_ins = [x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i] == 0]  # instructor



    nx.draw_networkx_nodes(G,pos,nodelist=x_train_ad,node_color='g')
    nx.draw_networkx_nodes(G, pos, nodelist=x_train_ins, node_color='orange')

    #nx.draw_networkx_nodes(G, pos, nodelist=x_test_nodes, node_color='lightgrey')
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ad, node_color='lightgreen')
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ins, node_color='navajowhite')

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    #nx.draw_networkx(data.Graph, with_labels=True)
    #plt.show(block=False)

    Administrator = mlines.Line2D([], [], color='g', marker='o', linestyle='None',
                              markersize=l_markersize, label='Administrator')
    Instructor = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                                  markersize=l_markersize, label='Instructor')

    mem_admin = mlines.Line2D([], [], color='lightgreen', marker='o', linestyle='None',
                              markersize=l_markersize, label='Member-Administrator')


    mem_instructor = mlines.Line2D([], [], color='navajowhite', marker='o', linestyle='None',
                              markersize=l_markersize, label='Member-Instructor')

    plt.legend(handles=[Administrator, Instructor, mem_admin, mem_instructor])

    title = 'Karate Club Dataset'

    plt.title(title)

    plt.show()

    return

def data_G_karate(filepath):
    data=process_karate_club(filepath)

    n=len(data.Label)
    one_hot_label=np.zeros((n,2),dtype=float)
    for i in range(n):
        one_hot_label[i][data.Label[i]]=1.0

    label_mask=np.zeros(n,dtype=float)
    for i in data.x_train:
        label_mask[i]=1

    Dataset = namedtuple('Dataset', field_names=['train_index', 'test_index', 'val_index','Graph', 'Feature', 'Label','Label_mask','classname'])

    classname=['Administrator','Instructor']

    data=Dataset(train_index=data.x_train,test_index=data.x_test, val_index=data.x_test, Graph=data.Graph,Feature=data.Feature,Label=one_hot_label,Label_mask=label_mask,classname=classname)

    return data

if __name__ == '__main__':
    path='/Users/siddharthashankardas/Purdue/Dataset/Karate/'

    zkc=process_karate_club(filepath=path)
    print(zkc)
    plot_karate_club(zkc)


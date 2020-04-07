from collections import namedtuple
from networkx import read_edgelist,set_node_attributes, to_numpy_matrix, to_scipy_sparse_matrix
from pandas import read_csv,Series
from numpy import array
import numpy as np
import scipy.sparse
import scipy as sp
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
import matplotlib.image as mpimg
import imageio
import torch
import platform
import os

#dataset source
#https://github.com/TobiasSkovgaardJepsen/posts/tree/master/HowToDoDeepLearningOnGraphsWithGraphConvolutionalNetworks/Part2_SemiSupervisedLearningWithSpectralGraphConvolutions


path='/Users/siddharthashankardas/Purdue/Dataset/Karate/'
dir = '/Users/siddharthashankardas/Purdue/Dataset/Karate/experiments/'

l_markersize = 7

def process_karate_club(filepath=path):

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

    # print(result)
    # print(*result)
    # print(zip(*result))

    #print(attributes)

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

def process_karate_club_gcn(filepath=path):

    data=process_karate_club(filepath)

    adj=nx.adjacency_matrix(data.Graph)

    features=sp.sparse.csr_matrix(data.Feature)


    n=len(data.Label)

    train_mask=np.array([False for i in range(n)])
    test_mask=np.array([False for i in range(n)])
    val_mask=np.array([False for i in range(n)])

    val_index=data.x_test

    for i in data.x_train: train_mask[i]=True
    for i in data.x_test: test_mask[i]=True
    for i in val_index: val_mask[i]=True

    y_train=np.zeros((n, 2), dtype='int')
    y_test=np.zeros((n, 2), dtype='int')
    y_val=np.zeros((n, 2), dtype='int')

    one_hot = np.zeros((n, 2), dtype='int')

    i = 0
    for node in range(n):
        one_hot[node, data.Label[node]] = 1
        i += 1


    for i in data.x_train: y_train[i]=one_hot[i]
    for i in data.x_test: y_test[i]=one_hot[i]
    for i in val_index: y_val[i]=one_hot[i]

    all_y=one_hot
    all_mask=np.array([True for i in range(n)])



    Dataset = namedtuple('Dataset', field_names=[
        'adj', 'features', 'y_train', 'y_val', 'y_test', 'train_mask', 'val_mask', 'test_mask','all_y','all_mask'
    ])

    dataset=Dataset(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,all_y,all_mask)

    #print(one_hot)
    # print(adj, adj.shape)
    # print(features,features.shape)
    # print(y_train, y_train.shape)
    # print(y_val)
    # print(y_test)
    # print(train_mask,train_mask.shape)
    # print(val_mask)
    # print(test_mask)

    #print(adj.shape, features.shape, y_train.shape, y_val.shape, y_test.shape, train_mask.shape, val_mask.shape,test_mask.shape)

    return dataset

def plot_karate_continuous(dataset,info):


    data=process_karate_club()
    G = data.Graph

    plt.clf()
    #pos = nx.spring_layout(G)
    pos=dict()
    outputs=info['layer']

    #outputs = (outputs - outputs.min(0)) / outputs.ptp(0)

    for i in range(len(data.Label)):
        pos[i]=tuple(outputs[i])

    # print(pos)
    # print(outputs)
    # sys.exit(0)

    #pos = {0: (40, 20), 1: (20, 30), 2: (40, 30), 3: (30, 10)}

    x_train_nodes = data.x_train
    y_train_nodes = list(data.y_train)

    x_train_ad = [x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i] == 1]  # admin
    x_train_ins = [x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i] == 0]  # instructor

    # x_test_nodes =  [item for sublist in data.x_test for item in sublist]
    x_test_nodes = data.x_test
    y_test_nodes = list(data.y_test)

    x_test_ad = [x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i] == 1]  # admin
    x_test_ins = [x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i] == 0]  # instructor


    node_size=50

    nx.draw_networkx_nodes(G, pos, nodelist=x_train_ad, node_color='g', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_train_ins, node_color='orange', node_size=node_size)

    # nx.draw_networkx_nodes(G, pos, nodelist=x_test_nodes, node_color='lightgrey', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ad, node_color='lightgreen', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ins, node_color='navajowhite', node_size=node_size)

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    #nx.draw_networkx_labels(G, pos)

    pred=info['pred']
    i_pred=list()
    for i in range(len(pred)):
        if pred[i]==0:
            i_pred.append(i)

    nx.draw_networkx_nodes(G,pos,
                       nodelist=i_pred,
                       node_color='k',
                       linewidths=2,
                       node_size=node_size,
                           node_shape='.',
                   alpha=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    ax = plt.gca()

    ax.xaxis.set_tick_params(which='both',labelbottom=True)
    ax.yaxis.set_tick_params(which='both', labelleft=True)

    Administrator = mlines.Line2D([], [], color='g', marker='o', linestyle='None',
                                  markersize=l_markersize, label='Administrator')
    Instructor = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                               markersize=l_markersize, label='Instructor')

    mem_admin = mlines.Line2D([], [], color='lightgreen', marker='o', linestyle='None',
                              markersize=l_markersize, label='Member-Administrator')

    mem_instructor = mlines.Line2D([], [], color='navajowhite', marker='o', linestyle='None',
                                   markersize=l_markersize, label='Member-Instructor')

    mistake = mlines.Line2D([], [], color='k', marker='.', linestyle='None',
                                   markersize=l_markersize, label='Misclassified')

    plt.legend(handles=[Administrator, Instructor, mem_admin, mem_instructor, mistake])

    title=info['name']+": epoch - "+str(info['epoch']+1)

    plt.title(title)

    plt.pause(0.001)

    return 0

def process_karate_club_gsage(filepath=path):

    data = process_karate_club(filepath)

    G=data.Graph

    n = len(data.Label)

    for i in range(n):
        if(i in data.x_train):
            G.node[i]['test']=False
            G.node[i]['val']=False
        else:
            G.node[i]['test'] = True
            G.node[i]['val'] = True

    for edge in G.edges():
        if edge[0] in data.x_train or edge[1] in data.x_train:
            G[edge[0]][edge[1]]['train_removed'] = False
            G[edge[0]][edge[1]]['test_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
            G[edge[0]][edge[1]]['test_removed'] = False

    feats = data.Feature


    id_map={i:i for i in range(n)}

    one_hot = np.zeros((n, 2), dtype='int')

    i = 0
    for node in range(n):
        one_hot[node, data.Label[node]] = 1
        i += 1

    class_map={i:list(one_hot[i]) for i in range(n)}

    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # print(degree_sequence)
    # dmax = max(degree_sequence)
    # print(dmax)

    walks = []

    print(len(G.nodes()))
    print(feats.shape)
    print(len(id_map))
    print(len(walks))
    print(len(class_map))
    print(G[0])
    #print(G[0][1])


    return G, feats, id_map, walks, class_map, [0,1]

images=[]

def plot_karate_animation(dataset,info):

#    fig=plt.figure()

    data=process_karate_club()
    G = data.Graph

    plt.clf()
    #pos = nx.spring_layout(G)
    pos=dict()
    outputs=info['layer']

    #outputs = (outputs - outputs.min(0)) / outputs.ptp(0)

    for i in range(len(data.Label)):
        pos[i]=tuple(outputs[i])

    # print(pos)
    # print(outputs)
    # sys.exit(0)

    #pos = {0: (40, 20), 1: (20, 30), 2: (40, 30), 3: (30, 10)}

    x_train_nodes = data.x_train
    y_train_nodes = list(data.y_train)

    x_train_ad = [x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i] == 1]  # admin
    x_train_ins = [x_train_nodes[i] for i in range(len(y_train_nodes)) if y_train_nodes[i] == 0]  # instructor

    # x_test_nodes =  [item for sublist in data.x_test for item in sublist]
    x_test_nodes = data.x_test
    y_test_nodes = list(data.y_test)

    x_test_ad = [x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i] == 1]  # admin
    x_test_ins = [x_test_nodes[i] for i in range(len(y_test_nodes)) if y_test_nodes[i] == 0]  # instructor


    node_size=50

    nx.draw_networkx_nodes(G, pos, nodelist=x_train_ad, node_color='g', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_train_ins, node_color='orange', node_size=node_size)

    # nx.draw_networkx_nodes(G, pos, nodelist=x_test_nodes, node_color='lightgrey', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ad, node_color='lightgreen', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=x_test_ins, node_color='navajowhite', node_size=node_size)

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    #nx.draw_networkx_labels(G, pos)

    pred=info['pred']
    i_pred=list()
    for i in range(len(pred)):
        if pred[i]==0:
            i_pred.append(i)

    nx.draw_networkx_nodes(G,pos,
                       nodelist=i_pred,
                       node_color='k',
                       linewidths=2,
                       node_size=node_size,
                           node_shape='.',
                   alpha=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    ax = plt.gca()

    ax.xaxis.set_tick_params(which='both',labelbottom=True)
    ax.yaxis.set_tick_params(which='both', labelleft=True)

    Administrator = mlines.Line2D([], [], color='g', marker='o', linestyle='None',
                                  markersize=l_markersize, label='Administrator')
    Instructor = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                               markersize=l_markersize, label='Instructor')

    mem_admin = mlines.Line2D([], [], color='lightgreen', marker='o', linestyle='None',
                              markersize=l_markersize, label='Member-Administrator')

    mem_instructor = mlines.Line2D([], [], color='navajowhite', marker='o', linestyle='None',
                                   markersize=l_markersize, label='Member-Instructor')

    mistake = mlines.Line2D([], [], color='k', marker='.', linestyle='None',
                                   markersize=l_markersize, label='Misclassified')

    plt.legend(handles=[Administrator, Instructor, mem_admin, mem_instructor, mistake])

    title=info['name']+": epoch - "+str(info['epoch']+1)

    plt.title(title)

    plt.pause(0.001)
    #plt.show(block=False)

    plt.savefig(dir+'temp.png')
    img = mpimg.imread(dir+'temp.png')
    images.append(img)


    if((info['epoch']+1)==info['maxepoch']):
        #imageio.mimsave(dir+info['name']+'.gif', images,loop=1)
        imageio.mimsave(dir + info['name'] + '.gif', images)
        writer = imageio.get_writer(dir+info['name']+'.mp4', fps=10)

        for im in images:
            writer.append_data(im)
        writer.close()

    return 0

def plot_performance(info):
    plt.figure()
    data=info['accuracy']

    epochs=[i for i in range(len(data))]

    plt.plot(epochs,data)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    title = info['name'] + ": Epoch vs Accuracy"

    plt.title(title)
    plt.savefig(dir+info['name']+'_accuracy.png')
    #plt.show()

def draw_DGL(i,all_logits,nx_G):
    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()

    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

    plt.show()
    # plt.close()

def draw_anime(all_logits,nx_G):
    fig = plt.figure(dpi=150)
    ani = animation.FuncAnimation(fig, draw_DGL, frames=len(all_logits), interval=200)

def load_karate_data():
    data=process_karate_club()


    features = torch.FloatTensor(data.Feature)
    labels = torch.LongTensor(data.Label)

    n = len(data.Label)

    train_mask=np.zeros(n)
    test_mask=np.zeros(n)
    val_mask=np.zeros(n)

    train_mask[data.x_train]=1.0
    test_mask[data.x_test]=1.0
    val_mask[data.x_test]=1.0

    Dataset=namedtuple('Dataset', field_names=['graph','features','labels','train_mask','val_mask','test_mask','num_labels'])
    ndata=Dataset(graph=data.Graph,features=features,labels=labels,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,num_labels=2)

    return ndata

def data_G_karate():
    data=process_karate_club()

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

    zkc=process_karate_club()
    print(zkc)
    plot_karate_club(zkc)

    # zkc_gcn=process_karate_club_gcn()
    # print(zkc_gcn)

    #plt.show()


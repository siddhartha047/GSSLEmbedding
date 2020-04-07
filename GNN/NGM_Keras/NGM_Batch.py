import numpy as np
#from GNN.NGM_Keras.NGM_Graph import EmbeddingsGraph
# from Dataset.CVE_Dataset import data_G
# from Dataset.Karate_Dataset import data_G_karate
import sys


np.random.seed(1)

def batch_iter_test(data,batch_size=5,labeled_only=True):
    test_samples=data.Feature[data.test_index]
    test_labels=data.Label[data.test_index]

    len_data=len(test_labels)

    np.random.seed(1)
    shuffle_indices=np.random.permutation(range(len_data))

    num_batches=int(len_data/batch_size)

    if(len_data%batch_size>0):
        num_batches+=1

    #print(num_batches)

    for batch_num in range(num_batches):
        start_index=batch_num*batch_size
        end_index=min(len_data,(batch_num+1)*batch_size)

        #print(start_index,end_index)

        batch_indices= shuffle_indices[start_index:end_index]

        #print(batch_indices)

        input_x=test_samples[batch_indices]
        labels=test_labels[batch_indices]

        yield  (input_x,labels)

def batch_iter_val(data,batch_size=5,labeled_only=True):
    test_samples=data.Feature[data.val_index]
    test_labels=data.Label[data.val_index]

    len_data=len(test_labels)

    np.random.seed(1)
    shuffle_indices=np.random.permutation(range(len_data))

    num_batches=int(len_data/batch_size)

    if(len_data%batch_size>0):
        num_batches+=1

    #print(num_batches)

    for batch_num in range(num_batches):
        start_index=batch_num*batch_size
        end_index=min(len_data,(batch_num+1)*batch_size)

        #print(start_index,end_index)

        batch_indices= shuffle_indices[start_index:end_index]

        #print(batch_indices)

        input_x=test_samples[batch_indices]
        labels=test_labels[batch_indices]

        yield  (input_x,labels)


def batch_iter_train(data,batch_size=5,labeled_only=True):

    graph=data.Graph

    data_size=len(graph.edges)

    edges=np.random.permutation(graph.edges())

    #print(edges)

    num_batches = int(data_size / batch_size)

    if (data_size % batch_size > 0):
        num_batches += 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(data_size, (batch_num + 1) * batch_size)

        #print(start_index,end_index)

        yield next_batch(data,edges,start_index,end_index)

def next_batch(data,edges,start,end):
    # start=10
    # end=15
    # print(start,end)
    graph=data.Graph


    edges_ll=list()
    edges_lu=list()
    edges_uu=list()
    weights_ll=list()
    weights_lu=list()
    weights_uu=list()

    batch_edges=edges[start:end]
    batch_edges=np.asarray(batch_edges)

    x_train = data.train_index
    x_val = data.val_index

    Features = data.Feature
    labels = data.Label


    for i,j in batch_edges:

        if(i in x_train and j in x_train):
            edges_ll.append((i,j))
            weights_ll.append(graph.get_edge_data(i,j)['weight'])

        elif(i in x_train and j in x_val):
            edges_lu.append((i, j))
            weights_lu.append(graph.get_edge_data(i, j)['weight'])

        elif(j in x_train and i in x_val):
            edges_lu.append((j, i))
            weights_lu.append(graph.get_edge_data(j, i)['weight']) # i, j and j, i can be omitted later on

        else:
            edges_uu.append((i, j))
            weights_uu.append(graph.get_edge_data(i, j)['weight'])


    # print(edges_ll)
    # print(weights_ll)
    #
    # print(edges_lu)
    # print(weights_lu)
    #
    # print(edges_uu)
    # print(weights_uu)

    # print(Features.shape)

    u_ll=[e[0] for e in edges_ll]
    c_ull=[1.0/graph.degree(n) for n in u_ll]

    v_ll=[e[1] for e in edges_ll]
    c_vll = [1.0 / graph.degree(n) for n in v_ll]

    nodes_ll_u=Features[u_ll]
    labels_ll_u=labels[u_ll]

    nodes_ll_v=Features[v_ll]
    labels_ll_v=labels[v_ll]

    u_lu = [e[0] for e in edges_lu]
    c_ulu = [1.0 / graph.degree(n) for n in u_lu]

    nodes_lu_u = Features[u_lu]
    labels_lu_u = labels[u_lu]

    nodes_lu_v=Features[[e[1] for e in edges_lu]]

    nodes_uu_u=Features[[e[0] for e in edges_uu]]
    nodes_uu_v = Features[[e[1] for e in edges_uu]]

    # print(nodes_ll_u)
    # print(nodes_ll_v)
    # print(nodes_lu_u)
    # print(nodes_lu_v)
    # print(nodes_uu_u)
    # print(nodes_uu_v)
    # print(labels_ll_u)
    # print(labels_ll_v)
    # print(labels_lu_u)
    # print(weights_ll)
    # print(weights_lu)
    # print(weights_uu)
    # print(c_ull,c_vll,c_ulu)

    return (nodes_ll_u,nodes_ll_v, nodes_lu_u,nodes_lu_v,nodes_uu_u,nodes_uu_v, labels_ll_u,labels_ll_v,labels_lu_u,weights_ll,weights_lu,weights_uu, c_ull,c_vll,c_ulu)


def batch_incremental(data, batch_size=5,labeled_only=True):

    indexs=data.train_index
    print(indexs)
    graph=data.Graph

    original_edges=graph.edges(indexs)

    data_size=len(original_edges)

    # print(graph.edges())
    # print(original_edges)

    edges = np.random.permutation(list(original_edges))
    #print(edges)

    # print(edges)

    num_batches = int(data_size / batch_size)

    if (data_size % batch_size > 0):
        num_batches += 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(data_size, (batch_num + 1) * batch_size)

        # print(start_index,end_index)

        yield next_batch(data, edges, start_index, end_index)


# if __name__ == '__main__':
#     dataset_name='CVE'
#     if (dataset_name == 'CVE'):
#         data = data_G(True)
#     elif (dataset_name == 'karate'):
#         data = data_G_karate()
#
#     print(batch_iter_train(data,5).__next__())
#     print(batch_iter_test(data,5).__next__())
#     print(batch_iter_val(data,5).__next__())
#     print(batch_incremental(data,5).__next__())

    # for _ in batch_iter(5):
    #      pass

    #batch_incremental(5).__next__()








# if (acc != acc):
#     print("u1->"), print(u1)
#     print("v1->"), print(v1)
#     print("u2->"), print(u2)
#     print("v2->"), print(v2)
#     print("u3->"), print(u3)
#     print("v3->"), print(v3)
#     print("lu1->"), print(lu1)
#     print("lv1->"), print(lv1)
#     print("lu2->"), print(lu2)
#     print("w_ll->"), print(w_ll)
#     print("w_lu->"), print(w_lu)
#     print("w_uu->"), print(w_uu)
#     print("c_ull->"), print(c_ull)
#     print("c_vll->"), print(c_vll)
#     print("c_ulu->"), print(c_ulu)

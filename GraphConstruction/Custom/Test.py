from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt

def test1():
    d = defaultdict(dict)


    d[0,1]=1
    d[1,10]=10

    print(d)


    print(d[0,1])

    if( (0,2) in d.keys()):
        print("exists")
    else:
        print("not here")


    def l_regularizer(outputs, l_index):
        return

def test2():
    cos=torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    x1=torch.tensor([1.0,2.0])
    x2=torch.tensor([2.0,3.0])

    print(cos(x1,x2))

    d=torch.pow(x1-x2,2).sum().sqrt()

    print(d)

    # M=torch.zeros(3,3)
    #
    # print(M)

#takes in two vector
def in_feature_distance(x1,x2):
    d=torch.pow(x1 - x2, 2).sum().sqrt()
    return d

def in_reweight(d,sigma):
    w = torch.exp(-d/(2.0*sigma*sigma))
    return w

def test3():
    from GraphConstruction.Custom.Synthetic import load_synthetic
    import GraphConstruction.Custom.utils as utils

    X_train, y_train, X_test, y_test = load_synthetic(labeled=3)
    n_training_examples = y_train.shape[0]
    n_class = 3
    X_train_all = np.concatenate((X_train, X_test))
    y_train_all = np.append(y_train, np.ones(y_test.shape[0]) * (-1)).astype(int)
    y_train_confidence = np.append(np.ones(y_train.shape[0]), np.zeros(y_test.shape[0]))

    y_train_1hot = utils.one_hot(y_train, n_class)
    y_test_1hot = utils.one_hot(y_test, n_class)
    y_train_all_1hot = np.concatenate((y_train_1hot, np.zeros((y_test.shape[0], n_class), dtype=int)))

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(
        y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)
    X_train_all, y_train_all, y_train_all_1hot = torch.from_numpy(X_train_all), torch.from_numpy(
        y_train_all), torch.from_numpy(y_train_all_1hot)
    X_train_all = X_train_all.type(torch.FloatTensor)

    n=len(y_train_all)
    sigma=0.5
    th=0.6
    M=torch.zeros(n,n)
    count=0

    for i in range(n):
        for j in range(i+1,n):
            if(i==j):continue
            x1=X_train_all[i, :]
            x2=X_train_all[j, :]
            d=in_feature_distance(x1,x2)
            #d=in_reweight(d,sigma)
            if(d<th):
                count+=1
                M[i,j]=d

    print("Edges: ",count)

    plt.clf()
    scatter=plt.scatter(X_train_all[:,0],X_train_all[:,1],c=np.append(y_train, y_test))

    for i in range(n):
        for j in range(i+1,n):
            if M[i,j]>1e-4:
                x1=X_train_all[i,:]
                x2=X_train_all[j,:]
                plt.plot([x1[0],x2[0]],[x1[1],x2[1]])


    plt.legend(*scatter.legend_elements())
    plt.show()



if __name__ == '__main__':
    test3()
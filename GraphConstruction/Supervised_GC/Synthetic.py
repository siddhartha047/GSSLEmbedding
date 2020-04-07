import os
import torch
from   torchvision import datasets, transforms
import numpy as np
import logging
import matplotlib.pyplot as plt

np.random.seed(123)


def load_data(data_file,weight_file,label1_file,label2_file,num_of_training=3):

    data=np.loadtxt(data_file,delimiter=',', usecols=(0, 1)).astype(float)
    label1=np.loadtxt(label1_file).astype(int)
    label2 = np.loadtxt(label2_file).astype(int)-1
    n=data.shape[0]
    per_class=int(num_of_training/3)
    train_index=np.random.randint(0,int(n/3),per_class)
    train_index=np.append(train_index,np.random.randint(int(n / 3), int(2*n / 3), per_class))
    train_index=np.append(train_index,np.random.randint(int(2*n / 3), int(n), per_class))
    train_index=list(train_index)
    test_index=[i for i in range(0,n) if i not in train_index]

    print(train_index)
    # print(test_index)
    # print(label2)

    plt.clf()
    plt.scatter(data[:,0],data[:,1],c=label2)
    plt.savefig("images/original_data.png")

    return (data, label2, train_index,test_index)

def load_synthetic(labeled=3):
    data_file = '/Users/siddharthashankardas/Purdue/Dataset/Synthetic/synthetic_data300_3.txt'
    weight_file = '/Users/siddharthashankardas/Purdue/Dataset/Synthetic/synthetic_weights300_3.txt'
    label1_file = '/Users/siddharthashankardas/Purdue/Dataset/Synthetic/synthetic_label300_3.txt'
    label2_file = '/Users/siddharthashankardas/Purdue/Dataset/Synthetic/synthetic_300_3.label'

    (data, label2, train_index, test_index) = load_data(data_file, weight_file, label1_file, label2_file,num_of_training=labeled)

    return data[train_index], label2[train_index], data[test_index], label2[test_index]


if __name__ == '__main__':
    x_train,y_train, x_test, y_test = load_synthetic(labeled=3)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train)

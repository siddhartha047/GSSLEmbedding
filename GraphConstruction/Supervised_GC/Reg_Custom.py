import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import logging
import time

from GraphConstruction.Custom.Synthetic import load_synthetic
import GraphConstruction.Custom.utils as utils
from GraphConstruction.Custom.FC_minibatcher import MiniBatcher
from collections import defaultdict


confidence=0.8

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        # 2D convolutional layer with max pooling layer and reLU activations
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        fc2_r = F.relu(x)
        fc3 = self.fc3(fc2_r)
        y = F.log_softmax(fc3, dim=1)

        return fc2_r,fc3,y

class MyCNNnetwork():
    def __init__(self,learning_rate=1e-4, gpu_id=-1):
        self.gpu_id=gpu_id
        self.learning_rate=learning_rate
        self.net=Net()
        if(gpu_id!=-1):
            print("using gpu")
            self.net=self.net.cuda(gpu_id)

    def extract_labeled(self,y_train_confidence):
        labeled = np.argwhere(y_train_confidence > confidence)
        return np.squeeze(labeled)




    def l_regularizer(self,outputs,l_index, L, D):
        Lloss=torch.zeros(1)
        Dloss=torch.zeros(1)
        cos=torch.nn.CosineSimilarity(dim=0, eps=1e-08)

        for keys in L:
            d=(1 - cos(outputs[keys[0]], outputs[keys[1]]))
            Lloss+=d/2

        for keys in D:
            d = (1 - cos(outputs[keys[0]], outputs[keys[1]]))
            d = 1-d / 2
            Dloss += d
        n=(len(l_index)*len(l_index)-len(l_index))
        loss=(Lloss + Dloss) / (2.0*n)

        return loss


    def getLDset(self,y_train,l_index):
        L = defaultdict(dict)
        D = defaultdict(dict)

        for i in l_index:
            for j in l_index:
                if(i==j):
                    continue
                if(y_train[i]==y_train[j]):
                    L[i,j]=1
                    L[j,i]=1
                else:
                    D[i,j]=1
                    D[j,i]=1

        return L,D

    #takes in two vector
    def in_feature_distance(self,x1,x2):
        d=torch.pow(x1 - x2, 2).sum().sqrt()

        return d

    def in_reweight(self,d,sigma):
        w = torch.exp(-d/(2.0*sigma*sigma))
        return w

    def out_feature_distance(self,x1,x2):

        return

    def getGraphEdges(self,X, fc3,y_train,y_train_confidence, L, D):

        LL = defaultdict(dict)
        LU = defaultdict(dict)
        UU = defaultdict(dict)

        n=len(y_train)
        sigma = 0.5
        th = 0.6

        count = 0
        l_index = self.extract_labeled(y_train_confidence)

        for i in range(n):
            for j in range(i + 1, n):
                if (i == j): continue
                x1 = X[i, :]
                x2 = X[j, :]
                d = self.in_feature_distance(x1, x2)

                if ((i, j) in L.keys()):
                    count += 1
                    LL[i,j]=d
                    LL[j,i]=d

                elif(i in l_index):
                    count += 1
                    LU[i,j]=d
                    LU[j,i] = d
                elif(i, j) not in D.keys() and (d < th ):
                    count += 1
                    UU[i, j] = d
                    UU[j, i] = d

        print("Edges: ", count)

        return LL, LU, UU

    def train_one_batch(self, X, y, y_1hot, y_train_confidence, printout=False):
        """Train for one batch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        self.net.train()
        fc2_r,fc3, output = self.net(X)

        l_index=self.extract_labeled(y_train_confidence)
        L,D = self.getLDset(y,l_index)
        LL,LU,UU = self.getGraphEdges(X,fc3,y,y_train_confidence,L,D)
        print(L)
        print(D)
        print(l_index)
        print(LL)
        print(LU)
        print(UU)

        # Computes the negative log-likelihood over the training data target.
        #     log(softmax(.)) avoids computing the normalization factor.
        #     Note that target does not need to be 1-hot encoded (pytorch will 1-hot encode for you)
        r_loss = self.l_regularizer(fc3, l_index, L, D)
        #e_loss = self.e_regulazier(fc3,l_index,LL,LU,UU)

        l_loss=F.nll_loss(output[l_index], y[l_index])

        loss = l_loss+r_loss



        # **Very important,** need to zero the gradient buffer before we can use the computed gradients
        self.net.zero_grad()  # zero the gradient buffers of all parameters

        # Backpropagate loss for all paramters over all examples
        loss.backward()

        # iterate over all model parameters
        for f in self.net.parameters():
            # why subtract? Are we minimizing or maximizing?
            f.data.sub_(f.grad.data * self.learning_rate)

        return loss

    def regulazer_loss(self, X, y, y_1hot, y_train_confidence):
            self.net.eval()
            fc2_r, fc3, output = self.net(X)

            l_index = self.extract_labeled(y_train_confidence)
            L, D = self.getLDset(y, l_index)

            # print(L)
            # print(D)

            r_loss = self.l_regularizer(fc3, l_index, L, D)
            loss = F.nll_loss(output[l_index], y[l_index]) + r_loss

            print(F.nll_loss(output[l_index], y[l_index]))
            print(r_loss)

            return loss


    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        self.net.eval()
        with torch.no_grad():
            fc2_r,fc3, output = self.net(X)

            # Computes the negative log-likelihood over the training data target.
            #     log(softmax(.)) avoids computing the normalization factor.
            #     Note that target does not need to be 1-hot encoded (pytorch will 1-hot encode for you)
            loss = F.nll_loss(output, y)

            return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        self.net.eval()
        with torch.no_grad():
            _,_,output = self.net(X)
            return torch.max(output, 1)[1]

    def intermediate(self,X_train,y_train,X_test,y_test,epoch=-1,directory=''):
        self.net.eval()
        with torch.no_grad():
            fc2_r,fc3, output = self.net(X_test)
            outputs=output.numpy()
            plt.clf()
            plt.scatter(outputs[:,0],outputs[:,1],c=y_test)
            plt.savefig(directory+"test_transform"+str(epoch)+".png")



    def update_confidence(self,X_train_all,y_train_all,y_train_all_1hot,y_train_confidence,epoch):
        self.net.eval()
        with torch.no_grad():
            fc2_r, fc3, _ = self.net(X_train_all)
            output=F.softmax(fc3,dim=1)
            maxs=output.max(dim=1)[0]

            for i in range(0,output.shape[0]):
                #print(maxs[i])
                if(maxs[i]>confidence and y_train_confidence[i]<0.99):
                    y_train_confidence[i]=maxs[i]
                    y_train_all[i]=output[i,].argmax()
                    y_train_all_1hot[i,y_train_all[i]]=1

            outputs = output.numpy()
            col=[y_train_all[i] if y_train_confidence[i]>0 else 3 for i in range(len(y_train_confidence))]
            plt.clf()
            scatter=plt.scatter(X_train_all[:, 0], X_train_all[:, 1], c=col)
            plt.legend(*scatter.legend_elements())

            plt.savefig("images/temp/confidence"+str(epoch)+".png")




if __name__ == '__main__':
    print("here")
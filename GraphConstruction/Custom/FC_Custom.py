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
        h2 = F.relu(x)
        x = self.fc3(h2)
        y = F.log_softmax(x, dim=1)

        return h2,y

class MyCNNnetwork():
    def __init__(self,learning_rate=1e-4, gpu_id=-1):
        self.gpu_id=gpu_id
        self.learning_rate=learning_rate
        self.net=Net()
        if(gpu_id!=-1):
            print("using gpu")
            self.net=self.net.cuda(gpu_id)

    def train_one_batch(self, X, y, y_1hot, printout=False):
        """Train for one batch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        self.net.train()
        M, output = self.net(X)

        # Computes the negative log-likelihood over the training data target.
        #     log(softmax(.)) avoids computing the normalization factor.
        #     Note that target does not need to be 1-hot encoded (pytorch will 1-hot encode for you)
        loss = F.nll_loss(output, y)

        # **Very important,** need to zero the gradient buffer before we can use the computed gradients
        self.net.zero_grad()  # zero the gradient buffers of all parameters

        # Backpropagate loss for all paramters over all examples
        loss.backward()

        # iterate over all model parameters
        for f in self.net.parameters():
            # why subtract? Are we minimizing or maximizing?
            f.data.sub_(f.grad.data * self.learning_rate)

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
            M, output = self.net(X)

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
            _,output = self.net(X)
            return torch.max(output, 1)[1]

    def intermediate(self,X_train,y_train,X_test,y_test):
        self.net.eval()
        with torch.no_grad():
            h, output = self.net(X_test)
            outputs=output.numpy()
            plt.clf()
            plt.scatter(outputs[:,0],outputs[:,1],c=y_test)
            plt.savefig("images/test_transform.png")


if __name__ == '__main__':
    print("here")
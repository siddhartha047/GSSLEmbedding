import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import dgl
import timeit
import sys
from dgl.nn.pytorch import GraphConv

####settings for reproducibility
seed=1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class GCN(nn.Module):
    def __init__(self,g,config):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(config['input_features'],config['hidden_neurons'][0], activation=config['activation']))

        # hidden layers
        for i in range(len(config['hidden_neurons']) - 1):
            self.layers.append(GraphConv(config['hidden_neurons'][i],config['hidden_neurons'][i+1], activation=config['activation']))

        # output layer
        #self.layers.append(GraphConv(config['hidden_neurons'][-1],config['out_features']))
        self.layers.append(nn.Linear(config['hidden_neurons'][-1], config['out_features']))
        self.dropout = nn.Dropout(p=config['dropout'][0])

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            if(layer.__class__.__name__=='Linear'):
                h=layer(h)
            else:
                h = layer(self.g, h)
        return h


#test_layer=GCNLayer
#test_layer=nn.Linear

# class GCN(nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         self.layers = nn.ModuleList()
#
#         # input layer
#         self.layers.append(test_layer(config['input_features'],config['hidden_neurons'][0]))
#         self.layers.append(nn.ReLU())
#
#         if(config['dropout']>0):
#             self.layers.append(nn.Dropout(config['dropout']))
#
#
#         # hidden layers
#         for i in range(len(config['hidden_neurons'])-1):
#             self.layers.append(test_layer(config['hidden_neurons'][i],config['hidden_neurons'][i+1]))
#             self.layers.append(nn.ReLU())
#             if (config['dropout'] > 0):
#                 self.layers.append(nn.Dropout(config['dropout']))
#
#         # output layer
#         self.layers.append(test_layer(config['hidden_neurons'][-1],config['out_features']))
#
#     def forward(self, G,features):
#         x = features
#         for layer in self.layers:
#
#             if(layer.__class__.__name__=='GCNLayer'):
#                 x=layer(G,x)
#             else:
#                 x = layer(x)
#         return x


def accuracy(pred_label, true_label):
    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    prediction = (pred_label==true_label.cpu().detach().numpy() ).astype(int)
    acc = np.mean(prediction)

    return acc


def train(config,data):
    #print(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('Using :  ',device)

    G = data.Graph

    model = GCN(G,config)
    print(model)

    model=model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    _, norm_labels = data.y.max(dim=1)


    features=data.x.to(device)
    norm_labels = norm_labels.to(device)
    train_index=data.train_index.to(device)
    val_index=data.val_index.to(device)
    test_index=data.test_index.to(device)

    train_accs = list()
    val_accs = list()
    test_accs = list()

    epochs = config['epoch']
    model.train()
    for epoch in range(epochs):

        optimizer.zero_grad()
        x = model(features)

        outputs=F.softmax(x,dim=1)
        loss = criterion(outputs[train_index], norm_labels[train_index])

        loss.backward()
        optimizer.step()

        train_acc=accuracy(outputs[train_index],norm_labels[train_index])
        val_acc=accuracy(outputs[val_index],norm_labels[val_index])
        test_acc=accuracy(outputs[test_index],norm_labels[test_index])


        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print("Epoch: {0}, Loss: {1}, Train Acc: {2}, Val Acc: {3}".format(epoch+1,loss.item(),train_acc, val_acc))

    print("Saving model")
    torch.save(model.state_dict(), config['output_path'] + config['dataset_name']+'_GCN_DGL')

    model.eval()
    test_pred=model(features)
    test_accuracy=accuracy(test_pred[test_index],norm_labels[test_index])
    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    from GNN.Utils import plot_train_val_accuracy
    filename = config['dataset_name'] + '_GCN_DGL_'+ 'epoch_' + str(epochs) + '_class_' + str(config['out_features'])
    plot_train_val_accuracy(config['output_path'],{'name': filename, 'train_accs': train_accs, 'val_accs': val_accs, 'test_accs':test_accs})


    ########
    pred_label=test_pred[test_index]
    true_label=norm_labels[test_index]

    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    true_label = true_label.cpu().detach().numpy()

    # print(pred_label)
    # print(true_label)

    from GNN.Utils import draw_confusion_matrix
    filename=config['output_path'] + config['dataset_name']+"_GCN_DGL" + str(epochs) + "_CM.png"
    draw_confusion_matrix(true_label, pred_label, data.classname,filename)

    return

def load_saved(config,data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using :  ', device)

    G = data.Graph

    model = GCN(G,config)
    model_name=config['output_path'] + config['dataset_name']+'_GCN_DGL'
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    model = model.to(device)
    features = data.x.to(device)
    _, norm_labels = data.y.max(dim=1)
    norm_labels = norm_labels.to(device)

    model.eval()
    test_pred = model(features)

    test_index = data.test_index.to(device)

    pred_label = test_pred[test_index]
    true_label = norm_labels[test_index]

    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    true_label = true_label.cpu().detach().numpy()

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + config['dataset_name'] + "_GCN_DGL" + str(config['epoch']) + "_CM.png"
    draw_confusion_matrix(true_label, pred_label, data.classname, filename)

    return


def learn(config,data):
    start = timeit.default_timer()
    train(config,data)
    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))


if __name__ == '__main__':
    start = timeit.default_timer()

    path="/Users/siddharthashankardas/Purdue/Dataset/Karate/"

    from GNN_configuration import getSettings, load_data_DGL
    load_config={
        "input_path":path,
        "labeled_only":True,
        "dataset_name":"karate"
    }
    data = load_data_DGL(load_config)

    print(data)

    gnn_settings = getSettings(load_config['dataset_name'])
    gnn_settings['output_path'] = path

    train(gnn_settings,data)
    #load_saved(gnn_settings,data)

    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))


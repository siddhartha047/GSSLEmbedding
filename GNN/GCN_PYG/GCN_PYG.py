import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import timeit

from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import sys

seed=1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

#settings
#layer=nn.Linear
layer=GCNConv

def setLayer(algorithm):
    global layer
    if(algorithm=='GCN'):
        layer=GCNConv
    elif(algorithm=='GraphSAGE'):
        layer=SAGEConv
    elif(algorithm=='GAT'):
        layer=GATConv
    else:
        print('Not defined yet')
        sys.exit(0)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = layer(34,10)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = layer(10, 10)
        #self.drop2 = nn.Dropout(0.5)
        self.conv3 = layer(10,2)

    def forward(self,data):
        x, edge_index=data.x,data.edge_index

        print(self.conv1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.conv2(x,edge_index)
        x = F.relu(x)
        #x = self.drop2(x)

        x = self.conv3(x, edge_index)

        return x

def make_data(dataset):

    x=torch.tensor(dataset.Feature,dtype=torch.float)
    y=torch.tensor(dataset.Label,dtype=torch.long)
    G=dataset.Graph
    edges=G.edges()

    u = [src for src,dst in edges]
    v = [dst for src, dst in edges]

    #make it undirected
    tmp_u=list(u)
    u.extend(v)
    v.extend(tmp_u)

    train_index=torch.tensor(dataset.train_index,dtype=torch.long)
    test_index=torch.tensor(dataset.test_index,dtype=torch.long)
    val_index=torch.tensor(dataset.val_index,dtype=torch.long)

    edge_index=torch.tensor([u,v],dtype=torch.long)
    data=Data(x=x,y=y,edge_index=edge_index,train_index=train_index,test_index=test_index,val_index=val_index)

    print(edge_index)
    sys.exit(0)



    return (data,dataset.classname)

def accuracy(pred_label, true_label):
    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    prediction = (pred_label==true_label.cpu().detach().numpy() ).astype(int)
    acc = np.mean(prediction)

    return acc

def train(config,data):

    (data,classname)=make_data(data)

    print(data)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    _, norm_labels = data.y.max(dim=1)


    train_accs = list()
    val_accs = list()
    test_accs = list()

    epochs = 100
    model.train()
    for epoch in range(epochs):

        optimizer.zero_grad()
        x = model(data)
        outputs=F.softmax(x,dim=1)

        loss = criterion(outputs[data.train_index], norm_labels[data.train_index])

        loss.backward()
        optimizer.step()

        train_acc=accuracy(outputs[data.train_index],norm_labels[data.train_index])
        val_acc=accuracy(outputs[data.val_index],norm_labels[data.val_index])
        test_acc=accuracy(outputs[data.test_index],norm_labels[data.test_index])


        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print("Epoch: {0}, Loss: {1}, Train Acc: {2}, Val Acc: {3}".format(epoch+1,loss.item(),train_acc, val_acc))


    model.eval()
    test_pred=model(data)
    test_accuracy=accuracy(test_pred[data.test_index],norm_labels[data.test_index])
    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    from GNN.Utils import plot_train_val_accuracy
    plot_train_val_accuracy(config['output_path'],{'name': 'GCN_PYG_'+layer.__name__, 'train_accs': train_accs, 'val_accs': val_accs, 'test_accs':test_accs})

    test_pred = model(data)
    pred_label=test_pred[data.test_index]
    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    true_label=norm_labels[data.test_index].cpu().detach().numpy()

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + config['dataset_name'] + "_GCN_PYG" + str(epochs) + "_CM.png"
    draw_confusion_matrix(true_label, pred_label, classname, filename)

    return


def learn(config,data,algorithm):
    start = timeit.default_timer()
    setLayer(algorithm)
    train(config,data)


    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))


if __name__ == '__main__':
    start = timeit.default_timer()

    path="/Users/siddharthashankardas/Purdue/Dataset/Karate/"

    from GNN_configuration import getSettings, get_dataset

    load_config={
        "input_path":path,
        "labeled_only":True,
        "dataset_name":"karate"
    }
    data = get_dataset(load_config)

    #print(data)

    gnn_settings = getSettings(load_config['dataset_name'])
    gnn_settings['output_path'] = path

    setLayer("GCN")
    train(gnn_settings,data)
    #load_saved(gnn_settings,data)

    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))


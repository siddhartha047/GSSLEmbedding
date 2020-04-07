import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timeit

####settings for reproducibility
seed=1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


class Net(nn.Module):
    def __init__(self,config):
        super(Net,self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(nn.Linear(config['input_features'],config['hidden_neurons'][0]))
        self.layers.append(nn.ReLU())

        if(config['dropout'][0]>0):
            self.layers.append(nn.Dropout(config['dropout'][0]))


        # hidden layers
        for i in range(len(config['hidden_neurons'])-1):
            self.layers.append(nn.Linear(config['hidden_neurons'][i],config['hidden_neurons'][i+1]))
            self.layers.append(nn.ReLU())
            if (config['dropout'][i+1] > 0):
                self.layers.append(nn.Dropout(config['dropout'][i+1]))

        # output layer
        self.layers.append(nn.Linear(config['hidden_neurons'][-1],config['out_features']))

    def forward(self, features):
        x = features
        for layer in self.layers:
            x = layer(x)
        return x


def accuracy(pred_label, true_label):
    pred_label = np.argmax(pred_label.detach().numpy(), axis=1)
    prediction = (pred_label==true_label.detach().numpy() ).astype(int)
    acc = np.mean(prediction)

    return acc

def correct(pred_label, true_label):
    pred_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1)
    prediction = (pred_label==true_label.cpu().detach().numpy() ).astype(int)
    return np.sum(prediction), len(prediction)

def batch(features,labels,batch_size=128):
    l=features.size(0)
    n_batch = int(l / batch_size)
    if (l % batch_size > 0):
        n_batch += 1

    for i in range(n_batch):
        s=i*batch_size
        t=min((i+1)*batch_size,l)

        batch_x=features[s:t]
        batch_y=labels[s:t]

        yield batch_x,batch_y


def train(config,data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using : ',device)

    model = Net(config)
    print(model)


    model=model.to(device)
    features=data.x.to(device)

    _, norm_labels = data.y.max(dim=1)
    norm_labels=norm_labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_accs = list()
    val_accs = list()

    epochs = config['epoch']

    x_train=features[data.train_index]
    y_train=norm_labels[data.train_index]

    x_val=features[data.val_index]
    y_val=norm_labels[data.val_index]

    x_test=features[data.test_index]
    y_test=norm_labels[data.test_index]


    for epoch in range(epochs):

        #Training
        total_loss=0
        t_correct=0.0
        t_count=0.0

        model.train()
        for x_train_batch, y_train_batch  in batch(x_train,y_train,config['batch_size']):
            optimizer.zero_grad()

            x = model(x_train_batch)
            x_outputs=F.softmax(x,dim=1)

            loss = criterion(x_outputs,y_train_batch)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            t_cor,t_cou=correct(x_outputs, y_train_batch)
            t_correct+=t_cor
            t_count+=t_cou

        t_acc=t_correct/t_count
        train_accs.append(t_acc)

        #validation
        v_correct=0.0
        v_count=0.0

        #model.eval()
        for x_val_batch, y_val_batch  in batch(x_val,y_val):
            val_pred = model(x_val_batch)
            v_cor,v_cou = correct(val_pred, y_val_batch)
            v_correct+=v_cor
            v_count+=v_cou
        v_acc=v_correct/v_count
        val_accs.append(v_acc)

        print("Epoch: {0}, Loss: {1}, Train Acc: {2}, Val Acc: {3}".format(epoch+1,total_loss,t_acc,v_acc))

    #Testing
    test_correct=0.0
    test_count=0.0

    print("Saving model")
    torch.save(model.state_dict(), config['output_path'] + config['dataset_name'] + '_FC')

    model.eval()
    for x_test_batch, y_test_batch in batch(x_test, y_test):
        test_pred = model(x_test_batch)
        test_cor, test_cou = correct(test_pred, y_test_batch)

        test_correct+=test_cor
        test_count+=test_cou

    test_accuracy=test_correct/test_count
    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    from GNN.Utils import plot_train_val_accuracy
    filename = config['dataset_name'] + '_FC_' + 'epoch_' + str(epochs) + '_class_' + str(config['out_features'])
    plot_train_val_accuracy(config['output_path'],{'name': filename, 'train_accs': train_accs, 'val_accs': val_accs})

    model.eval()
    test_pred=model(x_test)
    pred_label = np.argmax(test_pred.cpu().detach().numpy(), axis=1)
    true_label=y_test.cpu().detach().numpy()

    # print(pred_label)
    # print(true_label)

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + config['dataset_name'] + "_FC" + str(epochs) + "_CM.png"
    draw_confusion_matrix(true_label, pred_label, data.classname, filename)

    return

def load_saved(config,data):
    #from CVE.CVE_FC.CVE_FC import Net

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using : ', device)

    model = Net(config)
    model.load_state_dict(torch.load(config['output_path'] + config['dataset_name'] + '_FC', map_location=device))
    model.eval()

    model = model.to(device)
    features = data.x.to(device)

    _, norm_labels = data.y.max(dim=1)
    norm_labels = norm_labels.to(device)

    x_train = features[data.train_index]
    y_train = norm_labels[data.train_index]

    x_val = features[data.val_index]
    y_val = norm_labels[data.val_index]

    x_test = features[data.test_index]
    y_test = norm_labels[data.test_index]


    test_correct=0.0
    test_count=0.0

    for x_test_batch, y_test_batch in batch(x_test, y_test):
        test_pred = model(x_test_batch)
        test_cor, test_cou = correct(test_pred, y_test_batch)

        test_correct+=test_cor
        test_count+=test_cou

    test_accuracy=test_correct/test_count
    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    test_pred = model(x_test)
    pred_label = np.argmax(test_pred.cpu().detach().numpy(), axis=1)
    true_label = y_test.cpu().detach().numpy()

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + config['dataset_name'] + "_FC" + str(config['epoch']) + "_CM.png"
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
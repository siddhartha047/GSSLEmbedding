import sys
from collections import namedtuple
import dgl
import torch
import torch.nn.functional as F

def getSettings(dataset_name,data=""):

    config=dict()
    config['epoch'] = 100

    if (dataset_name == 'karate'):
        config['dataset_name'] = 'karate'
        config['labeled_only'] = True

        config['input_features'] = 34
        config['dropout'] = [0, 0]
        config['hidden_neurons'] = [10,10]
        config['out_features'] = 2
        config['batch_size'] = 5
        config['activation'] = F.relu
        config['learning_rate'] = 1e-2


    elif (dataset_name == 'CVE'):
        config['dataset_name'] = 'CVE'
        config['labeled_only'] = False
        config['input_features'] = 1000 #12432
        config['dropout'] = [0.5, 0.0, 0.0]
        config['hidden_neurons'] = [512,256]
        config['out_features'] = 11

        config['batch_size'] = 128
        config['epoch']=50
        config['activation']=F.relu
        config['learning_rate']=1e-3

    elif (dataset_name == 'imdb'):
        config['dataset_name'] = 'karate'
        config['labeled_only'] = True

        config['input_features'] = data.x.shape[1]
        config['dropout'] = [0, 0]
        config['hidden_neurons'] = [100,100]
        config['out_features'] = len(data.classname)
        config['batch_size'] = 32
        config['activation'] = F.relu
        config['learning_rate'] = 1e-2

    else:
        print('Dataset not found')
        sys.exit(0)

    return config

def get_dataset(config):
    if (config['dataset_name'] == 'CVE'):
        from Dataset.CVE_Dataset import data_G
        dataset = data_G(config['input_path'],config['labeled_only'])  # true for labeled only dataset
    elif (config['dataset_name'] == 'karate'):
        from Dataset.Karate_Dataset import data_G_karate
        dataset = data_G_karate(config['input_path'])
    elif (config['dataset_name'] == 'imdb'):
        from Dataset.Imdb_read import data_G_imdb
        dataset = data_G_imdb(config)
    else:
        sys.exit(0)

    return dataset

def load_data(config):
    dataset=get_dataset(config)

    x=torch.tensor(dataset.Feature,dtype=torch.float)
    y=torch.tensor(dataset.Label,dtype=torch.long)
    G=dataset.Graph
    edges=G.edges()

    u = [src for src,dst in edges]
    v = [dst for src, dst in edges]

    try:
        if(config['directed']):
            tmp_u=list(u)
            u.extend(v)
            v.extend(tmp_u)
    except:
        print("Considering undirected graph")

    train_index=torch.tensor(dataset.train_index,dtype=torch.long)
    test_index=torch.tensor(dataset.test_index,dtype=torch.long)
    val_index=torch.tensor(dataset.val_index,dtype=torch.long)

    edge_index=torch.tensor([u,v],dtype=torch.long)
    Dataset = namedtuple('Dataset', field_names=['x', 'y', 'edge_index', 'train_index', 'test_index', 'val_index','classname'])
    data=Dataset(x=x,y=y,edge_index=edge_index,train_index=train_index,test_index=test_index,val_index=val_index,classname=dataset.classname)

    return data

def load_data_DGL(config):
    dataset = get_dataset(config)

    x = torch.tensor(dataset.Feature, dtype=torch.float)
    y = torch.tensor(dataset.Label, dtype=torch.long)

    G = dgl.DGLGraph()
    G.from_networkx(dataset.Graph)


    train_index = torch.tensor(dataset.train_index, dtype=torch.long)
    test_index = torch.tensor(dataset.test_index, dtype=torch.long)
    val_index = torch.tensor(dataset.val_index, dtype=torch.long)

    Dataset = namedtuple('Dataset', field_names=['x', 'y', 'Graph', 'train_index', 'test_index', 'val_index','classname'])

    data = Dataset(x=x, y=y, Graph=G, train_index=train_index, test_index=test_index, val_index=val_index,classname=dataset.classname)

    return data

if __name__ == '__main__':
    config=getSettings('karate')
    config['input_path']='/Users/siddharthashankardas/Purdue/Dataset/Karate/'
    data=load_data(config)

    print(data)
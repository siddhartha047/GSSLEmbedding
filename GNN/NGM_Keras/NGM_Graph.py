from Dataset.Karate_Dataset import path, process_karate_club
from networkx import read_gpickle,write_gpickle
import numpy as np

class EmbeddingsGraph:

    def __init__(self):
        self.graph = read_gpickle(path+'Graph.gpickle')

        self.x_train = np.load(path + 'x_train.npy')
        self.y_train = np.load(path + 'y_train.npy')
        self.x_test = np.load(path + 'x_test.npy')
        self.y_test = np.load(path + 'y_test.npy')
        self.features=np.load(path+'features.npy')
        self.labels=np.load(path+'labels.npy')
        self.all_labels=np.array(self.labels)

        #make labels only availble for training mutate the test labels
        for i in self.x_test:
            self.labels[i]=[0, 0]
        self.updated_x_train=self.x_train


        self.X_train_feature=np.load(path + 'training_features.npy')
        self.Y_train_label=np.load(path + 'training_labels.npy')

        self.X_test_feature = np.load(path + 'test_features.npy')
        self.Y_test_label = np.load(path + 'test_labels.npy')

        #nx.set_edge_attributes(network, 1.0, name='weight')
        # network.edges.data('weight', default=1.0)
        # print(network.edges(data=True))
        #print(self.graph.get_edge_data(0, 5)['weight'])


def save_graph():
    data=process_karate_club()

    X_train = data.Feature[list(data.x_train)]
    Y_train = np.zeros((len(data.y_train), 2))
    for i in range(len(Y_train)): Y_train[i, data.y_train[i]] = 1

    X_test = data.Feature[list(data.x_test)]
    Y_test = np.zeros((len(data.y_test), 2))
    for i in range(len(Y_test)): Y_test[i, data.y_test[i]] = 1

    write_gpickle(data.Graph, path + 'Graph.gpickle')
    np.save(path + 'x_train', data.x_train)
    np.save(path + 'y_train', data.y_train)
    np.save(path + 'x_test', data.x_test)
    np.save(path + 'y_test', data.y_test)
    np.save(path + 'features', data.Feature)

    n=len(data.Label)
    onehot_labels = np.zeros((n, 2))
    i = 0
    for node in range(n):
        l = data.Label[node]
        onehot_labels[i, int(l)] = 1.0
        i += 1

    np.save(path + 'labels',onehot_labels)

    #print(onehot_labels)

    np.save(path + 'training_features',X_train)
    np.save(path + 'training_labels',Y_train)

    np.save(path + 'test_features', X_test)
    np.save(path + 'test_labels', Y_test)

    return data

if __name__ == '__main__':
    save_graph()

    #G = EmbeddingsGraph()

    # Dataset = namedtuple('Dataset', field_names=['x_train', 'y_train', 'x_test', 'y_test', 'Graph'])
    # plot_karate_club(Dataset(G.x_train,G.y_train,G.x_test,G.y_test,G.graph))
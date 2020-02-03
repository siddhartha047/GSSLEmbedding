import re
import gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import csv
import string
import json as jsn
from scipy import io
import pickle
import numpy as np

# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
stem = PorterStemmer()

stop_words = stopwords.words('english')

def processText(text):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    #remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ##Convert to list from string
    text = text.split()

    ##Stemming
    ps = PorterStemmer()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = [word for word in text if len(word)>2]
    # text = " ".join(text)
    return text

def processTextParagraph(text):
    return " ".join(processText(text))

def num(s):
    try:
        return float(s)
    except ValueError:
        return 0

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)


def save_data(data,data_vector,data_rating,output_file,output_label,output_data,comment=""):
    print("Started Writing data")
    pickle.dump(data, open(output_data, "wb"))

    print("Writing vector as mtx file")
    io.mmwrite(output_file, data_vector, comment=comment)

    f = open(output_label, 'w')
    f.write("%d\n" % len(data_rating))

    print("Writing Class label")
    with open(output_label, 'a') as f:
        for item in data_rating:
            f.write("%s\n" % item)
    f.close()

def save_data_mat(home_dir,data,data_vector,data_rating):
    import scipy as sp
    sp.io.savemat(home_dir+"data_mat.mat",mdict={'data': data})
    sp.io.savemat(home_dir + "data_vector_mat.mat", mdict={'data_vector': data_vector})
    sp.io.savemat(home_dir + "data_rating_mat.mat", mdict={'data_rating': data_rating})

def save_data_txt(home_dir,data,data_vector,data_rating):
    np.savetxt(home_dir + "data_txt.txt", data, delimiter='\t')
    np.savetxt(home_dir + "data_vector_txt.txt", data_vector, delimiter='\t')
    np.savetxt(home_dir + "data_rating_txt.txt", data_rating, delimiter='\t')

def save_data_numpy(home_dir,data,data_vector,data_rating):
    np.save(home_dir+"data_np",data)
    np.save(home_dir+"data_vector_np",data_vector)
    np.save(home_dir+"data_rating_np",data_rating)

def load_data(home_dir):
    data=np.load(home_dir+"data_np.npy")
    data_rating = np.load(home_dir + "data_rating_np.npy")
    data_vector = np.load(home_dir + "data_vector_np.npy")

    return (data,data_rating,data_vector)

def create_graph(home_dir):
    (data,data_rating,data_vector)=load_data(home_dir)

    n=data.shape[0]
    indexes=np.random.choice(n, min(1000,n))
    print(indexes)

    print("Data count: ", data.shape)
    print("Vector count: ", data_vector.shape)
    print("Rating count: ", data_rating.shape)

    data=data[indexes]
    data_rating = data_rating[indexes]
    data_vector = data_vector[indexes]

    print("Data count: ", data.shape)
    print("Vector count: ", data_vector.shape)
    print("Rating count: ", data_rating.shape)

    from sklearn.neighbors import kneighbors_graph
    import networkx as nx
    import scipy as sp

    mode='distance'
    metric='cosine'
    #metric = 'euclidean'

    k=5
    sparse_graph = kneighbors_graph(data_vector, k, mode=mode, metric=metric, include_self=False)

    if (mode == 'distance'):
        #sparse_graph.data = 1.0 / (1.0 + sparse_graph.data)  # poincare
        #sparse_graph.data = 1.0 - sparse_graph.data
        sparse_graph.data = sparse_graph.data+1e-10
        #sparse_graph.eliminate_zeros()

    print('Saving graph ----')
    graph = home_dir + 'graph.npz'
    gephi = home_dir + "graph_gephi.gexf"

    sp.sparse.save_npz(graph, sparse_graph)
    print('Graph saving Done')

    custom_labels=[]
    for i in range(data_rating.shape[0]):
        custom_labels.append(str(data_rating[i])+"->"+data[i][:50])

    print(custom_labels)

    threshold=3.0
    if("Imdb" in home_dir): threshold=5.0

    for i in range(data_rating.shape[0]):
        if(data_rating[i]>threshold):
            data_rating[i]=1
        else:
            data_rating[i] = 0


    labels = dict(zip(range(data_rating.shape[0]), data_rating))
    G = nx.from_scipy_sparse_matrix(sparse_graph)
    nx.set_node_attributes(G, labels, 'labels')
    print("Writing gephi")
    nx.write_gexf(G, gephi)

    return


def load_model(model_name):
    from gensim.models.doc2vec import Doc2Vec
    import numpy as np

    model = Doc2Vec.load(model_name)
    vectors = np.zeros((model.corpus_count, model.vector_size), np.float)
    print(vectors.shape)
    for i in range(model.corpus_count):
        vectors[i] = model.docvecs[i]
    print(vectors.shape)

    return vectors


if __name__ == '__main__':
    # home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/Imdb/aclImdb/"
    # #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/Yelp/"
    # #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/DBpedia/dbpedia_csv/"
    # #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/AmazonReview/"
    # create_graph(home_dir)

    home_dir = "/Users/siddharthashankardas/Purdue/Dataset/Imdb/aclImdb/word2vec/"
    #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/Yelp/"
    #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/DBpedia/dbpedia_csv/"
    #home_dir = "/Users/sid/Purdue/Research/GCSSL/Dataset/AmazonReview/"
    #create_graph(home_dir)
    #load_model("Models/imdb_d2v.model")
    #data=[['This','is'],['not','meh']]
    data=np.random.rand(5,5)
    data_rating=np.random.rand(5,1)
    data_vector=np.random.rand(5,10)
    save_data_txt(home_dir, data,data_vector,data_rating)
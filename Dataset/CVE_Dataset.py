import numpy as np
import pickle
import gensim
import pandas as pd
from gensim.models import Word2Vec
from Dataset.CVE_path_lib import *
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from nltk import sent_tokenize
import json
import re
import timeit
import sys
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import csv
import random
from collections import namedtuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from graphviz import Digraph,Graph
from collections import OrderedDict
#import xlrd

random.seed(1)

model_name='CYBER'


# download('punkt') #tokenizer, run once
# download('stopwords') #stopwords dictionary, run once
# download('wordnet')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
stem = PorterStemmer()

stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def preprocess2(text):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ##Convert to list from string
    text = text.split()

    ##Stemming
    ps = PorterStemmer()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    #text = " ".join(text)
    return text

def preprocess3(text):
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
    # text = " ".join(text)
    return text


def cwe_filter(x):
    return list(map(int,re.findall('\d+', x)))

#https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary

def extract_cves(row):
    item=row['Observed Examples']
    cve_p = re.compile('CVE-\\d+-\\d+')
    results=cve_p.findall(item)
    #print(results)
    return results

def add_to_dictionary(row,CWEs):
    id=row['CWE-ID']
    cves=row['CVEs']

    for cve in cves:
        if(cve in CWEs):
            #print('cve already there->')
            items=CWEs[cve]
            if(id in items[0]):
                #print('CWE is already mapped correctly')
                continue
            else:
                (CWEs[cve])[0].append(id)
        else:
            CWEs[cve]=[[id]]

    return

def addmoredata(cwe_file,CWEs):
    cwe_c = pd.read_csv(cwe_file, delimiter=',', encoding='latin1')
    cwe_c = cwe_c[cwe_c['Observed Examples'].map(lambda x: x==x)]
    #print(cwe_c['Observed Examples'].head())
    cwe_c['CVEs']= cwe_c.apply(lambda row: extract_cves(row), axis=1)
    #print(cwe_c.head())

    # row1={'CWE-ID':10,'CVEs':[1,2,3]}
    # row2 = {'CWE-ID': 11, 'CVEs': [1, 2, 3,4]}
    # add_to_dictionary(row1,CWEs)
    # add_to_dictionary(row2,CWEs)
    cwe_c['CVEs'] = cwe_c.apply(lambda row: add_to_dictionary(row,CWEs), axis=1)

    return cwe_c

def process_cwe(take_redhat=True):
    # load the dataset form redhat
    CWEs=dict()

    if(take_redhat):
        CWEs = pd.read_csv(cwe_data, delimiter=',',quotechar='"',header=None,encoding='latin1')
        #print(CWEs.head())
        CWEs[1]=CWEs[1].apply(cwe_filter)
        print(CWEs.head())

        CWEs.set_index(0, drop=True, inplace=True)
        CWEs=CWEs.T.to_dict(orient='list')
    #print(CWEs.head())

    #print(CWEs)

    # addmore CWEs cvemitre

    addmoredata(cwe_rc,CWEs)
    addmoredata(cwe_ac, CWEs)
    addmoredata(cwe_dc, CWEs)

    #print(CWEs)
    #sys.exit(0)

    return CWEs

def map_cve_cwe(row,CWEs):

    if(row['Name'] in CWEs):
        return CWEs[row['Name']][0]
    return []


def getText(row):
    #text=str(row['Description'])+' '+str(row['References'])+' '+str(row['Phase'])+' '+str(row['Votes'])+' '+str(row['Comments'])
    text=str(row['Description'])+'. '+str(row['Comments'])

    return text

def apply_embedding(row,model):
    #print(row['Description'])
    doc=preprocess3(getText(row))
    #doc = getText(row).split()
    #print(doc)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word in model.wv.vocab]
    #print(doc)
    #print(np.mean(model.wv[doc], axis=0))
    return np.mean(model.wv[doc], axis=0)

# def apply_embedding(row,model):
#     #print(row['Description'])
#     doc=preprocess3(getText(row))
#     #print(doc)
#     doc = [word for word in doc if word in model.wv.vocab]
#     #print(doc)
#     #print(np.mean(model.wv[doc], axis=0))
#     return np.mean(model.wv[doc], axis=0)
def process_text(row,model):
    doc=preprocess3(getText(row))
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word in model.wv.vocab]
    doc = " ".join(doc)
    return doc

def apply_embedding2(row,model):
    if (row['Name'] == ''):
        return apply_embedding(row, model)
    return row['Embeddings']


def process_text2(row,model):
    if(row['Name']==''):
        return process_text(row,model)
    return row['PDescription']

global_model=None

def load_model(name):
    global global_model

    if(global_model!=None):
        return global_model

    if(name=='CYBER'):
        global_model = Word2Vec.load(model_path)

    elif(name=='GOOGLE'):
        global_model= gensim.models.KeyedVectors.load_word2vec_format(os.path.join(GOOGLE_NEWS), binary=True)

    elif(name=='GLOVE'):
        return []
    else:
        sys.exit('Not Defined yet')

    return global_model

def makeEmpty():
    return []

def process_cve(take_redhat=True):

    CWEs=process_cwe(take_redhat)
    #print(CWEs['CVE-2013-1942'])

    # load the dataset
    CVEs = pd.read_csv(cve_data+'.csv', delimiter=',',quotechar='"',header=0,encoding='latin1')
    CVEs = CVEs.fillna('')
    print(CVEs.head())

    # xl = pd.ExcelFile(cve_data+'.xls')
    # CVEs = xl.parse(0)
    # CVEs = pd.read_excel(cve_data+'.xls', sheet_name=0)
    # print(CVEs.head())
    # print(CVEs.count)
    # sys.exit(0)

    CVEs['CWEs']=CVEs.apply(lambda row: map_cve_cwe(row,CWEs),axis=1)

    model=load_model(model_name)

    CVEs['Embeddings'] = CVEs.apply(lambda row: apply_embedding(row,model), axis=1)
    CVEs['PDescription'] = CVEs.apply(lambda row: process_text(row, model), axis=1)
    CVEs['MDescription'] = CVEs.apply(lambda row: getText(row), axis=1)

    print(CVEs.head())

    CVEs.to_pickle(cve_all)

    # select only labeled example
    CVEs=CVEs[CVEs['CWEs'].map(lambda x: len(x)) > 0]
    print(CVEs.head())

    CVEs.to_pickle(cve_labeled)

    return CVEs

def load_cve(filename):
    print('Loading CVEs : {0}'.format(filename))
    CVEs=pd.read_pickle(filename)
    #print(CVEs.head())

    return CVEs

def get_cwes(filename):
    print('Loading CWEs_RC : {0}'.format(filename))
    cwe_c = pd.read_csv(filename, delimiter=',', quotechar='"', encoding='latin1')
    #print(cwe_c.head())
    cwe_ids = cwe_c['CWE-ID'].tolist()
    print(cwe_ids)
    return cwe_ids

def CWE_mapping(CWEs,cluster_label, view):

    cwe_ids=get_cwes(cwe_rc)
    cwe_ids.extend(get_cwes(cwe_dc))
    cwe_ids.extend(get_cwes(cwe_ac))

    #print(len(cwe_ids), cwe_ids)
    cwe_ids=set(cwe_ids)
    cwe_ids=list(cwe_ids)
    #print(len(cwe_ids), cwe_ids)

    ingore_cwe=[227, 592, 661]
    #cwe_ids.extend([227,592,661]) #have to put in dictionary manually

    cwe_cluster_map=cluster_cwes(cluster_label,view)

    # cwe_cluster_map[227]=664
    # cwe_cluster_map[592]=664
    # cwe_cluster_map[661]=664


    print(cwe_ids)
    cwe_ids = list(set(cwe_cluster_map.values()))
    print('Cluster numbers ',end=' : ')
    print(len(cwe_ids),end='->')
    print(cwe_ids)
    print('CWE to Cluster map ', end=' : ')
    print(cwe_cluster_map)

    n = len(cwe_ids)
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    index_cwe_map = dict(zip(list(range(n)),cwe_ids))

    print('Index to Cluster map ', end=' : ')
    print(index_cwe_map)

    np.save(cwe_cluster_number,cwe_ids)
    pickle.dump(cwe_map, open(cwe_map_file, "wb"))
    pickle.dump(index_cwe_map, open(index_cwe_map_file, "wb"))

    y=np.zeros((CWEs.shape[0],n),dtype=float)
    i=0

    labeled_mask=np.ones(CWEs.shape[0],dtype=int)

    print('Cluster to Index map ', end=' : ')
    print(cwe_map)

    print('Original CWEs ', end=' : ')
    print(CWEs.tolist())

    for row in CWEs.tolist():
        if not row:
            labeled_mask[i]=0
        for j in row:

            if(j in ingore_cwe):continue

            y[i][cwe_map[cwe_cluster_map[j]]]=1.0
        i+=1

    return y,labeled_mask,CWEs.tolist(),cwe_cluster_map,cwe_map,index_cwe_map

def distance(w1,w2):

    return 1.0/(1.0+similarity(w1,w2))

def similarity(w1, w2):

    return np.dot(gensim.matutils.unitvec(w1), gensim.matutils.unitvec(w2))

# def create_graph(CVEs,cluster_label=0,view=True,K=100):
#     x=CVEs['Embeddings']
#     y,labeled_mask,original_y,cwe_cluster_map,cwe_map,index_cwe_map=CWE_mapping(CVEs['CWEs'],cluster_label,view)
#
#     print(CVEs.head())
#
#     # print(CVEs['CWEs'].head())
#     # print(y[0])
#
#     x=x.values
#     l=x.shape[0]
#     x=np.concatenate(x).reshape(l,-1)
#
#     print(x.shape)
#     print(y.shape)
#     print(labeled_mask.shape)
#     print(np.sum(labeled_mask))
#
#     print('Saving Features')
#     np.save(cve_x,x)
#     np.save(cve_y,y)
#     np.save(cve_label_mask,labeled_mask)
#
#     # x=np.array([[1,1],[1,2],[1,2],[3,3]])
#     # print(x)
#     # K=2
#
#     print('Constructing Graph using K = {0} Neighbors -----'.format(K))
#     A = kneighbors_graph(x, K, mode='distance', metric='cosine',include_self=False)
#     #A = kneighbors_graph(x, K, mode='connectivity', metric='cosine',include_self=False)
#
#     print('Correcting edge weights -----')
#     A.data=1.0 / (1.0 + A.data) #poincare
#     # A.data = 1.0-A.data #normal distance
#
#     print('Saving graph ----')
#     sp.sparse.save_npz(cve_graph, A)
#     print('Graph saving Done')
#
#     if(use_labeled_only):
#         #onehot
#         multi_label=original_y
#         single_label=np.argmax(y, axis=1)
#         one_hot_label=y
#
#         print(multi_label)
#         print(single_label)
#         print(one_hot_label)
#
#         single_label=[index_cwe_map[i] for i in single_label]
#         print(single_label)
#
#         labels = dict(zip(range(len(single_label)), single_label))
#
#         G=nx.from_scipy_sparse_matrix(A)
#         nx.set_node_attributes(G, labels, 'labels')
#         #nx.set_node_attributes(G,x,'features')
#         print("Writing gephi")
#         nx.write_gexf(G, cve_gephi)
#
#
#     return

#keep edges between training data
#all other nodes will have K number of neighbors

def dummy_cve():
    CVEs = pd.DataFrame()
    descriptions=np.array(['Eval injection in Perl program using an ID that should only contain hyphens and numbers',
                            'SQL injection through an ID that was supposed to be numeric',
                            'lack of input validation in spreadsheet program leads to buffer overflows, integer overflows, array index errors, and memory corruption',
                            'not in blacklist for web server, allowing path traversal attacks when the server is run in Windows and other OSes',
                           'Arbitrary files may be read files via .. (dot dot) sequences in an HTTP request.',
                           'Directory traversal vulnerability in search engine for web server allows remote attackers to read arbitrary files via .. sequences in queries',
                           'Multiple FTP clients write arbitrary files via absolute paths in server responses',
                           'ZIP file extractor allows full path',
                           'Path traversal using absolute pathname'])

    # descriptions=np.array(['sql injection',
    #                        'injecting sql',
    #                        'sql whatever',
    #                        'buffer overflow attack',
    #                        'data overflow attack',
    #                        'buffer attack',
    #                        'file missing',
    #                        'file missing',
    #                        'common file errors'])

    CVEs['Description']=descriptions
    #print(CVEs.head())

    # x = np.array([[1, 2],
    #               [2, 4],
    #               [4, 6],
    #               [1, 0],
    #               [2, 0],
    #               [3, 0],
    #               [-1, -2],
    #               [-2, -4],
    #               [-4,-6]])
    #
    # CVEs['Embeddings']=x

    CWEs = np.array([
        [1],
        [1],
        [1],
        [2],
        [2],
        [2],
        [],
        [],
        []
    ])
    CVEs['CWEs']=CWEs

    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    train_index = np.array([0, 1, 2, 3, 4, 5])
    test_index = np.array([6, 7])
    val_index = np.array([8])

    model =load_model(model_name)
    CVEs['Embeddings'] = CVEs.apply(lambda row: apply_embedding(row, model), axis=1)

    x = CVEs['Embeddings']
    x = x.values
    l = x.shape[0]
    x = np.concatenate(x).reshape(l, -1)

    print(x)

    K1 = 3
    K2 = 2

    return CVEs,x,y,train_index,test_index,val_index,K1,K2


def getStatsCVEs(CVEs,train_index):
    stats = dict()
    i = 0
    for row in CVEs['CWEs'].tolist():
        if (i in train_index):
            for cwe in row:
                if (cwe not in stats):
                    stats[cwe] = [i]
                else:
                    stats[cwe].append(i)
        i += 1


    return stats

def getStatsY(y,train_index):
    stats = dict()
    i = 0
    for cwe in y:
        if (i in train_index):
            if (cwe not in stats):
                stats[cwe] = [i]
            else:
                stats[cwe].append(i)
        i += 1

    return stats

def getAllY(y):
    stats = dict()
    i = 0
    for cwe in y:
        if (cwe not in stats):
            stats[cwe] = [i]
        else:
            stats[cwe].append(i)
        i += 1

    return stats

def manifold_transform(x,k):
    print("NMF transformation")
    #X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    from sklearn.decomposition import NMF
    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(x)

    return W

# def CVE_FC_transform(x):
#
#     from CVE.CVE_FC.CVE_FC import Net
#     import torch
#     import torch.nn.functional as F
#
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using : ', device)
#
#     x=torch.tensor(x,dtype=torch.float)
#
#     model = Net()
#     model.load_state_dict(torch.load(cve_result_dir + 'CVE_FC',map_location=device))
#     model.eval()
#
#     model = model.to(device)
#     features = x.to(device)
#
#     new_features = model(features)
#     new_features = F.softmax(new_features, dim=1)
#
#     x=new_features.cpu().detach().numpy()
#
#     return x

def CVE_FC_TF3_transform(descriptions,max_words,sequence_length):
    # count feature
    from tensorflow import keras
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(descriptions)
    sequences = tokenize.texts_to_sequences(descriptions)
    #print(sequences[:5])

    from keras.preprocessing.sequence import pad_sequences

    data_sequence = pad_sequences(sequences, maxlen=sequence_length)

    word_index = tokenize.word_index
    print("unique words : {0}".format(len(word_index)))

    import tensorflow as tf
    from Dataset.CVE_path_lib import cve_result_dir

    save_file = cve_result_dir + 'CVE_FC_TF3.h5'
    model = tf.keras.models.load_model(save_file)

    y_softmax = model.predict(data_sequence)
    print(y_softmax.shape)

    return y_softmax


def sparse_graph(CVEs,x,y,train_index,val_index,test_index,K1,K2):

    #x=CVE_FC_transform(x)
    #x=manifold_transform(x,100)
    print("Transformed Feature shape: {}".format(x.shape))
# def sparse_graph():
#     # test graph dummy data start
#     CVEs,x,y,train_index,test_index,val_index,K1,K2=dummy_cve()

    #stats=getStatsCVEs(CVEs,train_index)
    #stats=getStatsY(y,train_index)
    stats=getStatsY(y,train_index)

    # stats=getAllY(y)
    # stats2=getStatsY(y,train_index)
    # print(stats2[9])
    #
    # print(stats)
    #
    # descriptions=CVEs['PDescription'].tolist()
    # original=CVEs['CWEs'].tolist()
    # cluster_id=CVEs['Cluster_CWE'].tolist()
    # cves_name=CVEs['Name'].tolist()
    #
    # for key, cwes in stats.items():
    #     print("key {0} -> {1}".format(key, len(cwes)))
    #
    #     if(key==9):
    #         print(cwes)
    #
    #         for i in cwes:
    #             print('{0}${1}${2}${3}${4}'.format(i,original[i],cluster_id[i],cves_name[i],descriptions[i]))
    #
    #
    #
    # sys.exit(0)

    n=x.shape[0]

    include_self=True
    mode='distance'
    #mode='connectivity'
    train_mode='connectivity'
    use_train=False
    metric='cosine'

    A = sp.sparse.lil_matrix((n, n))
    if (include_self):
        A = sp.sparse.lil_matrix(sp.sparse.eye(n))

    if(use_train==True):
        #best_neigh=np.zeros(n,dtype=float)
        best_neigh=dict()

        for key, cwes in stats.items():
            for ci in range(len(cwes)):
                i=cwes[ci]

                for cj in range(len(cwes)):
                    j=cwes[cj]

                    if(include_self==False and ci==cj):continue

                    best_neigh[j]=similarity(x[i],x[j])

                #print(best_neigh)
                #besk K selection

                best_neigh_tup=list(best_neigh.items())
                best_neigh_tup.sort(key=lambda x: x[1],reverse=True)
                best_k=[b_indexes for (b_indexes,value) in best_neigh_tup[:K1]]

                #best_k=best_neigh.argsort()[-min(K1,len(cwes)):][::-1]

                #print(best_k)
                #print(best_neigh)
                # if(key==9):
                #     print(i,best_k)

                for kj in best_k:
                    if (train_mode == 'distance'):
                        A[i, kj] = best_neigh[kj]
                    else:
                        A[i, kj] = 1.0

    #sys.exit(0)
    if(use_train):
        print("Using Training KNN done")

    A_others = kneighbors_graph(x, K2, mode=mode, metric=metric,include_self=include_self)

    if(mode=='distance'):
        #A_others.data = 1.0 / (1.0 + A_others.data)  # poincare
        A_others.data = 1.0-A_others.data
        A_others.eliminate_zeros()

    # print(A_others)
    # print("Before----")
    # print(A)

    if (use_train==True):
        A[test_index]=A_others[test_index]
        A[val_index] = A_others[val_index]

    # print("After----")
    # print(A)

    if(use_train==True):
        return A.tocsr()
    else:
        return A_others

    # A=A.tocsr()
    # labels = dict(zip(range(len(y)), y))
    # #G = nx.from_scipy_sparse_matrix(A)
    #
    # G = nx.from_scipy_sparse_matrix(A_others)
    # # print(G.edges())
    # # G=G.to_directed()
    # # print(G.edges())
    #
    # nx.set_node_attributes(G, labels, 'labels')
    # print("Writing gephi")
    # nx.write_gexf(G, cve_graph_dir+'dummy.gexf')
    # return A

def  feature_transform(CVEs):
    max_words=5000
    from tensorflow import keras
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)

    tokenize.fit_on_texts(CVEs['PDescription'])
    #"binary", "count", "tfidf", "freq"
    feature = tokenize.texts_to_matrix(CVEs['PDescription'].values,mode='binary')

    # print(tokenize.word_docs)
    # sys.exit(0)

    # for i in range(100):
    #     print(np.sum(feature[i]))

    return feature

#https://www.kaggle.com/mikhailborovinskikh/tfidf-keras
def feature_transform_tf_idf(CVEs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    max_words=5000

    description=CVEs['PDescription']

    count_vectorizer = TfidfVectorizer(ngram_range=(2, 2),
                                       max_df=0.5,
                                       max_features=max_words,
                                       tokenizer=nltk.word_tokenize,
                                       strip_accents='unicode',
                                       lowercase=True, analyzer='word', token_pattern=r'\w+',
                                       use_idf=True, smooth_idf=True, sublinear_tf=False,
                                       stop_words='english')
    bag_of_words = count_vectorizer.fit_transform(description)
    print(bag_of_words.shape)

    vocab = count_vectorizer.vocabulary_
    print(vocab)
    # sys.exit(0)

    from sklearn.feature_extraction.text import TfidfTransformer

    transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False)
    transformer_bag_of_words = transformer.fit_transform(bag_of_words)
    print(transformer_bag_of_words.shape)

    feature = count_vectorizer.transform(description)
    feature = np.array(transformer.transform(feature).todense())

    # print(feature)
    # print(feature.shape)
    # print(feature[0].shape)
    # sys.exit(0)

    return feature


def feature_tranform_count(CVEs):
    from sklearn.feature_extraction.text import CountVectorizer
    max_words=5000
    import nltk

    description = CVEs['PDescription']
    vectorizer = CountVectorizer(ngram_range=(3, 3),
                                       max_df=0.5,
                                       max_features=max_words,
                                       tokenizer=nltk.word_tokenize,
                                       strip_accents='unicode',
                                       lowercase=True, analyzer='word', token_pattern=r'\w+',
                                       stop_words='english')
    X = vectorizer.fit_transform(description)
    print(vectorizer.get_feature_names())
    #print(X[0])
    return X.toarray()

def distrubtion(train_index,y):
    n_y=np.argmax(y,axis=1)

    stats = {}
    i = 0
    for cwe in n_y:
        if(i in train_index):
            if (cwe not in stats):
                stats[cwe] = [i]
            else:
                stats[cwe].append(i)
        i += 1

    for key, items in stats.items():
        print("{0}->{1}".format(key,len(items)))

    return stats

def stokenize(text):
    '''text: list of text documents'''
    tokenized =  sent_tokenize(text)
    return tokenized

def wtokenize(text):
    '''text: list of text documents'''
    tokenized =  word_tokenize(text)
    return tokenized

def partition_sentence(text,k):
    text=text.split()
    results=[]
    p=round(len(text)/k)

    for i in range(p):
        results.append(" ".join(text[i*k:min((i+1)*k,len(text))]))

    return results

def data_augmentation(CVEs,train_index,y, labeled_mask, original_y,index_cwe_map):
    print("Before augmentation: ")
    stats=distrubtion(train_index,y)

    column=y.shape[1]

    max_item=np.max([len(items) for items in stats.values()])
    print(max_item)
    percent=0.25
    target=max_item*percent

    descriptions=CVEs['PDescription'].values
    new_o_y=[]
    new_original_y=[]
    newCVEs = pd.DataFrame(columns=['Name', 'Status', 'Description', 'References', 'Phase', 'Votes',
                                    'Comments', 'CWEs', 'Embeddings', 'PDescription','MDescription', 'Cluster_CWE'])

    #keys=[9,2,10,4]

    for key, items in stats.items():
        item_count=len(items)

        if(key==key):
        #if(key in keys):

            texts = descriptions[items]
            #print(texts)
            texts = ' '.join(texts)
            #texts=wtokenize(texts)
            #texts=stokenize(texts)
            texts =partition_sentence(texts,10)
            #print(texts)
            for i in range(int(target-item_count)):
                random.shuffle(texts)
                #print(texts)
                text=' '.join(texts[:20])
                #print(text)
                row=np.zeros(column,dtype=float)
                row[key]=1.00
                new_o_y.append(row)
                new_original_y.append([index_cwe_map[key]])



                dict={ 'Name': '',
                       'Status': '',
                       'Description': text,
                       'References': '',
                       'Phase': '',
                       'Votes': '',
                       'Comments': '',
                       'CWEs':[index_cwe_map[key]],'Embeddings':[],'PDescription':'','MDescription':text,'Cluster_CWE':index_cwe_map[key]}
                newCVEs=newCVEs.append(dict,ignore_index=True)

            # new_o_y=np.array(new_o_y)
            # print(new_description)
            # print(new_description.shape)
            # print(new_o_y)
            # print(new_o_y.shape)

    # print(CVEs.iloc[[0,-3,-2,-1]])
    # print(CVEs.tail(1).index)
    #
    # print(newCVEs.iloc[[0,-3,-2,-1]])
    # print(newCVEs.tail(1).index)

    CVEs=CVEs.append(newCVEs)

    model=load_model(model_name)

    CVEs['Embeddings'] = CVEs.apply(lambda row: apply_embedding2(row, model), axis=1)
    CVEs['PDescription'] = CVEs.apply(lambda row: process_text2(row, model), axis=1)

    # print(CVEs.iloc[[0, -3, -2, -1]])
    # print(CVEs.tail(1).index)

    original_y.extend(new_original_y)

    new_o_y = np.array(new_o_y)
    print(new_o_y.shape)
    print(y.shape)


    y=np.vstack((y,new_o_y))
    print(y.shape)

    new_train_indexes = [i for i in range(len(labeled_mask),y.shape[0])]
    train_index=np.append(train_index,new_train_indexes)

    print("Saving new training size {0}".format(train_index.shape))
    np.save(cve_train_index, train_index)

    labeled_mask=np.append(labeled_mask,[1 for i in range(len(new_train_indexes))])

    print("After augmentation: ")
    distrubtion(train_index, y)

    return (CVEs,train_index,y, labeled_mask, original_y)

def cwe_data_augmentation(CVEs,train_index,new_train_indexes,y, labeled_mask, original_y,index_cwe_map):
    print("Before augmentation: ")

    stats2 = distrubtion(train_index, y)
    stats=distrubtion(new_train_indexes,y)

    column=y.shape[1]
    max_item=np.max([len(items) for items in stats.values()])
    #max_item=np.max([len(items) for items in stats2.values()])
    print(max_item)
    sentence=True
    sentence_count=6
    k_count=10
    percent=1.00
    target=max_item*percent

    descriptions=CVEs['PDescription'].values

    print(target)

    #keys=[9,4,10]

    new_o_y=[]
    new_original_y=[]
    newCVEs = pd.DataFrame(columns=['Name', 'Status', 'Description', 'References', 'Phase', 'Votes',
                                    'Comments', 'CWEs', 'Embeddings', 'PDescription','MDescription', 'Cluster_CWE'])

    for key, items in stats.items():
        #item_count=len(items)
        item_count = len(stats2[key])

        #print(item_count)

        if(key==key):
        #if(key in keys):

            texts = descriptions[items]
            #print(texts)
            texts = '. '.join(texts)


            if(sentence==True):
                texts = stokenize(texts)
            else:
                #texts=wtokenize(texts)
                texts=partition_sentence(texts,k_count)

            #print(texts)
            for i in range(int(target-item_count)):
                random.shuffle(texts)
                #print(texts)
                text=' '.join(texts[:sentence_count])
                #print(text)

                #sys.exit(0)

                row=np.zeros(column,dtype=float)
                row[key]=1.00
                new_o_y.append(row)
                new_original_y.append([index_cwe_map[key]])

                dict={ 'Name': '',
                       'Status': '',
                       'Description': text,
                       'References': '',
                       'Phase': '',
                       'Votes': '',
                       'Comments': '',
                       'CWEs':[index_cwe_map[key]],'Embeddings':[],'PDescription':'','MDescription':text,'Cluster_CWE':index_cwe_map[key]}
                newCVEs=newCVEs.append(dict,ignore_index=True)
            # new_o_y=np.array(new_o_y)
            # print(new_description)
            # print(new_description.shape)
            # print(new_o_y)
            # print(new_o_y.shape)

    #sys.exit(0)
    # print(CVEs.iloc[[0,-3,-2,-1]])
    # print(CVEs.tail(1).index)
    #
    # print(newCVEs.iloc[[0,-3,-2,-1]])
    # print(newCVEs.tail(1).index)

    CVEs=CVEs.append(newCVEs)

    model=load_model(model_name)

    CVEs['Embeddings'] = CVEs.apply(lambda row: apply_embedding2(row, model), axis=1)
    CVEs['PDescription'] = CVEs.apply(lambda row: process_text2(row, model), axis=1)

    # print(CVEs.iloc[[0, -3, -2, -1]])
    # print(CVEs.tail(1).index)

    original_y.extend(new_original_y)

    new_o_y = np.array(new_o_y)
    print(new_o_y.shape)
    print(y.shape)


    y=np.vstack((y,new_o_y))
    print(y.shape)

    new_train_indexes = [i for i in range(len(labeled_mask),y.shape[0])]
    train_index=np.append(train_index,new_train_indexes)

    print("Saving new training size {0}".format(train_index.shape))
    np.save(cve_train_index, train_index)

    labeled_mask=np.append(labeled_mask,[1 for i in range(len(new_train_indexes))])

    print("After augmentation: ")
    distrubtion(train_index, y)

    return (CVEs,train_index,y, labeled_mask, original_y)


def getCWEText(row):
    # columns=['Name','Weakness Abstraction', 'Status', 'Description',
    #          'Extended Description',	'Related Weaknesses', 'Weakness Ordinalities',
    #          'Applicable Platforms',	'Background Details',	'Alternate Terms',	'Modes Of Introduction',	'Exploitation Factors',	'Likelihood of Exploit',	'Common Consequences',	'Detection Methods',	'Potential Mitigations',	'Observed Examples'	,'Functional Areas','Affected Resources',	'Taxonomy Mappings',	'Related Attack Patterns',	'Notes']

    columns = ['Name', 'Description','Extended Description']
    #columns = ['Name','Description']

    text=''
    for i in columns:
        if(row[i]==''):
            continue
        text+=str(row[i])+'. '

    return text

def add_cwe_training(CVEs,train_index,y, labeled_mask, original_y,index_cwe_map,cwe_cluster_map,cwe_map):
    #Cwe_map requires

    cwe_c = pd.read_csv(cwe_rc, delimiter=',', encoding='latin1')
    cwe_c['Description'] = cwe_c.apply(lambda row: getCWEText(row), axis=1)

    newCVEs = pd.DataFrame(columns=['Name', 'Status', 'Description', 'References', 'Phase', 'Votes',
       'Comments', 'CWEs', 'Embeddings', 'PDescription','MDescription', 'Cluster_CWE'])

    cweid=cwe_c['CWE-ID'].tolist()
    cwedescription=cwe_c['Description'].tolist()

    column=y.shape[1]
    new_o_y = []
    new_original_y = []

    for i in range(len(cweid)):

        dict={'Name':'',
              'Status':'',
              'Description':cwedescription[i],
              'References':'',
              'Phase':'',
              'Votes':'',
              'Comments':'',
              'CWEs': [cweid[i]],'Embeddings':[],'PDescription': [],'MDescription':cwedescription[i],'Cluster_CWE':cwe_cluster_map[cweid[i]]}
        newCVEs=newCVEs.append(dict,ignore_index=True)

        key= cwe_map[cwe_cluster_map[cweid[i]]]
        row = np.zeros(column, dtype=float)
        row[key] = 1.00
        new_o_y.append(row)
        new_original_y.append([index_cwe_map[key]])

    CVEs = CVEs.append(newCVEs)

    model = load_model(model_name)

    CVEs['Embeddings'] = CVEs.apply(lambda row: apply_embedding2(row, model), axis=1)
    CVEs['PDescription'] = CVEs.apply(lambda row: process_text2(row, model), axis=1)

    # print(CVEs.iloc[[0, -3, -2, -1]])
    # print(CVEs.tail(1).index)

    original_y.extend(new_original_y)

    new_o_y = np.array(new_o_y)
    # print(new_o_y.shape)
    # print(y.shape)

    y = np.vstack((y, new_o_y))
    # print(y.shape)

    new_train_indexes = [i for i in range(len(labeled_mask), y.shape[0])]
    train_index = np.append(train_index, new_train_indexes)

    print("Saving new training size {0}".format(train_index.shape))
    np.save(cve_train_index, train_index)

    labeled_mask = np.append(labeled_mask, [1 for i in range(len(new_train_indexes))])

    return (CVEs,train_index,y, labeled_mask, original_y,new_train_indexes)

def create_graph_train(CVEs,cluster_label=0,view=True, K1=100, K2=100):

    augmentation=True
    cwe_augmentation=False
    cwe_data_aug=True
    load_saved=False

    if(load_saved):
        if (use_labeled_only):
            dir = temp_dir + 'temp/'
        else:
            dir = temp_dir + 'temp_all/'

        labeled_mask = np.load(dir + 'label_mask.npy')
        train_index = np.load(dir + 'train_index.npy')
        test_index = np.load(dir + 'test_index.npy')
        val_index = np.load(dir + 'val_index.npy')
        y = np.load(dir + 'y.npy')
        original_y = np.load(dir + 'original_y.npy',allow_pickle=True)
        cwe_cluster_map=pickle.load(open(dir + 'cwe_cluster_map', "rb"))
        cwe_map=pickle.load(open(dir + 'cwe_map', "rb"))
        index_cwe_map=pickle.load(open(dir + 'index_cwe_map', "rb"))

        CVEs = pd.read_pickle(dir+'CVEfile')

    else:
        print(CVEs.head())
        #get true labels
        y,labeled_mask,original_y,cwe_cluster_map,cwe_map,index_cwe_map=CWE_mapping(CVEs['CWEs'],cluster_label,view)

        #store true labels
        y_one = np.argmax(y, axis=1)
        y_true = [index_cwe_map[i] for i in y_one]
        CVEs['Cluster_CWE']=y_true

        #print(CVEs['Cluster_CWE'].value_counts())
        #(train_index, val_index, test_index) = stats_data(CVEs)
        (train_index, val_index, test_index) = stats_data_index(CVEs,labeled_mask)

        # #add training data from CWE file
        if(cwe_augmentation==True):

            (CVEs, train_index, y, labeled_mask, original_y,new_train_indexes) = add_cwe_training(CVEs, train_index, y, labeled_mask,original_y, index_cwe_map,cwe_cluster_map,cwe_map)

            if(cwe_data_aug==True):
                (CVEs, train_index, y, labeled_mask, original_y) = cwe_data_augmentation(CVEs, train_index,new_train_indexes, y, labeled_mask,original_y, index_cwe_map)

        if(augmentation==True):
            (CVEs,train_index,y, labeled_mask, original_y)=data_augmentation(CVEs,train_index,y, labeled_mask, original_y,index_cwe_map)

        ###
        if(use_labeled_only):
            dir = temp_dir + 'temp/'
        else:
            dir = temp_dir + 'temp_all/'

        np.save(dir + 'label_mask',labeled_mask)
        np.save(dir + 'y',y)
        np.save(dir + 'original_y',original_y)
        np.save(dir + 'train_index',train_index)
        np.save(dir + 'test_index',test_index)
        np.save(dir + 'val_index',val_index)
        pickle.dump(cwe_cluster_map, open(dir+'cwe_cluster_map', "wb"))
        pickle.dump(cwe_map, open(dir+'cwe_map', "wb"))
        pickle.dump(index_cwe_map, open(dir+'index_cwe_map', "wb"))
        CVEs.to_pickle(dir + 'CVEfile')
        ##

    #######
    y_one = np.argmax(y, axis=1)
    y_true = [index_cwe_map[i] for i in y_one]

    #get train and test data
    #what will be the feature
    x = CVEs['Embeddings']
    x=x.values
    l=x.shape[0]
    x=np.concatenate(x).reshape(l,-1)

    #features=x
    #features = np.eye(len(labeled_mask))
    features=feature_transform(CVEs)
    #features=feature_transform_tf_idf(CVEs)
    #features=feature_tranform_count(CVEs)
    #features=CVE_FC_TF3_transform(CVEs['MDescription'].values,10000,1000)

    text_feature=CVEs['MDescription'].values
    #print(text_feature)

    print(train_index.shape)
    print(val_index.shape)
    print(test_index.shape)

    print('Text Feature shape', end=':'),print(text_feature.shape)
    print('Feature shape',end=':'),print(features.shape)
    print('X shape',end=':'),print(x.shape)
    print('Y shape',end=':'),print(y.shape)
    print('Labeled Mask',end=':'),print(labeled_mask.shape)
    print('Labeled Count',end=':'),print(np.sum(labeled_mask))

    print('Saving Features')
    np.save(cve_text,text_feature)
    np.save(cve_x,features)
    np.save(cve_y,y)
    np.save(cve_label_mask,labeled_mask)

    print("Saved features and exit")
    #sys.exit(0)


    print('Constructing Graph using train K1 {0}, K2 = {1} Neighbors -----'.format(K1,K2))
    A = sparse_graph(CVEs,features,y_one,train_index,val_index,test_index, K1,K2)

    # A = kneighbors_graph(x, K, mode='distance', metric='cosine',include_self=False)
    # A = kneighbors_graph(x, K, mode='connectivity', metric='cosine',include_self=False)
    # print('Correcting edge weights -----')
    # A.data=1.0 / (1.0 + A.data) #poincare
    # A.data = 1.0-A.data #normal distance

    print('Saving graph ----')
    sp.sparse.save_npz(cve_graph, A)
    print('Graph saving Done')

    #onehot
    multi_label=original_y
    single_label=np.argmax(y, axis=1)
    one_hot_label=y

    print(multi_label)
    print(single_label)
    print(one_hot_label)

    single_label=[index_cwe_map[i] for i in single_label]

    for i in range(len(single_label)):
        if(labeled_mask[i]<0.5):
            single_label[i]=0

    print(single_label)

    labels = dict(zip(range(len(single_label)), single_label))

    G=nx.from_scipy_sparse_matrix(A)
    nx.set_node_attributes(G, labels, 'labels')
    #nx.set_node_attributes(G,x,'features')
    print("Writing gephi")
    nx.write_gexf(G, cve_gephi)


    return


def stats_data(CVEs):
    stats={}
    i=0
    for row in CVEs['CWEs'].tolist():
        #print(row)
        for cwe in row:
            if(cwe not in stats):
                stats[cwe]=[i]
            else:
                stats[cwe].append(i)
        i+=1

    print(stats)

    # for key, val in stats.items():
    #     print("{0},{1}".format(key,val))

    pickle.dump(stats, open(stats_file, "wb"))

    # w = csv.writer(open("output.csv", "w"))
    # for key, val in stats.items():
    #     w.writerow([key, len(val)])

    # for key, val in stats.items():
    #     print("{0},{1}".format(key,len(val)))

    train_index = list()
    test_index = list()
    val_index = list()

    #split percentage
    train=0.6
    validation=0.1

    for key, val in stats.items():
        #print("{0},{1}".format(key,val))
        random.shuffle(val)

        l=len(val)
        t_length=int(l*train)
        v_length=int(l*validation)

        train_index.extend(val[:t_length])
        val_index.extend(val[t_length:t_length+v_length])
        test_index.extend(val[t_length+v_length:])

        #print("{0},{1}".format(key, val))

    train_index=np.array(train_index)
    val_index=np.array(val_index)
    test_index=np.array(test_index)

    print(train_index.shape,train_index)
    print(val_index.shape,val_index)
    print(test_index.shape, test_index)

    np.save(cve_train_index,train_index)
    np.save(cve_test_index,test_index)
    np.save(cve_val_index,val_index)

    print("Training Test and Validation data are saved")

    return (train_index,val_index,test_index)

def stats_data_index(CVEs,labeled_mask):
    stats={}
    i=0
    for cwe in CVEs['Cluster_CWE'].tolist():
        if(labeled_mask[i]>0):
            if(cwe not in stats):
                stats[cwe]=[i]
            else:
                stats[cwe].append(i)
        i+=1

    print(stats)

    # for key, val in stats.items():
    #     print("{0},{1}".format(key,len(val)))
    #
    # sys.exit(0)

    pickle.dump(stats, open(stats_file, "wb"))

    # w = csv.writer(open("output.csv", "w"))
    # for key, val in stats.items():
    #     w.writerow([key, len(val)])

    # for key, val in stats.items():
    #     print("{0},{1}".format(key,len(val)))

    train_index = list()
    test_index = list()
    val_index = list()

    #split percentage
    train=0.7
    validation=0.1

    for key, val in stats.items():
        #print("{0},{1}".format(key,val))
        random.shuffle(val)

        l=len(val)
        t_length=int(l*train)
        v_length=int(l*validation)

        train_index.extend(val[:t_length])
        val_index.extend(val[t_length:t_length+v_length])
        test_index.extend(val[t_length+v_length:])

        #print("{0},{1}".format(key, val))

    train_index=np.array(train_index)
    val_index=np.array(val_index)
    test_index=np.array(test_index)

    print(train_index.shape,train_index)
    print(val_index.shape,val_index)
    print(test_index.shape, test_index)

    np.save(cve_train_index,train_index)
    np.save(cve_test_index,test_index)
    np.save(cve_val_index,val_index)

    print("Training Test and Validation data are saved")

    return (train_index,val_index,test_index)

def data_G(labeled_only=True):

    dir=base_dir + 'cve_graph_all/'

    if(labeled_only):
        dir = base_dir+'cve_graph_labeled/'

    if os.uname()[1].find('purdue')!=-1:
        if (labeled_only == False):
            dir = '/scratch/gilbreth/das90/GNN_data/cve_graph_all/'

    edges = sp.sparse.load_npz(dir + 'CVE_CWE_graph.npz')
    features=np.load(dir+'features.npy')
    descriptions=np.load(dir+'descriptions.npy',allow_pickle = True)
    cwe_cluster_ids=np.load(dir+'cwe_cluster.npy',allow_pickle = True)
    labels=np.load(dir+'labels.npy')
    label_mask=np.load(dir+'label_mask.npy')
    train_index=np.load(dir+'train_index.npy')
    test_index=np.load(dir+'test_index.npy')
    val_index=np.load(dir+'val_index.npy')

    G=nx.from_scipy_sparse_matrix(edges)

    N=nx.normalized_laplacian_matrix(G, weight='weight')
    G=nx.from_scipy_sparse_matrix(N)

    #G=nx.from_scipy_sparse_matrix()


    #nx.set_edge_attributes(G, 1.0, name='weight')

    # #features = np.eye(len(label_mask))
    # features = np.array(nx.adjacency_matrix(G).todense())
    # # sys.exit(0)

    Dataset = namedtuple('Dataset', field_names=['train_index', 'test_index', 'val_index','Graph', 'Feature', 'Label','Label_mask','Description','classname'])

    data=Dataset(train_index=train_index,test_index=test_index, val_index=val_index, Graph=G,Feature=features,Label=labels,Label_mask=label_mask,Description=descriptions,classname=cwe_cluster_ids)

    return data

def data_GSAGE(labeled_only=True):

    dataset=data_G(labeled_only)
    G=dataset.Graph


    n = len(dataset.Label_mask)

    for i in range(n):
        if (i in dataset.train_index):
            G.node[i]['test'] = False
            G.node[i]['val'] = False
        elif(i in dataset.val_index):
            G.node[i]['test'] = False
            G.node[i]['val'] = True
        else:
            G.node[i]['test'] = True
            G.node[i]['val'] = False

    for edge in G.edges():
        G[edge[0]][edge[1]]['train_removed'] = False
        G[edge[0]][edge[1]]['test_removed'] = False

    feats = dataset.Feature

    id_map = {i: i for i in range(n)}

    one_hot = dataset.Label

    class_map = {i: list(one_hot[i]) for i in range(n)}

    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # print(degree_sequence)
    # dmax = max(degree_sequence)
    # print(dmax)

    walks = []

    #nx.set_edge_attributes(G, 1.0, name='weight')
    # print(G[0])
    # sys.exit(0)

    print(len(G.nodes()))
    print(feats.shape)
    print(len(id_map))
    print(len(walks))
    print(len(class_map))
    print(G[0])
    #print(G[0][1])
    # print(len(G.nodes()))
    # print(feats.shape)
    # print(len(id_map))
    # print(len(walks))
    # print(len(class_map))

    return G, feats, id_map, walks, class_map, dataset.classname


def visualize_data(labeled_only):

    data=data_G(labeled_only)
    print(data)

    #plt.figure(1, figsize=(30, 20), )
    y = np.argmax(data.Label, axis=1)
    print(y)

    ##plot_pca
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data.Feature)
    y = np.argmax(data.Label, axis=1)

    plt.scatter(x_pca[:, 0], x_pca[:, 1], s=100, c=y, alpha=0.2)
    plt.show()

    # # ### Plot t-SNE
    # X_tsne = TSNE(n_components=2, verbose=2).fit_transform(X)
    #
    # plt.figure(1, figsize=(30, 20), )
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, c=y, alpha=0.2)
    # plt.show()



    return

def open_cwes(filename):
    print('Loading CWEs_RC : {0}'.format(filename))
    cwe_c = pd.read_csv(filename, delimiter=',', quotechar='"', encoding='latin1')
    return cwe_c

def gviz():
    dot = Digraph(comment='The Round Table')
    dot.node('A', 'King Aurthuer')
    dot.node('B', 'Sir Bedevre the wise')
    dot.node('L', 'Sir Lancelot the brave')

    dot.edges(['AB', 'AL'])
    #dot.edge('B', 'L', constraint='false')

    print(dot.source)

    cve_graphviz=cve_graph_dir+'cwe_hierarchy.gv'
#    dot.format='svg'

    dot.render(cve_graphviz,view=True)

    return

def cwe_child(row):
    item = str(row['Related Weaknesses'])
    cve_p = re.compile('ChildOf:CWE ID:\\d+')
    results = cve_p.findall(item)
    item=''.join(results)
    cve_p = re.compile('\\d+')
    results = cve_p.findall(item)
    results = list(map(int, results))
    results=list(OrderedDict.fromkeys(results)) #preserve order
    #results=list(set(results)) #order not preserve

    # print(str(row['CWE-ID'])+'->', end="")
    # print(results)
    if(len(results)>0):
        return results[0]
    else:
        return -1

    #return results

def cluster_cwes(cluser_label,view):
    # gviz()
    cwe_c=open_cwes(cwe_rc)


    cwe_c['parent'] = cwe_c.apply(lambda row: cwe_child(row), axis=1)
    #print(cwe_c.head())

    nodes=cwe_c['CWE-ID'].tolist()
    parents=cwe_c['parent'].tolist()
    names=cwe_c['Name'].tolist()

    print("Clustering on label {0}".format(cluser_label))

    # nodes=[1,2,3,4,5,6,7,8]
    # parents=[-1,1,2,2,1,-1,6,7]
    group_map = create_group(nodes, parents, cluser_label)

    dot = Digraph(comment='Research Concepts Hierarchy',engine='dot',node_attr={'shape': 'box'})
    dot.graph_attr['rankdir'] = 'LR'
    #dot=Graph(format='png')

    for i in range(len(nodes)):
    #for i in range(5):
        if(group_map[nodes[i]]==nodes[i]):
            dot.node(str(nodes[i]), "CWE-ID "+str(nodes[i])+":"+names[i],color='red')
        else:
            dot.node(str(nodes[i]), "CWE-ID " + str(nodes[i]) + ":" + names[i])


    for i in range(len(nodes)):
    #for i in range(5):
        if(parents[i]>0):
            dot.edge(str(parents[i]),str(nodes[i]))

    #print(dot.source)

    cve_graphviz = cve_graph_dir + 'cwe_hierarchy.gv'
    dot.format='pdf'
    dot.render(cve_graphviz, view=view)
    #dot.render(cve_graphviz)

    return group_map

def update(cwe_id,parent_id,index_cwe_map,cwe_index_map,parents,group_map):

    if(parent_id==-1):
        return cwe_id

    grandparent_id=parents[cwe_index_map[parent_id]]
    group_id=update(parent_id, grandparent_id, index_cwe_map, cwe_index_map, parents,group_map)
    group_map[cwe_id]=group_id

    return group_id

def collpase_all(parent_id,cluser_id,group_map,nodes,parents):

    for i in range(len(parents)):
        if(parents[i]!=parent_id):
            continue
        #got child
        child_id=nodes[i]
        group_map[child_id]=cluser_id
        collpase_all(child_id,cluser_id,group_map,nodes,parents)
    return

def unfold(parent_id,group_map,nodes,parents,cluster_label):

    if(cluster_label==0):
        #start collasping
        collpase_all(parent_id,parent_id,group_map,nodes,parents)
        return

    for i in range(len(parents)):
        if(parents[i]!=parent_id):
            continue
        #got child
        child_id=nodes[i]
        unfold(child_id,group_map,nodes,parents,cluster_label-1)
    return

def create_group(nodes,parents,cluster_label):

    group_map = dict(zip(nodes, nodes))

    if (cluster_label == -1):
        return group_map

    index_cwe_map = dict(zip(range(len(nodes)), nodes))
    cwe_index_map = dict(zip(nodes,range(len(nodes))))
    #
    # print(group_map)
    # print(index_cwe_map)
    # print(cwe_index_map)

    for i in range(len(nodes)):
        update(nodes[i],parents[i],index_cwe_map,cwe_index_map,parents,group_map)
        #print(group_map)

    top_parents=list(set(group_map.values()))
    group_map=dict(zip(nodes,nodes))

    for i in top_parents:
        unfold(i,group_map,nodes,parents,cluster_label)


    return group_map

if __name__ == '__main__':
    # start=timeit.default_timer()
    # process_cve(True)
    # end=timeit.default_timer()
    # print("Embedding time: {0}ms".format(end-start))

    if(use_labeled_only):
        print('Only Labeled data')
    else:
        print('Using all data')


    start=timeit.default_timer()
    CVEs=load_cve(cve_file)
    create_graph_train(CVEs,cluster_label=0,view=False,K1=5, K2=11) #create graph, -1 for all, 0 root, 1 increasingly
    end=timeit.default_timer()
    print("Graph time: {0}".format(end-start))

    #sparse_graph()
    #cluster_cwes()
    #visualize_data(use_labeled_only)


def plot_performance(info):
    plt.figure()
    train_accs=info['train_accs']
    val_accs=info['val_accs']

    epochs=[i for i in range(len(train_accs))]

    plt.plot(epochs,train_accs)
    plt.plot(epochs,val_accs)

    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    title = info['name'] + ": Epoch vs Accuracy"

    plt.title(title)
    plt.legend(['Train Accuracy','Validation Accuracy'])
    plt.savefig(cve_result_dir+info['name']+'_accuracy.png')
    #plt.show()

def ptesting():
    # data=np.load(file1)
    # print(data.shape)
    # print(data[0])

    # model = Word2Vec.load(model_path)
    # print(model.wv['buffer'])
    # print(model.wv.similarity('buffer','overflow'))
    # model = Word2Vec.load(model_path)
    # text='the quick brown fox-bu jumps over the lazy dog12'
    # print(preprocess2(text))

    # load the dataset
    # CVEs = pd.read_csv(cve_data, delimiter=',',quotechar='"',header=0,encoding='latin1')
    # print(CVEs.head())

    #
    # #print(CVEs['Description'].head())
    # CVEs['word_count'] = CVEs['Description'].apply(lambda x: len(str(x).split(" ")))
    # print(CVEs[['Description', 'word_count']].head())
    #
    # print(CVEs.word_count.describe())

    # model = Word2Vec.load(model_path)
    # for word, vocab_obj in model.wv.vocab.items():
    #     print(word)
    # row={'Description':'buffer overflow attack asdfk'}
    # apply_embedding(row,model)
    # print(preprocess3('buffer overflow attack'))

    # #
    # A = csr_matrix([[0, 1, 0.5], [1, 0, 1], [1, 1, 0]])
    # single_label = np.array([1, 2, 3])
    # x = np.array([[1, 1], [2, 2], [3, 3]])
    # print(A)
    #
    # G = nx.from_scipy_sparse_matrix(A)
    # single_label=list(single_label)
    # print(single_label)
    # x=x.tolist()
    # print(x)
    #
    # labels=dict(zip(range(len(single_label)),single_label))
    # features=dict(zip(range(len(x)),x))
    # print(features)
    #
    # nx.set_node_attributes(G, labels, 'labels')
    # #nx.set_node_attributes(G, features, 'features')
    #
    # nx.write_gexf(G, cve_gephi)
    # print(G.node[0])
    # nx.draw_networkx(G,nx.spring_layout(G))
    # plt.show()

    return




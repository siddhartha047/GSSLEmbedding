import sys
import os
from Path import *

#vectorize configuration
VEC_config={
    "dataset_name":"reuters",
    "method":"TF_IDF",
    #"saving_format": "numpy", #numpy, mtx, mat, binary, txt
    "saving_format":['txt','mtx','mat','numpy'],
    "load_saved": False, #resume if possible in any stage (data loading, model loading etc.)
    "visualize":False
}


def vectorize():
    if(VEC_config['method']=="doc2vec"):
        from Vectorize import VectorizeDataset
        Doc2Vec_config = {
            "max_epochs": 1000,
            "vec_size": 300
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Doc2Vec_config)

    elif (VEC_config['method'] == "word2vec"):
        from Vectorize import VectorizeDataset
        Word2Vec_config = {
            "vec_size": 300,
            "pretrained_model": pretrained_model,
            "pretrained_model_name": "GOOGLE",
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Word2Vec_config)
        return

    elif(VEC_config['method']=="word2vec_avg"):
        from Vectorize import VectorizeDataset
        Word2Vec_config = {
            "vec_size": 300,
            "pretrained_model": pretrained_model,
            "pretrained_model_name": "GOOGLE",
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Word2Vec_config)

    elif (VEC_config['method'] == "TF_IDF"):
        from Vectorize import VectorizeDataset
        TF_IDF_config = {
            'max_features':3000
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, TF_IDF_config)

    elif (VEC_config['method'] == "LSI"):
        print("Not implemented yet")

    elif (VEC_config['method'] == "LDA"):
        print("Not implemented yet")

    elif (VEC_config['method'] == "FAST_TEXT"):
        print("Not implemented yet")

    else:
        print("improper method")
        sys.exit(0)

    return

GRAPH_data_config={
    "algorithm":'bmatch',
    "dataset_name":'reuters',
    "method":'TF_IDF',
    "multi_label":True,
    #"saving_format":["numpy","mat","gephi","mtx","txt"]  #avilable formats are: "saving_format": "numpy", #numpy, mtx, mat
    "saving_format":["gephi","mtx","txt"]  #avilable formats are: "saving_format": "numpy", #numpy, mtx, mat
}


def construct_graph():

    if(GRAPH_data_config['algorithm']=='knn'):
        from GraphConstruction.KNN.knn import KNN_construction
        KNN_config = {
            # "k": range(5,101,5),
            "k": [5],
            "mode": 'distance',  # connectivity will give 1,0
            "metric": "cosine",
            "include_self": False
        }
        KNN_construction(dataset_info[GRAPH_data_config['dataset_name']],GRAPH_data_config,KNN_config)

    elif(GRAPH_data_config['algorithm']=='bmatch'):
        from GraphConstruction.Bmatch.bmatch import bmatch_construction
        bMatching_config = {
            # "b":range(5,101,5),
            "b": [5],
            "mode": 'distance',  # connectivity will give 1,0
            "metric": "euclidean",  # used internally
            "include_self": False
        }
        bmatch_construction(dataset_info[GRAPH_data_config['dataset_name']], GRAPH_data_config, bMatching_config)


    else:
        print(GRAPH_data_config['algorithm']," -> The graph construction algorithm is not implemented yet")
        sys.exit(0)


    return

def learn_test():

    return

#running the program
RUN_vectorize=True
RUN_graph_construction=True
RUN_learn=False

if __name__ == '__main__':
    if(RUN_vectorize):
        vectorize()

    if(RUN_graph_construction):
        construct_graph()

    if(RUN_learn):
        learn_test()
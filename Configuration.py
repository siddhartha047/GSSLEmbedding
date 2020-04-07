from Path import *

#vectorize configuration
VEC_config={
    "dataset_name":"imdb",
    "method":"TF_IDF",
    #"saving_format": "numpy", #numpy, mtx, mat, binary, txt, mtx2
    "saving_format":['numpy'],
    "load_saved": False, #resume if possible in any stage (data loading, model loading etc.)
    "visualize":False
}

def vectorize():
    if(VEC_config['method']=="doc2vec"):
        from Vectorize import VectorizeDataset
        Doc2Vec_config = {
            "max_epochs": 500,
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
            'max_features':5000,
            'ngram':(1,1) #min max range
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, TF_IDF_config)

    elif (VEC_config['method'] == "LSI"):
        from Vectorize import VectorizeDataset
        LSI_config = {
            'tfidf_features':5000,
            'max_features': 300,
            'ngram': (1, 1)  # min max range
        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, LSI_config)

    elif (VEC_config['method'] == "LDA"):
        from Vectorize import VectorizeDataset
        LDA_config = {
            'ngram': (1, 1),  # min max range
            'features':3000, #use -1 to adapt this automatically
            'max_iter':10,
            'max_features': 100 #final number of feature

        }
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, LDA_config)

    elif (VEC_config['method'] == "FAST_TEXT"):
        print("Not implemented yet")

    else:
        print("improper vectorize name")
        sys.exit(0)

    return

GRAPH_data_config={
    "algorithm":'knn',
    "dataset_name":'imdb',
    "method":'TF_IDF',
    "multi_label":False,
    #"saving_format":["numpy","mat","gephi","mtx","txt"]  #avilable formats are: "saving_format": "numpy", #numpy, mtx, mat
    "saving_format":["numpy","gephi"]  #avilable formats are: "saving_format": "numpy", #numpy, mtx, mat
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

    elif (GRAPH_data_config['algorithm'] == 'complete'):
        from GraphConstruction.Complete.ScipyComplete import csr2graph
        complete_config = {
            "metric": "cosine",
            "include_self": False
        }
        csr2graph(dataset_info[GRAPH_data_config['dataset_name']], GRAPH_data_config, complete_config)

    elif(GRAPH_data_config['algorithm']=='bmatch'):
        from GraphConstruction.Bmatch.bmatch import bmatch_construction
        bMatching_config = {
            #"b":range(5,101,5),
            "b": [5,100],
            "mode": 'distance',  # connectivity will give 1,0
            "metric": "euclidean",  # used internally
            "max_iterations":[-1,-1], #this is given, -1 means default max_interations
            #"max_iterations": np.ones(len(range(5,101,5)))*-1,  # this is given, -1 means default max_interations
            "include_self": False
        }
        bmatch_construction(dataset_info[GRAPH_data_config['dataset_name']], GRAPH_data_config, bMatching_config)

    else:
        print(GRAPH_data_config['algorithm']," -> The graph construction algorithm is not implemented yet")
        sys.exit(0)


    return

LEARNING_data_config={
    "algorithm":'GCN_DGL',
    "dataset_name":'imdb',
    "directed":False,
    "labled_only": True, #load only data that has known labels
    "vectorize":"TF_IDF",
    "graph_algorithm":"knn",
    "train": 0.5,
    "val": 0.2}

def learn():
    if (LEARNING_data_config['algorithm'] == 'GCN_DGL'):
        from GNN.GCN_DGL.GCN_DGL_main import learn
        from GNN_configuration import getSettings,load_data_DGL

        LEARNING_data_config['input_path']=dataset_info[LEARNING_data_config['dataset_name']]['path']+LEARNING_data_config["vectorize"]+"/"
        data=load_data_DGL(LEARNING_data_config)

        gnn_settings = getSettings(LEARNING_data_config['dataset_name'],data)
        gnn_settings['output_path']=dataset_info[LEARNING_data_config['dataset_name']]['output_path']

        learn(gnn_settings,data)

    elif (LEARNING_data_config['algorithm'] == 'GSAGE_DGL'):
        from GNN.GSAGE_DGL.GSAGE_DGL_main import learn
        from GNN_configuration import getSettings,load_data_DGL

        LEARNING_data_config['input_path']=dataset_info[LEARNING_data_config['dataset_name']]['path']
        data=load_data_DGL(LEARNING_data_config)

        gnn_settings = getSettings(LEARNING_data_config['dataset_name'])
        gnn_settings['output_path']=dataset_info[LEARNING_data_config['dataset_name']]['output_path']

        learn(gnn_settings,data)

    elif (LEARNING_data_config['algorithm'] == 'GAT_DGL'):
        from GNN.GAT_DGL.GAT_DGL_main import learn
        from GNN_configuration import getSettings,load_data_DGL

        LEARNING_data_config['input_path']=dataset_info[LEARNING_data_config['dataset_name']]['path']
        data=load_data_DGL(LEARNING_data_config)

        gnn_settings = getSettings(LEARNING_data_config['dataset_name'])
        gnn_settings['output_path']=dataset_info[LEARNING_data_config['dataset_name']]['output_path']

        learn(gnn_settings,data)

    elif (LEARNING_data_config['algorithm'] == 'FC'):
        from GNN.FNN_PT_TF_Keras.FC import learn
        from GNN_configuration import getSettings,load_data_DGL

        LEARNING_data_config['input_path']=dataset_info[LEARNING_data_config['dataset_name']]['path']
        data=load_data_DGL(LEARNING_data_config)

        gnn_settings = getSettings(LEARNING_data_config['dataset_name'])
        gnn_settings['output_path']=dataset_info[LEARNING_data_config['dataset_name']]['output_path']

        learn(gnn_settings,data)

    else:
        print("this GNN is not incorporated yet")
        sys.exit(0)

    return

#running the program
RUN_vectorize=False
RUN_graph_construction=False
RUN_learn=True

if __name__ == '__main__':
    if(RUN_vectorize):
        vectorize()

    if(RUN_graph_construction):
        construct_graph()

    if(RUN_learn):
        learn()

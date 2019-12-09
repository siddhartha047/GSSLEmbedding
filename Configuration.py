import sys

dataset_info={
    "custom"    :{"name":"custom",
                  "path":"/"},
    "karate"    :{"name":"karate",
                  "path":"/"},
    "yelp"      :{"name":"yelp",
                  "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Yelp/",
                  "output_path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Yelp/"},
    "dbpedia"   :{"name":"dbpedia",
                  "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/DBpedia/dbpedia_csv/",
                  "output_path":"/Users/sid/Purdue/Research/GCSSL/Dataset/DBpedia/dbpedia_csv/"},
    "amazon"    :{"name":"amazon",
                  "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/AmazonReview/",
                  "output_path":"/Users/sid/Purdue/Research/GCSSL/Dataset/AmazonReview/"},
    "imdb"      :{"name":"imdb",
                  "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Imdb/aclImdb/",
                  "output_path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Imdb/aclImdb/"}
}

pretrained_model={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":"/Users/sid/Purdue/Research/GCSSL/Dataset/Model/cybersecurity/1million.word2vec.model"}
}


#vectorize configuration
VEC_config={
    "dataset_name":"imdb",
    "method":"word2vec",
    #"saving_format": "numpy", #numpy, mtx, mat
    "saving_format":"mat",
    "load_saved": True, #resume if possible in any stage (data loading, model loading etc.)
    "visualize":False
}

Doc2Vec_config={
    "max_epochs":1000,
    "vec_size":300
}

Word2Vec_config={
    "vec_size":300,
    "pretrained_model":pretrained_model,
    "pretrained_model_name":"GLOVE",
}

#graph construction configuration


#learning and test configuration


#running the program
RUN_vectorize=False
RUN_graph_construction=True
RUN_learn=False


def vectorize():
    if(VEC_config['method']=="doc2vec"):
        from Vectorize import VectorizeDataset
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Doc2Vec_config)

    elif (VEC_config['method'] == "word2vec"):
        from Vectorize import VectorizeDataset
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Word2Vec_config)
        return

    elif(VEC_config['method']=="word2vec_avg"):
        from Vectorize import VectorizeDataset
        VectorizeDataset.main(dataset_info[VEC_config['dataset_name']], VEC_config, Word2Vec_config)


    else:
        print("improper method")
        sys.exit(0)

    return

GRAPH_data_config={
    "algorithm":'knn',
    "dataset_name":'imdb',
    "method":'word2vec'
}

KNN_config={
        "k":5,
        "mode":'distance', #connectivity will give 1,0
        "metric":"cosine",
        "include_self":True
}

bMatching_config={

}

edge_cover_config={

}


def construct_graph():

    if(GRAPH_data_config['algorithm']=='knn'):
        from GraphConstruction.KNN.knn import KNN_construction
        KNN_construction(dataset_info[GRAPH_data_config['dataset_name']],GRAPH_data_config,KNN_config,save_gephi=True)

    else:
        print(GRAPH_data_config['algorithm']," -> The graph construction algorithm is not implemented yet")
        sys.exit(0)


    return

def learn_test():

    return

if __name__ == '__main__':
    if(RUN_vectorize):
        vectorize()

    if(RUN_graph_construction):
        construct_graph()

    if(RUN_learn):
        learn_test()

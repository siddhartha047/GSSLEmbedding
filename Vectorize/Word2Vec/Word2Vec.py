from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import os
import gensim
import numpy as np
import sys


def learn(data, model_info, vec_size, model_name,load_saved=False,visualize=False):

    if(load_saved==True and os.path.exists(model_name)==True):
        print("loading saved w2v model")
        model_2=Word2Vec.load(model_name)
    else:
        print("Training w2v model")
        if(model_info['name']=="GLOVE"):
            model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_info['path']), binary=False,
                                                                    encoding="ISO-8859-1")
        elif(model_info['name']=="GOOGLE"):
            model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_info['path']), binary=True)

        else:
            print("Model not implemented yet")
            sys.exit(0)

        # train model
        model_2 = Word2Vec(size=vec_size, min_count=1)
        model_2.build_vocab(data)
        total_examples = model_2.corpus_count

        model_2.build_vocab([list(model.vocab.keys())], update=True)

        if (model_info['name'] == "GLOVE"):
            model_2.intersect_word2vec_format(model_info['path'], binary=False, lockf=1.0)
        elif (model_info['name'] == "GOOGLE"):
            model_2.intersect_word2vec_format(model_info['path'], binary=True, lockf=1.0)


        model_2.train(data, total_examples=total_examples, epochs=model_2.epochs)
        model_2.save(model_name)
        print("Model Saved")

    X=np.zeros((len(data),vec_size),dtype=float)
    print(X.shape)

    for i in range(len(data)):
        X[i]=np.mean(model_2.wv[data[i]], axis=0)

    if(visualize==True):
        n=min(20,len(data))

        pca = PCA(n_components=2)
        result = pca.fit_transform(X[0:n,:])
        pyplot.scatter(result[:, 0], result[:, 1])

        words=[ " ".join(data[i]) for i in range(n)]
        print(words)

        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
        # # # fit a 2d PCA model to the vectors
        # # model_1 = Word2Vec(data, size=vec_size, min_count=1)
        # # X = model_2[model_1.wv.vocab]
        # # print(X.shape)
        #
        # pca = PCA(n_components=2)
        # result = pca.fit_transform(X)
        # # create a scatter plot of the projection
        # pyplot.scatter(result[:, 0], result[:, 1])
        #
        # words = list(model_1.wv.vocab)
        # for i, word in enumerate(words):
        #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        # pyplot.show()

    return X

pretrained_model={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/cybersecurity/1million.word2vec.model"}
}

if __name__ == '__main__':
    # data = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
    #              ['this', 'is', 'the', 'second', 'sentence'],
    #              ['yet', 'another', 'sentence'],
    #              ['one', 'more', 'sentence'],
    #              ['and', 'the', 'final', 'sentence']]


    # data = [['i', 'am', 'the', 'king'],
    #         ['this', 'is', 'the', 'second', 'sentence'],
    #         ['yet', 'another', 'sentence'],
    #         ['one', 'more', 'sentence'],
    #         ['and', 'the', 'final', 'sentence'],
    #         ['queen','you','are']]

    data = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    data = [text.lower() for text in data]

    data = [item.split() for item in data]
    print(data)

    data=np.array(data)

    model_name="GOOGLE"


    X=learn(data,pretrained_model['GLOVE'],vec_size=300,model_name='w2v.model',load_saved=False,visualize=True)
    print(X.shape)




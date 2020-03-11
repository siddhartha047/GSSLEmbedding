#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import multiprocessing
import smart_open
from sklearn.decomposition import PCA
from matplotlib import pyplot


def learn_d2v(data, model_name="d2v.model",max_epochs = 100, vec_size = 20):

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        #print(tagged_data)
        alpha = 0.025

        NUM_WORKERS = multiprocessing.cpu_count()
        print("using thread : ",NUM_WORKERS)

        import gensim
        model = gensim.models.doc2vec.Doc2Vec(
            vector_size=vec_size,
            min_count=2,
            epochs=max_epochs,
        workers=NUM_WORKERS)

        model.build_vocab(tagged_data)

        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # model = Doc2Vec(vector_size=vec_size,
        #                 alpha=alpha,
        #                 min_alpha=0.025,
        #                 min_count=1, #set this 2
        #                 dm=1,
        #                 workers=NUM_WORKERS)
        #
        # model.build_vocab(tagged_data)
        #
        # for epoch in range(max_epochs):
        #         print('iteration {0}'.format(epoch))
        #         model.train(tagged_data,
        #                     total_examples=model.corpus_count,
        #                     epochs=model.iter)
        #         # decrease the learning rate
        #         model.alpha -= 0.001
        #         # fix the learning rate, no decay
        #         model.min_alpha = model.alpha

        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))

        model.save(model_name)
        print("Model Saved")

        return model



def sanity_checking(model,data):

    # from pprint import pprint
    # pprint(data[0:min(10,len(data))])

    ranks = []
    second_ranks = []
    for doc_id in range(len(data)):
        inferred_vector = model.infer_vector(data[doc_id].split())
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

        # print([int(docid) for docid, sim in sims])
        # print(doc_id)
        # print(sims)

        rank = [int(docid) for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])


        # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(data[doc_id].split())))
        # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(data[int(sims[index][0])].split())))

    import collections

    counter = collections.Counter(ranks)
    print(counter)

    # Pick a random document from the corpus and infer a vector from the model
    import random

    print("Random 5 quality check")
    for i in range(5):
        doc_id = random.randint(0, len(data) - 1)

        # Compare and print the second-most-similar document
        print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(data[doc_id].split())))
        sim_id = second_ranks[doc_id]
        print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(data[int(sim_id[0])].split())))

    return

def get_vector(data,model_name,max_epochs,vec_size,use_saved=False,sanity=False,visualize = False):
    import numpy as np

    if(use_saved==True and os.path.exists(model_name)==True):
        model = Doc2Vec.load(model_name)
    else:
        model = learn_d2v(data, model_name, max_epochs=max_epochs, vec_size=vec_size)


    data_vector = np.zeros((model.corpus_count, model.vector_size), np.float)
    print(data_vector.shape)
    for i in range(model.corpus_count):
        data_vector[i] = model.docvecs[i]
    print(data_vector.shape)

    if (visualize == True):
        n = min(20, len(data))
        pca = PCA(n_components=2)
        result = pca.fit_transform(data_vector[0:n, :])
        pyplot.scatter(result[:, 0], result[:, 1])

        #words = [data[i][0:min(20,len(data[i]))] for i in range(n)]
        words = [data[i] for i in range(n)]
        print(words)

        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()

    if(sanity==True):
        sanity_checking(model,data)


    return data_vector

if __name__ == '__main__':
        # data = ["I love machine learning. Its awesome.",
        #         "I love coding in python",
        #         "I love building chatbots",
        #         "they chat amagingly well"]

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

        data=[text.lower() for text in data]

        ##return model
        #model=learn_d2v(data,"Models/test_d2v.model",max_epochs=400,vec_size=50)
        # sanity_checking(model, data)

        ##load model
        # model=Doc2Vec.load("Models/test_d2v.model")
        # sanity_checking(model, data)

        ##return vector only and visualize
        data_vector=get_vector(data, "Models/test_d2v.model", max_epochs=400, vec_size=50, use_saved=False, sanity=True, visualize=True)



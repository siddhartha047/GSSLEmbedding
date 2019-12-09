from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

def main():
    # define training data
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                ['this', 'is', 'the', 'second', 'sentence'],
                ['yet', 'another', 'sentence'],
                ['one', 'more', 'sentence'],
                ['and', 'the', 'final', 'sentence']]

    # train model
    model_1 = Word2Vec(sentences, size=300, min_count=1)

    # fit a 2d PCA model to the vectors
    X = model_1[model_1.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model_1.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    model_2 = Word2Vec(size=300, min_count=1)
    model_2.build_vocab(sentences)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)
    model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)

    # fit a 2d PCA model to the vectors
    X = model_2[model_1.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model_1.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

if __name__ == '__main__':
    main()
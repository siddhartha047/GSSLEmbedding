import gensim

def get_data():
    text_corpus = [
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

    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

    return processed_corpus

def learn():

    data=get_data()
    print(data)

    from gensim import corpora

    dictionary = corpora.Dictionary(data)
    print(dictionary)

    print(dictionary.token2id)

    new_doc="human computer interaction"
    new_vec=dictionary.doc2bow(new_doc.lower().split())
    print(new_vec)

    bow_corpus=[dictionary.doc2bow(text) for text in data]
    print(bow_corpus)

    from gensim import models

    #train the model
    tfidf=models.TfidfModel(bow_corpus)
    words="system minors".lower().split()
    print(tfidf[dictionary.doc2bow(words)])

    from gensim import similarities

    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

    query_document = 'system engineering system'.split()
    query_bow=dictionary.doc2bow(query_document)
    print(query_bow)
    print(tfidf[query_bow])
    sims=index[tfidf[query_bow]]
    print(list(enumerate(sims)))

    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)

    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # img = mpimg.imread('run_core_concepts.png')
    # imgplot = plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    return

if __name__ == '__main__':
    learn()